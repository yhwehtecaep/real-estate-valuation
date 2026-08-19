"""
main.py
=======
Executable end-to-end workflow for the Real Estate Valuation & Pricing Model.

Run:
    python main.py

Environment
-----------
    export FRED_API_KEY="your_free_fred_api_key"   # optional but recommended;
                                                      # https://fred.stlouisfed.org/docs/api/api_key.html
Without a FRED key (or without live network access), the pipeline logs a
clear warning and falls back to a cached table of the same real published
FRED figures -- it never fabricates macro data. Live geocoding (OpenStreetMap
Nominatim + osmnx) degrades the same way: if unreachable, those optional
spatial columns are simply omitted rather than invented.

What it does
------------
1. Loads the real Ames, Iowa housing dataset (De Cock 2011).
2. Fetches real macro data (live FRED API, or cached real fallback) and
   attempts live geocoding of the real Neighborhood field (OpenStreetMap).
3. Splits off an IMMUTABLE 10% holdout test set (most recent real sales by
   date) BEFORE any tuning or cross-validation -- it is never touched again
   until the final unbiased evaluation in step 7.
4. Tunes XGBoost / LightGBM with Optuna, and picks a Ridge alpha, all under
   `GroupTimeSeriesSplit` spatial-temporal CV on the remaining 90% dev set.
5. Runs full out-of-fold CV for all three models on the dev set and prints
   a metrics comparison table (MAE, RMSE, MAPE, R^2).
6. Refits the best model on the FULL dev set and runs the `StressTester`
   suite (macro shock, missing-attribute sensitivity, OOD physical
   features, residual diagnostics).
7. Evaluates that refit model ONCE on the untouched holdout set -- the
   final, unbiased, pre-deployment metric.
8. Demonstrates `screening_engine.DealScreener` on a real sample batch
   drawn from the holdout set (their own real recorded sale prices stand
   in for "current asking price" in this demo -- see inline note).
"""

import logging
import numpy as np
import pandas as pd

import config
from data_pipeline import (
    load_raw_data, apply_log_target, get_raw_feature_columns,
    get_macro_series, GeoEnricher, SpatialInfrastructureEngineer,
)
import models
from models import (
    GroupTimeSeriesSplit, preprocess_fold, OptunaTuner, regression_metrics,
    build_ridge_spatial_pipeline, tune_ridge_alpha,
)
import xgboost as xgb
import lightgbm as lgb
from stress_test import StressTester
from screening_engine import DealScreener

logger = logging.getLogger("real_estate_valuation.main")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def run_cv_for_model(model_name: str, X: pd.DataFrame, y: pd.Series, best_params: dict = None,
                      ridge_alpha: float = None) -> dict:
    """Runs full GroupTimeSeriesSplit CV for one model type, returns aggregated OOF metrics."""
    splitter = GroupTimeSeriesSplit(n_splits=config.N_SPATIAL_FOLDS)
    fold_metrics = []
    last_fold_artifacts = None  # keep last fold's fitted model + transformers for stress testing

    for fold_i, (train_idx, val_idx) in enumerate(splitter.split(X)):
        X_tr_raw = X.iloc[train_idx].reset_index(drop=True)
        X_va_raw = X.iloc[val_idx].reset_index(drop=True)
        y_tr_raw = y.iloc[train_idx].reset_index(drop=True)
        y_va_raw = y.iloc[val_idx].reset_index(drop=True)

        X_tr, y_tr, X_va, fitted = preprocess_fold(X_tr_raw, y_tr_raw, X_va_raw)

        if model_name == "xgboost":
            model = xgb.XGBRegressor(
                **best_params, random_state=config.RANDOM_STATE, tree_method="hist",
                objective="reg:squarederror", early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
                eval_metric="rmse",
            )
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va_raw)], verbose=False)
            preds = model.predict(X_va)

        elif model_name == "lightgbm":
            model = lgb.LGBMRegressor(
                max_depth=best_params["max_depth"], learning_rate=best_params["learning_rate"],
                n_estimators=best_params["n_estimators"], reg_alpha=best_params["reg_alpha"],
                reg_lambda=best_params["reg_lambda"], subsample=best_params["subsample"],
                colsample_bytree=best_params["colsample_bytree"],
                min_child_samples=best_params["min_child_weight"],
                random_state=config.RANDOM_STATE, verbosity=-1,
            )
            model.fit(
                X_tr, y_tr, eval_set=[(X_va, y_va_raw)],
                callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            preds = model.predict(X_va)

        elif model_name == "ridge_spatial":
            model = build_ridge_spatial_pipeline(ridge_alpha, fitted["numeric_cols"])
            model.fit(X_tr[fitted["numeric_cols"]], y_tr)
            preds = model.predict(X_va[fitted["numeric_cols"]])

        else:
            raise ValueError(model_name)

        m = regression_metrics(y_va_raw.to_numpy(), preds)
        m["fold"] = fold_i
        m["n_train"] = len(X_tr)
        m["n_val"] = len(X_va)
        fold_metrics.append(m)

        residual_std = np.std(y_va_raw.to_numpy() - preds)
        last_fold_artifacts = {
            "model": model, "fitted": fitted, "X_va_raw": X_va_raw, "y_va_raw": y_va_raw,
            "residual_std_logspace": residual_std,
        }

    metrics_df = pd.DataFrame(fold_metrics)
    agg = metrics_df[["MAE_$", "RMSE_$", "MAPE_%", "R2"]].mean().to_dict()
    agg["model"] = model_name
    return {"per_fold": metrics_df, "aggregate": agg, "last_fold_artifacts": last_fold_artifacts}


def main():
    logger.info("=" * 78)
    logger.info("REAL ESTATE VALUATION & PRICING MODEL -- full pipeline run")
    logger.info("=" * 78)

    # 1. Data -- real Ames, Iowa sales (2,930 rows)
    df = load_raw_data()
    df = apply_log_target(df)
    feature_cols = get_raw_feature_columns()
    extra_cols = [c for c in [config.DATE_COL, config.GROUP_COL] if c not in feature_cols]
    df = df[feature_cols + extra_cols + [config.TARGET_RAW, config.TARGET_LOG]].copy()

    # 2. Real macro data (live FRED, or cached real fallback)
    start = df[config.DATE_COL].min().strftime("%Y-%m-%d")
    end = df[config.DATE_COL].max().strftime("%Y-%m-%d")
    macro_frame, macro_was_live = get_macro_series(start, end)

    # 3. Real live geocoding probe (OpenStreetMap Nominatim + osmnx). Only
    # wired into the feature set if it actually returns usable (non-NaN)
    # real coordinates for this environment's network access.
    geo_enricher = GeoEnricher()
    spatial_engineer = SpatialInfrastructureEngineer()
    probe = geo_enricher.enrich_with_distances(df[[config.GROUP_COL]].drop_duplicates())
    use_geo_features = probe["DistToCommercialCenterKm"].notna().any()
    if use_geo_features:
        logger.info("Live geocoding available -- distance-to-CBD/ISU features enabled.")
    else:
        logger.warning(
            "Live geocoding unavailable in this environment (no network path to "
            "OpenStreetMap Nominatim). Proceeding WITHOUT fabricated coordinates -- "
            "the real Condition1/2 proximity flags (NearRailroad/NearArtery/"
            "NearPositiveFeature) carry the spatial-infrastructure signal instead."
        )
    models.set_pipeline_context(macro_frame, geo_enricher, spatial_engineer, use_geo_features)

    X_full = df[feature_cols + extra_cols].copy()
    y_full = df[config.TARGET_LOG].copy()

    # 4. IMMUTABLE 10% holdout split -- most recent real sales by date,
    # taken BEFORE any tuning or CV and never touched until step 7.
    order = df[config.DATE_COL].sort_values().index
    n_holdout = int(len(df) * config.HOLDOUT_FRACTION)
    holdout_idx = order[-n_holdout:]
    dev_idx = order[:-n_holdout]
    X_holdout, y_holdout = X_full.loc[holdout_idx].reset_index(drop=True), y_full.loc[holdout_idx].reset_index(drop=True)
    X, y = X_full.loc[dev_idx].reset_index(drop=True), y_full.loc[dev_idx].reset_index(drop=True)
    logger.info(
        "Immutable holdout split: %d dev rows (<= %s), %d holdout rows (> %s). "
        "Holdout is untouched until final evaluation.",
        len(X), df.loc[dev_idx, config.DATE_COL].max().date(),
        len(X_holdout), df.loc[dev_idx, config.DATE_COL].max().date(),
    )

    # 5. Hyperparameter tuning (Optuna) -- conservative bounds in config.py, dev set only
    xgb_tuner = OptunaTuner("xgboost", n_trials=config.N_OPTUNA_TRIALS)
    xgb_best_params = xgb_tuner.tune(X, y)

    lgb_tuner = OptunaTuner("lightgbm", n_trials=config.N_OPTUNA_TRIALS)
    lgb_best_params = lgb_tuner.tune(X, y)

    ridge_alpha = tune_ridge_alpha(X, y, [c for c in feature_cols if c not in config.CATEGORICAL_HIGH_CARD])

    # 6. Full GroupTimeSeriesSplit CV per model, dev set only
    logger.info("Running full out-of-fold evaluation for all three models (dev set) ...")
    results = {}
    results["xgboost"] = run_cv_for_model("xgboost", X, y, best_params=xgb_best_params)
    results["lightgbm"] = run_cv_for_model("lightgbm", X, y, best_params=lgb_best_params)
    results["ridge_spatial"] = run_cv_for_model("ridge_spatial", X, y, ridge_alpha=ridge_alpha)

    comparison = pd.DataFrame([r["aggregate"] for r in results.values()]).set_index("model")
    print("\n" + "=" * 78)
    print(f"OUT-OF-FOLD METRICS COMPARISON -- dev set (GroupTimeSeriesSplit) | macro: "
          f"{'LIVE FRED' if macro_was_live else 'cached real fallback'} | geo: "
          f"{'live' if use_geo_features else 'unavailable, real Condition1/2 flags used'}")
    print("=" * 78)
    print(comparison.round(4).to_string())

    best_model_name = comparison["RMSE_$"].idxmin()
    logger.info("Best model by dev-set OOF RMSE($): %s", best_model_name)

    # 7. Refit best model on FULL dev set, evaluate ONCE on the untouched holdout
    X_tr, y_tr, X_ho, fitted = preprocess_fold(X, y, X_holdout)
    if best_model_name == "xgboost":
        final_model = xgb.XGBRegressor(**xgb_best_params, random_state=config.RANDOM_STATE, tree_method="hist",
                                        objective="reg:squarederror")
        final_model.fit(X_tr, y_tr)
        holdout_preds = final_model.predict(X_ho)
    elif best_model_name == "lightgbm":
        final_model = lgb.LGBMRegressor(
            max_depth=lgb_best_params["max_depth"], learning_rate=lgb_best_params["learning_rate"],
            n_estimators=lgb_best_params["n_estimators"], reg_alpha=lgb_best_params["reg_alpha"],
            reg_lambda=lgb_best_params["reg_lambda"], subsample=lgb_best_params["subsample"],
            colsample_bytree=lgb_best_params["colsample_bytree"],
            min_child_samples=lgb_best_params["min_child_weight"],
            random_state=config.RANDOM_STATE, verbosity=-1,
        )
        final_model.fit(X_tr, y_tr)
        holdout_preds = final_model.predict(X_ho)
    else:
        final_model = build_ridge_spatial_pipeline(ridge_alpha, fitted["numeric_cols"])
        final_model.fit(X_tr[fitted["numeric_cols"]], y_tr)
        holdout_preds = final_model.predict(X_ho[fitted["numeric_cols"]])

    holdout_metrics = regression_metrics(y_holdout.to_numpy(), holdout_preds)
    print("\n" + "=" * 78)
    print(f"FINAL UNBIASED HOLDOUT EVALUATION -- {best_model_name} ({len(X_holdout)} untouched real sales)")
    print("=" * 78)
    print(pd.Series(holdout_metrics).round(4).to_string())

    # 8. Stress testing on the dev-CV artifacts for the best model
    artifacts = results[best_model_name]["last_fold_artifacts"]
    tester = StressTester(
        model=artifacts["model"], fitted_transformers=artifacts["fitted"],
        feature_cols=feature_cols, residual_std_logspace=artifacts["residual_std_logspace"],
    )

    print("\n" + "=" * 78)
    print(f"STRESS TEST SUITE -- best model: {best_model_name}")
    print("=" * 78)

    print("\n[1] Macro Shock Test (+300bps mortgage rate, +3pp unemployment)")
    print(tester.macro_shock_test(artifacts["X_va_raw"]).round(3).to_string())

    print("\n[2] Missing Attribute Sensitivity")
    print(tester.missing_attribute_sensitivity(artifacts["X_va_raw"]).round(3).to_string())

    print("\n[3] Out-of-Distribution / Out-of-Bounds Physical Features")
    print(tester.out_of_distribution_test(artifacts["X_va_raw"]).round(3).to_string())

    print("\n[4] Residual Diagnostics (plots saved to artifacts/plots/)")
    paths = tester.residual_diagnostics(
        artifacts["X_va_raw"], artifacts["y_va_raw"].to_numpy(), tag=best_model_name
    )
    for k, v in paths.items():
        print(f"  - {k}: {v}")

    # 9. Live Deal Screening demo -- a real sample batch from the holdout
    # set. NOTE: the holdout set's real recorded SalePrice stands in for
    # "current asking price" here purely for demonstration (this repo has
    # no live MLS feed); in production, `listings` would be a real feed of
    # active/incoming asking prices, which typically differ from eventual
    # sale price -- that gap is exactly what DealScreener is meant to find.
    print("\n" + "=" * 78)
    print("LIVE DEAL SCREENING DEMO (screening_engine.DealScreener)")
    print("=" * 78)
    demo_listings = df.loc[holdout_idx, feature_cols + extra_cols + [config.TARGET_RAW]].sample(
        n=min(20, len(holdout_idx)), random_state=config.RANDOM_STATE
    ).rename(columns={config.TARGET_RAW: "ListPrice"}).reset_index(drop=True)

    screener = DealScreener(
        model=final_model, fitted_transformers=fitted, feature_cols=feature_cols,
        baseline_anomaly_scores=X_tr["is_outlier_score"],
    )
    ranking = screener.screen_listings(demo_listings, list_price_col="ListPrice")
    print(ranking.round(2).to_string(index=False))

    print("\nDone.")
    return comparison, holdout_metrics, ranking


if __name__ == "__main__":
    main()
