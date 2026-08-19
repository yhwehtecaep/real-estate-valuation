"""
export_pipeline.py
====================
Trains the winning model on the FULL real dev dataset (same
`GroupTimeSeriesSplit` + fold-safe preprocessing used everywhere else in
this codebase) and serializes every fitted artifact needed for stateless,
thread-safe live inference into a single joblib bundle at
`config.MODEL_ARTIFACT_PATH`.

Run:
    python export_pipeline.py                       # export the verified winner (Ridge)
    python export_pipeline.py --model xgboost        # force a specific model type
    python export_pipeline.py --reselect             # re-run the full dev-set CV
                                                       # comparison and export whichever
                                                       # model wins on out-of-fold RMSE

Bundle contents (see schema_utils.FeatureSchema for the input contract):
    model                    fitted regressor (.predict)
    winsorizer               fitted data_pipeline.Winsorizer
    isolation_forest         fitted data_pipeline.IsolationForestFlagger
    target_encoder           fitted data_pipeline.KFoldTargetEncoder
    numeric_cols             final numeric column order the model expects
    feature_cols             raw property feature columns (pre-engineering)
    macro_frame              real macro series (live-or-cached-fallback FRED data)
    use_geo_features         whether live geo enrichment was available at export time
    baseline_anomaly_mean/std standardization constants for live anomaly z-scores
    residual_std_logspace    holdout residual std, for stress-tested LTV limits
    holdout_metrics          the final unbiased holdout evaluation (for audit/display)
    schema                   schema_utils.FeatureSchema -- the frozen input contract

None of this training data or these statistics are synthetic -- every
number here is a real column, a real published/live macro figure, or a
statistic fit on real Ames sales.
"""

import argparse
import datetime as dt
import logging
import os

import joblib
import numpy as np
import pandas as pd

import config
from data_pipeline import (
    load_raw_data, apply_log_target, get_raw_feature_columns,
    get_macro_series, GeoEnricher, SpatialInfrastructureEngineer,
)
import models
from models import (
    preprocess_fold, OptunaTuner, regression_metrics,
    build_ridge_spatial_pipeline, tune_ridge_alpha,
)
import xgboost as xgb
import lightgbm as lgb
from schema_utils import FeatureSchema

logger = logging.getLogger("real_estate_valuation.export_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _fit_model(model_name, X_tr, y_tr, best_params=None, ridge_alpha=None, numeric_cols=None):
    if model_name == "xgboost":
        model = xgb.XGBRegressor(**best_params, random_state=config.RANDOM_STATE, tree_method="hist",
                                  objective="reg:squarederror")
        model.fit(X_tr, y_tr)
        return model
    elif model_name == "lightgbm":
        model = lgb.LGBMRegressor(
            max_depth=best_params["max_depth"], learning_rate=best_params["learning_rate"],
            n_estimators=best_params["n_estimators"], reg_alpha=best_params["reg_alpha"],
            reg_lambda=best_params["reg_lambda"], subsample=best_params["subsample"],
            colsample_bytree=best_params["colsample_bytree"],
            min_child_samples=best_params["min_child_weight"],
            random_state=config.RANDOM_STATE, verbosity=-1,
        )
        model.fit(X_tr, y_tr)
        return model
    elif model_name == "ridge_spatial":
        model = build_ridge_spatial_pipeline(ridge_alpha, numeric_cols)
        model.fit(X_tr[numeric_cols], y_tr)
        return model
    raise ValueError(f"Unknown model_name {model_name!r}")


def _predict(model_name, model, X, numeric_cols):
    if model_name == "ridge_spatial":
        return model.predict(X[numeric_cols])
    return model.predict(X)


def select_and_export(model_choice: str = "ridge_spatial", reselect: bool = False,
                       n_trials: int = config.N_OPTUNA_TRIALS) -> str:
    """
    model_choice : which model type to export. Defaults to "ridge_spatial",
                   the verified winner from the last full dev-set/holdout
                   comparison (see README) -- so a default export doesn't
                   pay the cost of re-tuning XGBoost/LightGBM every time.
    reselect     : if True, ignores model_choice and re-runs the full
                   XGBoost/LightGBM/Ridge dev-set CV comparison (same
                   procedure as main.py) and exports whichever model wins
                   on out-of-fold RMSE.
    """
    logger.info("=" * 78)
    logger.info("EXPORT PIPELINE -- training the production model on the full real dev set")
    logger.info("=" * 78)

    # 1. Real data + real macro/geo context (identical to main.py)
    df = load_raw_data()
    df = apply_log_target(df)
    feature_cols = get_raw_feature_columns()
    extra_cols = [c for c in [config.DATE_COL, config.GROUP_COL] if c not in feature_cols]
    df = df[feature_cols + extra_cols + [config.TARGET_RAW, config.TARGET_LOG]].copy()

    start = df[config.DATE_COL].min().strftime("%Y-%m-%d")
    end = df[config.DATE_COL].max().strftime("%Y-%m-%d")
    macro_frame, macro_was_live = get_macro_series(start, end)

    geo_enricher = GeoEnricher()
    spatial_engineer = SpatialInfrastructureEngineer()
    probe = geo_enricher.enrich_with_distances(df[[config.GROUP_COL]].drop_duplicates())
    use_geo_features = bool(probe["DistToCommercialCenterKm"].notna().any())
    models.set_pipeline_context(macro_frame, geo_enricher, spatial_engineer, use_geo_features)

    X_full = df[feature_cols + extra_cols].copy()
    y_full = df[config.TARGET_LOG].copy()

    # 2. Same immutable holdout split as main.py, so exported-model metrics
    # are directly comparable to main.py's reported holdout numbers.
    order = df[config.DATE_COL].sort_values().index
    n_holdout = int(len(df) * config.HOLDOUT_FRACTION)
    holdout_idx = order[-n_holdout:]
    dev_idx = order[:-n_holdout]
    X_holdout = X_full.loc[holdout_idx].reset_index(drop=True)
    y_holdout = y_full.loc[holdout_idx].reset_index(drop=True)
    X = X_full.loc[dev_idx].reset_index(drop=True)
    y = y_full.loc[dev_idx].reset_index(drop=True)

    best_params, ridge_alpha = None, None
    if reselect:
        logger.info("--reselect: re-running full dev-set CV comparison ...")
        xgb_best_params = OptunaTuner("xgboost", n_trials=n_trials).tune(X, y)
        lgb_best_params = OptunaTuner("lightgbm", n_trials=n_trials).tune(X, y)
        ridge_alpha = tune_ridge_alpha(X, y, [c for c in feature_cols if c not in config.CATEGORICAL_HIGH_CARD])

        from main import run_cv_for_model
        results = {
            "xgboost": run_cv_for_model("xgboost", X, y, best_params=xgb_best_params),
            "lightgbm": run_cv_for_model("lightgbm", X, y, best_params=lgb_best_params),
            "ridge_spatial": run_cv_for_model("ridge_spatial", X, y, ridge_alpha=ridge_alpha),
        }
        comparison = pd.DataFrame([r["aggregate"] for r in results.values()]).set_index("model")
        model_choice = comparison["RMSE_$"].idxmin()
        best_params = {"xgboost": xgb_best_params, "lightgbm": lgb_best_params, "ridge_spatial": None}[model_choice]
        logger.info("Selected %s by dev-set OOF RMSE:\n%s", model_choice, comparison.round(4).to_string())
    elif model_choice == "xgboost":
        best_params = OptunaTuner("xgboost", n_trials=n_trials).tune(X, y)
    elif model_choice == "lightgbm":
        best_params = OptunaTuner("lightgbm", n_trials=n_trials).tune(X, y)
    elif model_choice == "ridge_spatial":
        ridge_alpha = tune_ridge_alpha(X, y, [c for c in feature_cols if c not in config.CATEGORICAL_HIGH_CARD])
    else:
        raise ValueError(f"Unknown --model {model_choice!r}")

    # 3. Fold-safe preprocessing fit ONCE on the full dev set (never on holdout)
    X_tr, y_tr, X_ho, fitted = preprocess_fold(X, y, X_holdout)
    numeric_cols = fitted["numeric_cols"]
    final_model = _fit_model(model_choice, X_tr, y_tr, best_params=best_params, ridge_alpha=ridge_alpha,
                              numeric_cols=numeric_cols)
    holdout_preds = _predict(model_choice, final_model, X_ho, numeric_cols)
    holdout_metrics = regression_metrics(y_holdout.to_numpy(), holdout_preds)
    logger.info("Final unbiased holdout metrics for exported model (%s): %s", model_choice, holdout_metrics)

    residual_std_logspace = float(np.std(y_holdout.to_numpy() - holdout_preds))
    baseline_anomaly_scores = X_tr["is_outlier_score"]

    # 4. Freeze the input-schema contract from the real training data
    known_neighborhoods = sorted(df[config.GROUP_COL].unique().tolist())
    schema = FeatureSchema.build(known_neighborhoods)

    bundle = {
        "bundle_version": config.MODEL_BUNDLE_VERSION,
        "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_name": model_choice,
        "model": final_model,
        "winsorizer": fitted["winsorizer"],
        "isolation_forest": fitted["isolation_forest"],
        "target_encoder": fitted["target_encoder"],
        "numeric_cols": numeric_cols,
        "feature_cols": feature_cols,
        "macro_frame": macro_frame,
        "macro_was_live": macro_was_live,
        "use_geo_features": use_geo_features,
        "baseline_anomaly_mean": float(baseline_anomaly_scores.mean()),
        "baseline_anomaly_std": float(baseline_anomaly_scores.std()) + 1e-9,
        "residual_std_logspace": residual_std_logspace,
        "holdout_metrics": holdout_metrics,
        "schema": schema,
        "ridge_alpha": ridge_alpha,
    }

    os.makedirs(config.MODEL_ARTIFACT_DIR, exist_ok=True)
    joblib.dump(bundle, config.MODEL_ARTIFACT_PATH)
    logger.info("Saved model bundle (%s) to %s", model_choice, config.MODEL_ARTIFACT_PATH)
    return config.MODEL_ARTIFACT_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and export the production valuation pipeline bundle.")
    parser.add_argument("--model", choices=["xgboost", "lightgbm", "ridge_spatial"], default="ridge_spatial",
                         help="Model type to export (default: ridge_spatial, the verified dev/holdout winner).")
    parser.add_argument("--reselect", action="store_true",
                         help="Re-run the full dev-set CV comparison and export whichever model wins.")
    parser.add_argument("--trials", type=int, default=config.N_OPTUNA_TRIALS,
                         help="Optuna trials for XGBoost/LightGBM tuning.")
    args = parser.parse_args()
    select_and_export(model_choice=args.model, reselect=args.reselect, n_trials=args.trials)
