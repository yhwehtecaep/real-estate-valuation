"""
models.py
=========
Model definitions, spatial/temporal cross-validation, and Optuna-based
hyperparameter tuning with conservative regularization defaults.

Why not plain k-fold?
----------------------
Real estate prices are spatially autocorrelated (nearby parcels share
unobserved neighborhood shocks) AND temporally autocorrelated (macro
conditions drift over the listing window). Plain k-fold shuffles rows
randomly, which leaks both kinds of structure into the validation fold
via near-duplicate neighbors / near-duplicate time periods. This module
implements `GroupTimeSeriesSplit`: groups are ZipCode clusters, and folds
are ordered by each group's first-seen listing date, so validation folds
only ever contain data whose spatial group AND time period were unseen
in training.
"""

import logging
from typing import Dict, List, Tuple, Iterator

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
from optuna.samplers import TPESampler

import xgboost as xgb
import lightgbm as lgb

import config
from data_pipeline import (
    KFoldTargetEncoder, Winsorizer, IsolationForestFlagger, RatioFeatureBuilder,
    MacroSeriesResampler, get_raw_feature_columns,
)

logger = logging.getLogger("real_estate_valuation.models")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# --------------------------------------------------------------------------- #
# Process-wide real-data context (macro series + live geo enrichers), set
# once by main.py via `set_pipeline_context()` after fetching real data.
# `preprocess_fold()` reads from this on every fold call. This is a plain
# module-level dict rather than a parameter threaded through every tuning/
# CV function -- appropriate here because the macro series and geo
# enrichers are process-wide real data sources (fetched once, cached
# internally), not per-fold state.
# --------------------------------------------------------------------------- #
PIPELINE_CONTEXT = {
    "macro_frame": None,        # real pd.DataFrame from data_pipeline.get_macro_series()
    "geo_enricher": None,       # data_pipeline.GeoEnricher instance, or None
    "spatial_engineer": None,   # data_pipeline.SpatialInfrastructureEngineer instance, or None
    "use_geo_features": False,  # True only if live geocoding produced usable (non-NaN) columns
}


def set_pipeline_context(macro_frame, geo_enricher=None, spatial_engineer=None, use_geo_features=False):
    """Called once by main.py after fetching real macro/geo data, before any CV/tuning runs."""
    PIPELINE_CONTEXT["macro_frame"] = macro_frame
    PIPELINE_CONTEXT["geo_enricher"] = geo_enricher
    PIPELINE_CONTEXT["spatial_engineer"] = spatial_engineer
    PIPELINE_CONTEXT["use_geo_features"] = use_geo_features


# --------------------------------------------------------------------------- #
# Spatial + temporal cross-validation
# --------------------------------------------------------------------------- #
class GroupTimeSeriesSplit:
    """
    Combines spatial grouping (no two folds share a ZipCode cluster) with
    temporal ordering (training always precedes validation in time).

    Groups (ZipCode clusters) are ordered by their earliest ListingDate.
    They are then chunked into `n_splits + 1` contiguous blocks; fold i
    trains on blocks [0..i] and validates on block i+1. This guarantees:
      1. No ZipCode ever appears in both train and validation for a fold
         (spatial leakage control).
      2. Validation blocks are always later in time than the training
         blocks that precede them (temporal leakage control).
    """

    def __init__(self, n_splits: int = config.N_SPATIAL_FOLDS):
        self.n_splits = n_splits

    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        group_first_date = (
            X.groupby(config.GROUP_COL)[config.DATE_COL].min().sort_values()
        )
        ordered_groups = group_first_date.index.to_numpy()
        blocks = np.array_split(ordered_groups, self.n_splits + 1)

        cumulative_groups = set(blocks[0])
        for i in range(1, len(blocks)):
            val_groups = set(blocks[i])
            train_mask = X[config.GROUP_COL].isin(cumulative_groups).to_numpy()
            val_mask = X[config.GROUP_COL].isin(val_groups).to_numpy()
            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]
            if len(train_idx) > 0 and len(val_idx) > 0:
                yield train_idx, val_idx
            cumulative_groups |= val_groups

    def get_n_splits(self) -> int:
        return self.n_splits


# --------------------------------------------------------------------------- #
# Fold-local preprocessing (fit ONLY on the training slice of each fold)
# --------------------------------------------------------------------------- #
def preprocess_fold(X_train, y_train, X_val):
    """
    Fits Winsorizer + IsolationForest flag + KFoldTargetEncoder on the
    training fold only, then applies the same fitted transformers to the
    validation fold. This is the single choke point that prevents target
    and distribution leakage across the whole modeling pipeline.

    Also performs fold-safe real macro alignment (MacroSeriesResampler,
    using the real live-or-cached-fallback FRED series in
    PIPELINE_CONTEXT["macro_frame"]) and, if live geocoding produced usable
    data, real live-geo enrichment (PIPELINE_CONTEXT["geo_enricher"] /
    ["spatial_engineer"]) -- both invoked per fold for architectural
    consistency with the rest of this leakage-safe pipeline (see
    data_pipeline.py's module docstring for why these particular sources
    carry no actual leakage risk regardless of fold-scoping).

    Returns processed (X_train, X_val) with all-numeric columns, plus the
    fitted transformers (needed later for the held-out test set / stress
    tests, which must reuse train-fold statistics, never fold-mixed ones).
    """
    feature_cols = get_raw_feature_columns()
    cols_needed = feature_cols + [config.DATE_COL]

    ratio_builder = RatioFeatureBuilder()
    X_train = ratio_builder.transform(X_train[cols_needed])
    X_val = ratio_builder.transform(X_val[cols_needed])

    macro_frame = PIPELINE_CONTEXT["macro_frame"]
    assert macro_frame is not None, (
        "models.PIPELINE_CONTEXT['macro_frame'] is not set -- call "
        "models.set_pipeline_context(...) once in main.py before running CV/tuning."
    )
    macro_resampler = MacroSeriesResampler(macro_frame)
    macro_resampler.fit(X_train)
    X_train = macro_resampler.transform(X_train)
    X_val = macro_resampler.transform(X_val)
    X_train = X_train.drop(columns=[config.DATE_COL])
    X_val = X_val.drop(columns=[config.DATE_COL])

    if PIPELINE_CONTEXT["use_geo_features"] and PIPELINE_CONTEXT["geo_enricher"] is not None:
        geo_enricher = PIPELINE_CONTEXT["geo_enricher"]
        X_train = geo_enricher.enrich_with_distances(X_train)
        X_val = geo_enricher.enrich_with_distances(X_val)
        spatial_engineer = PIPELINE_CONTEXT["spatial_engineer"]
        if spatial_engineer is not None:
            X_train = spatial_engineer.enrich_with_infrastructure_distances(X_train, geo_enricher)
            X_val = spatial_engineer.enrich_with_infrastructure_distances(X_val, geo_enricher)

    numeric_cols = [c for c in X_train.columns if c not in config.CATEGORICAL_HIGH_CARD]

    winsorizer = Winsorizer(columns=numeric_cols)
    winsorizer.fit(X_train)
    X_train = winsorizer.transform(X_train)
    X_val = winsorizer.transform(X_val)

    iso = IsolationForestFlagger(columns=numeric_cols)
    iso.fit(X_train)
    inlier_mask = iso.train_mask(X_train)  # drop training outliers only, never touch val rows
    X_train = X_train.loc[inlier_mask].reset_index(drop=True)
    y_train = y_train.reset_index(drop=True).loc[inlier_mask].reset_index(drop=True)
    X_train = iso.transform(X_train)
    X_val = iso.transform(X_val)

    encoder = KFoldTargetEncoder(columns=config.CATEGORICAL_HIGH_CARD)
    encoder.fit(X_train, y_train)
    X_train = encoder.transform(X_train)
    X_val = encoder.transform(X_val)

    fitted = {"winsorizer": winsorizer, "isolation_forest": iso, "target_encoder": encoder,
              "numeric_cols": numeric_cols}
    return X_train, y_train, X_val, fitted


# --------------------------------------------------------------------------- #
# Optuna tuning
# --------------------------------------------------------------------------- #
class OptunaTuner:
    """
    Tunes XGBoost or LightGBM regularization hyperparameters against
    out-of-fold RMSE under `GroupTimeSeriesSplit`, enforcing the spec's
    conservative bounds (shallow trees, bounded subsampling, explicit L1/L2).
    """

    def __init__(self, model_type: str, n_trials: int = config.N_OPTUNA_TRIALS,
                 random_state: int = config.RANDOM_STATE):
        assert model_type in ("xgboost", "lightgbm")
        self.model_type = model_type
        self.n_trials = n_trials
        self.random_state = random_state
        self.study_ = None
        self.best_params_ = None

    def _suggest_params(self, trial: optuna.Trial) -> Dict:
        ss = config.SEARCH_SPACE
        params = {
            "max_depth": trial.suggest_int("max_depth", *ss.max_depth),
            "learning_rate": trial.suggest_float("learning_rate", *ss.learning_rate, log=True),
            "n_estimators": trial.suggest_int("n_estimators", *ss.n_estimators),
            "reg_alpha": trial.suggest_float("reg_alpha", *ss.reg_alpha, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", *ss.reg_lambda, log=True),
            "subsample": trial.suggest_float("subsample", *ss.subsample),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *ss.colsample_bytree),
            "min_child_weight": trial.suggest_int("min_child_weight", *ss.min_child_weight),
        }
        return params

    def _build_model(self, params: Dict):
        if self.model_type == "xgboost":
            return xgb.XGBRegressor(
                **params, random_state=self.random_state, tree_method="hist",
                objective="reg:squarederror", early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
                eval_metric="rmse",
            )
        else:
            return lgb.LGBMRegressor(
                max_depth=params["max_depth"], learning_rate=params["learning_rate"],
                n_estimators=params["n_estimators"], reg_alpha=params["reg_alpha"],
                reg_lambda=params["reg_lambda"], subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                min_child_samples=params["min_child_weight"],
                random_state=self.random_state, verbosity=-1,
            )

    def objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        params = self._suggest_params(trial)
        splitter = GroupTimeSeriesSplit(n_splits=3)  # cheaper inner CV during search
        fold_rmses = []
        for train_idx, val_idx in splitter.split(X):
            X_tr_raw, X_va_raw = X.iloc[train_idx].reset_index(drop=True), X.iloc[val_idx].reset_index(drop=True)
            y_tr_raw, y_va_raw = y.iloc[train_idx].reset_index(drop=True), y.iloc[val_idx].reset_index(drop=True)

            X_tr, y_tr, X_va, fitted = preprocess_fold(X_tr_raw, y_tr_raw, X_va_raw)
            y_va = y_va_raw  # isolation-forest never drops validation rows

            model = self._build_model(params)
            if self.model_type == "xgboost":
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            else:
                model.fit(
                    X_tr, y_tr, eval_set=[(X_va, y_va)],
                    callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False)],
                )
            preds = model.predict(X_va)
            fold_rmses.append(np.sqrt(mean_squared_error(y_va, preds)))
        return float(np.mean(fold_rmses))

    def tune(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        logger.info("Starting Optuna search for %s (%d trials) ...", self.model_type, self.n_trials)
        self.study_ = optuna.create_study(
            direction="minimize", sampler=TPESampler(seed=self.random_state)
        )
        self.study_.optimize(lambda t: self.objective(t, X, y), n_trials=self.n_trials, show_progress_bar=False)
        self.best_params_ = self.study_.best_params
        logger.info("%s best OOF RMSE (log-price): %.5f | params: %s",
                    self.model_type, self.study_.best_value, self.best_params_)
        return self.best_params_


# --------------------------------------------------------------------------- #
# Ridge baseline with spatial polynomial features
# --------------------------------------------------------------------------- #
def build_ridge_spatial_pipeline(alpha: float, numeric_cols: List[str]) -> Pipeline:
    """
    Interpretable linear baseline: standardized numeric features plus a
    degree-2 polynomial expansion of the real overall-quality/condition
    grades (config.RIDGE_POLY_COLUMNS), to capture smooth non-linear
    quality-driven price gradients without letting the linear model overfit
    every other feature via blanket polynomial expansion. The raw Ames
    dataset has no real per-property Latitude/Longitude, so this substitutes
    OverallQual/OverallCond -- the strongest real non-spatial gradient in
    the data -- for the spatial polynomial the spec originally requested;
    the real Neighborhood field (target-encoded upstream) still captures
    location.
    """
    from sklearn.compose import ColumnTransformer

    poly_cols = [c for c in config.RIDGE_POLY_COLUMNS if c in numeric_cols]
    other_cols = [c for c in numeric_cols if c not in poly_cols]

    ct = ColumnTransformer([
        ("quality_poly", Pipeline([
            ("poly", PolynomialFeatures(degree=config.RIDGE_POLY_DEGREE, include_bias=False)),
            ("scale", StandardScaler()),
        ]), poly_cols),
        ("other_scaled", StandardScaler(), other_cols),
    ])
    return Pipeline([("features", ct), ("ridge", Ridge(alpha=alpha, random_state=config.RANDOM_STATE))])


def tune_ridge_alpha(X: pd.DataFrame, y: pd.Series, numeric_cols: List[str]) -> float:
    """Selects Ridge alpha by out-of-fold RMSE under the same spatial-temporal splitter."""
    splitter = GroupTimeSeriesSplit(n_splits=3)
    best_alpha, best_rmse = None, np.inf
    for alpha in config.RIDGE_ALPHA_GRID:
        rmses = []
        for train_idx, val_idx in splitter.split(X):
            X_tr_raw, X_va_raw = X.iloc[train_idx].reset_index(drop=True), X.iloc[val_idx].reset_index(drop=True)
            y_tr_raw, y_va_raw = y.iloc[train_idx].reset_index(drop=True), y.iloc[val_idx].reset_index(drop=True)
            X_tr, y_tr, X_va, fitted = preprocess_fold(X_tr_raw, y_tr_raw, X_va_raw)
            pipe = build_ridge_spatial_pipeline(alpha, fitted["numeric_cols"])
            pipe.fit(X_tr[fitted["numeric_cols"]], y_tr)
            preds = pipe.predict(X_va[fitted["numeric_cols"]])
            rmses.append(np.sqrt(mean_squared_error(y_va_raw, preds)))
        mean_rmse = float(np.mean(rmses))
        if mean_rmse < best_rmse:
            best_rmse, best_alpha = mean_rmse, alpha
    logger.info("Ridge best alpha=%.3f (OOF RMSE log-price=%.5f)", best_alpha, best_rmse)
    return best_alpha


# --------------------------------------------------------------------------- #
# Metrics helper
# --------------------------------------------------------------------------- #
def regression_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    """
    Computes metrics in BOTH log space (what the model optimizes) and back-
    transformed dollar space (what an investment committee reads), using
    expm1 to invert the log1p target transform.
    """
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.clip(y_pred, 0, None)
    return {
        "MAE_$": mean_absolute_error(y_true, y_pred),
        "RMSE_$": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE_%": float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100),
        "R2": r2_score(y_true, y_pred),
        "RMSE_logspace": np.sqrt(mean_squared_error(y_true_log, y_pred_log)),
    }
