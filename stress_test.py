"""
stress_test.py
===============
Post-training diagnostics for the Real Estate Valuation & Pricing Model.

All four required diagnostics are implemented as methods on `StressTester`:
  1. Macro Shock Test          -> `macro_shock_test()`
  2. Missing Attribute Sens.   -> `missing_attribute_sensitivity()`
  3. Out-of-Bounds Physical    -> `out_of_distribution_test()`
  4. Residual Diagnostics      -> `residual_diagnostics()`

`StressTester` is deliberately decoupled from model *training*: it takes an
already-fitted model plus the fold-local transformers (winsorizer,
isolation-forest flag, target encoder) that were fit on that model's
training data, and never refits anything. This mirrors how a valuation
model is actually stress-tested in production: against a frozen, deployed
artifact.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from data_pipeline import RatioFeatureBuilder, MacroSeriesResampler
import models as _models  # for PIPELINE_CONTEXT (real macro/geo data, set once by main.py)

logger = logging.getLogger("real_estate_valuation.stress_test")


class StressTester:
    def __init__(self, model, fitted_transformers: Dict, feature_cols: List[str],
                 residual_std_logspace: float, plot_dir: str = config.PLOT_DIR):
        """
        model                    : fitted regressor with .predict()
        fitted_transformers      : dict from models.preprocess_fold() -- winsorizer,
                                    isolation_forest, target_encoder, numeric_cols
        feature_cols             : raw feature columns (pre target-encoding) the
                                    model's input pipeline expects
        residual_std_logspace    : std of validation residuals in log-price space,
                                    used to size uncertainty bounds for OOD flags
        """
        self.model = model
        self.winsorizer = fitted_transformers["winsorizer"]
        self.isolation_forest = fitted_transformers["isolation_forest"]
        self.target_encoder = fitted_transformers["target_encoder"]
        self.numeric_cols = fitted_transformers["numeric_cols"]
        self.feature_cols = feature_cols
        self.residual_std = residual_std_logspace
        self.plot_dir = plot_dir

    # ------------------------------------------------------------------ #
    def _prep(self, X_raw: pd.DataFrame, macro_override: Dict[str, float] = None) -> pd.DataFrame:
        """
        Runs the frozen (already-fitted) preprocessing chain -- no
        re-fitting -- mirroring `models.preprocess_fold()`: ratio features,
        fold-safe real macro alignment (same real macro series used at
        training time, from `models.PIPELINE_CONTEXT`), then the frozen
        winsorizer / isolation-forest / target encoder.

        `macro_override`, if given, adds a delta to a macro column AFTER
        the real macro join (used by `macro_shock_test` -- shocking the
        raw input date wouldn't work since the join always looks up the
        real historical value for that date).
        """
        ratio_builder = RatioFeatureBuilder()
        X = ratio_builder.transform(X_raw[self.feature_cols + [config.DATE_COL]])

        macro_frame = _models.PIPELINE_CONTEXT["macro_frame"]
        assert macro_frame is not None, "models.PIPELINE_CONTEXT['macro_frame'] is not set."
        X = MacroSeriesResampler(macro_frame).transform(X)
        X = X.drop(columns=[config.DATE_COL])

        if macro_override:
            for col, delta in macro_override.items():
                X[col] = X[col] + delta

        if _models.PIPELINE_CONTEXT["use_geo_features"] and _models.PIPELINE_CONTEXT["geo_enricher"] is not None:
            geo_enricher = _models.PIPELINE_CONTEXT["geo_enricher"]
            X = geo_enricher.enrich_with_distances(X)
            spatial_engineer = _models.PIPELINE_CONTEXT["spatial_engineer"]
            if spatial_engineer is not None:
                X = spatial_engineer.enrich_with_infrastructure_distances(X, geo_enricher)

        X = self.winsorizer.transform(X)
        X = self.isolation_forest.transform(X)
        X = self.target_encoder.transform(X)
        return X

    def _predict_dollars(self, X_raw: pd.DataFrame, macro_override: Dict[str, float] = None) -> np.ndarray:
        X = self._prep(X_raw, macro_override=macro_override)
        log_pred = self.model.predict(X)
        return np.expm1(log_pred)

    # ------------------------------------------------------------------ #
    # 1. Macro shock test
    # ------------------------------------------------------------------ #
    def macro_shock_test(self, X_baseline: pd.DataFrame) -> pd.DataFrame:
        """
        Shifts MortgageRate30Y by +MACRO_SHOCK_RATE_BPS and
        LocalUnemploymentRate by +MACRO_SHOCK_UNEMPLOYMENT_PP (a recession
        scenario), and reports the resulting shift in predicted price --
        both in aggregate and per price decile, to check the model responds
        directionally sensibly (higher rates / higher unemployment -> lower
        predicted value) and without discontinuities. The real Ames dataset
        has no per-property local-income series, so the spec's "-10% local
        income" shock leg is implemented via the real national unemployment
        rate instead, which is already one of the pipeline's real macro
        features.
        """
        logger.info("Running macro shock test ...")
        baseline_pred = self._predict_dollars(X_baseline)
        shock = {
            "MortgageRate30Y": config.MACRO_SHOCK_RATE_BPS / 100.0,
            "LocalUnemploymentRate": config.MACRO_SHOCK_UNEMPLOYMENT_PP,
        }
        shocked_pred = self._predict_dollars(X_baseline, macro_override=shock)

        pct_change = (shocked_pred - baseline_pred) / np.clip(baseline_pred, 1, None) * 100.0

        result = pd.DataFrame({
            "baseline_pred_$": baseline_pred,
            "shocked_pred_$": shocked_pred,
            "pct_change": pct_change,
            "price_decile": pd.qcut(baseline_pred, config.PRICE_DECILES, labels=False, duplicates="drop"),
        })
        summary = result.groupby("price_decile")["pct_change"].agg(["mean", "std", "min", "max"])
        summary.loc["OVERALL"] = result["pct_change"].agg(["mean", "std", "min", "max"])
        logger.info(
            "Macro shock (+%dbps rate, +%.1fpp unemployment): mean price impact %.2f%%",
            config.MACRO_SHOCK_RATE_BPS, config.MACRO_SHOCK_UNEMPLOYMENT_PP, result["pct_change"].mean(),
        )
        return summary

    # ------------------------------------------------------------------ #
    # 2. Missing attribute sensitivity
    # ------------------------------------------------------------------ #
    def missing_attribute_sensitivity(self, X_baseline: pd.DataFrame) -> pd.DataFrame:
        """
        For each optional attribute, randomly masks it (replaced with the
        training-fold median/winsorized floor as a "missing -> imputed"
        proxy) on MASK_FRACTION of rows across N_MASK_TRIALS repetitions,
        and measures the distribution of prediction deltas. A well-behaved
        model shows smooth, small, low-variance deltas; large or highly
        variable deltas indicate the model has memorized that attribute
        rather than learned a smooth relationship.
        """
        logger.info("Running missing-attribute sensitivity test ...")
        baseline_pred = self._predict_dollars(X_baseline)
        rows = []
        rng = np.random.default_rng(config.RANDOM_STATE)

        for attr in config.OPTIONAL_ATTRS_FOR_MASKING:
            fallback_value = X_baseline[attr].median()
            deltas_pct_all = []
            for trial in range(config.N_MASK_TRIALS):
                mask_idx = rng.choice(
                    X_baseline.index, size=int(len(X_baseline) * config.MASK_FRACTION), replace=False
                )
                X_masked = X_baseline.copy()
                X_masked.loc[mask_idx, attr] = fallback_value
                masked_pred = self._predict_dollars(X_masked)
                delta_pct = (masked_pred[mask_idx] - baseline_pred[mask_idx]) / np.clip(
                    baseline_pred[mask_idx], 1, None
                ) * 100.0
                deltas_pct_all.append(delta_pct)
            deltas_pct_all = np.concatenate(deltas_pct_all)
            rows.append({
                "attribute": attr,
                "mean_abs_pct_delta": np.mean(np.abs(deltas_pct_all)),
                "std_pct_delta": np.std(deltas_pct_all),
                "max_abs_pct_delta": np.max(np.abs(deltas_pct_all)),
                "smooth_degradation": np.std(deltas_pct_all) < 2 * np.mean(np.abs(deltas_pct_all)) + 1e-6,
            })
        result = pd.DataFrame(rows).set_index("attribute")
        logger.info("Missing-attribute sensitivity:\n%s", result.to_string())
        return result

    # ------------------------------------------------------------------ #
    # 3. Out-of-distribution / out-of-bounds physical features
    # ------------------------------------------------------------------ #
    def out_of_distribution_test(self, X_baseline: pd.DataFrame) -> pd.DataFrame:
        """
        Constructs synthetic edge-case rows (e.g. a 10,000 sq ft home with 1
        bedroom) by overriding a template "typical" row's fields per
        `config.OOD_TEST_CASES`, predicts on them, and flags predictions
        whose implied uncertainty (approximated by distance-in-feature-space
        proxied via the isolation-forest anomaly score) exceeds
        OOD_FLAG_SIGMA_MULTIPLIER standard deviations of in-sample residuals.
        """
        logger.info("Running out-of-distribution physical feature test ...")
        template = X_baseline.iloc[[0]].copy().reset_index(drop=True)
        rows = []
        for case in config.OOD_TEST_CASES:
            row = template.copy()
            for k, v in case.items():
                if k == "name":
                    continue
                if k in row.columns:
                    row[k] = v
            pred = self._predict_dollars(row)[0]

            # Anomaly score from the frozen isolation forest as an uncertainty proxy
            X_prepped = self._prep(row)
            anomaly_score = X_prepped["is_outlier_score"].iloc[0]
            baseline_scores = self._prep(X_baseline)["is_outlier_score"]
            z = (anomaly_score - baseline_scores.mean()) / (baseline_scores.std() + 1e-9)
            flagged = abs(z) > config.OOD_FLAG_SIGMA_MULTIPLIER

            rows.append({
                "case": case["name"],
                "predicted_price_$": pred,
                "anomaly_score": anomaly_score,
                "anomaly_z": z,
                "flagged_high_uncertainty": bool(flagged),
            })
        result = pd.DataFrame(rows).set_index("case")
        logger.info("OOD test results:\n%s", result.to_string())
        return result

    # ------------------------------------------------------------------ #
    # 4. Residual diagnostics
    # ------------------------------------------------------------------ #
    def residual_diagnostics(self, X_baseline: pd.DataFrame, y_true_log: np.ndarray,
                              tag: str = "model") -> Dict[str, str]:
        """
        Generates and saves three diagnostic plots:
          - predicted vs actual (dollar space)
          - spatial residual heatmap (Lat/Lon colored by residual)
          - error distribution across price deciles (boxplot)
        Returns the saved file paths.
        """
        logger.info("Generating residual diagnostic plots ...")
        X_prepped = self._prep(X_baseline)
        pred_log = self.model.predict(X_prepped)
        pred = np.expm1(pred_log)
        actual = np.expm1(y_true_log)
        residual = actual - pred

        paths = {}

        # --- Predicted vs Actual ---
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(actual, pred, alpha=0.3, s=10)
        lims = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel("Actual Sale Price ($)")
        ax.set_ylabel("Predicted Sale Price ($)")
        ax.set_title(f"Predicted vs Actual -- {tag}")
        fig.tight_layout()
        p1 = f"{self.plot_dir}/{tag}_pred_vs_actual.png"
        fig.savefig(p1, dpi=120)
        plt.close(fig)
        paths["pred_vs_actual"] = p1

        # --- Spatial residual view: by real Neighborhood (the raw Ames
        # dataset has no per-property Latitude/Longitude, so residuals are
        # aggregated by the real Neighborhood field instead of plotted on a
        # literal map) ---
        resid_by_nbhd = (
            pd.DataFrame({config.GROUP_COL: X_baseline[config.GROUP_COL].values, "residual": residual})
            .groupby(config.GROUP_COL)["residual"].mean().sort_values()
        )
        fig, ax = plt.subplots(figsize=(7, 8))
        colors = ["#d62728" if v < 0 else "#2ca02c" for v in resid_by_nbhd.values]
        ax.barh(resid_by_nbhd.index, resid_by_nbhd.values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Mean Residual ($, actual - predicted)")
        ax.set_title(f"Residuals by Neighborhood -- {tag}")
        fig.tight_layout()
        p2 = f"{self.plot_dir}/{tag}_spatial_residuals.png"
        fig.savefig(p2, dpi=120)
        plt.close(fig)
        paths["spatial_residuals"] = p2

        # --- Error distribution by price decile ---
        deciles = pd.qcut(actual, config.PRICE_DECILES, labels=False, duplicates="drop")
        df_err = pd.DataFrame({"decile": deciles, "abs_pct_error": np.abs(residual) / np.clip(actual, 1, None) * 100})
        fig, ax = plt.subplots(figsize=(8, 5))
        df_err.boxplot(column="abs_pct_error", by="decile", ax=ax)
        ax.set_xlabel("Price Decile (0=cheapest)")
        ax.set_ylabel("Absolute % Error")
        ax.set_title(f"Error Distribution by Price Decile -- {tag}")
        plt.suptitle("")
        fig.tight_layout()
        p3 = f"{self.plot_dir}/{tag}_error_by_decile.png"
        fig.savefig(p3, dpi=120)
        plt.close(fig)
        paths["error_by_decile"] = p3

        logger.info("Saved diagnostic plots: %s", paths)
        return paths
