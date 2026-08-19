"""
screening_engine.py
====================
Live Deal Screening & Automated Underwriting Module.

Ingests a real batch of active/incoming property listings (an MLS export,
a hand-compiled watchlist, or any real CSV of candidates -- never
synthetic), scores each one through the fitted, leakage-safe valuation
pipeline (the exact same frozen preprocessing chain used for the holdout
evaluation and `stress_test.py` -- no transformer is refit here), and
produces an Investment Ranking Table with:

  - Model Estimated Fair Value (P_fair) vs. Listing Price (P_list)
  - Discount Percentage: Delta = (P_fair - P_list) / P_list
  - Anomaly Z-Score from the frozen IsolationForestFlagger, standardized
    against the real training-fold anomaly-score distribution
  - An automated decision flag:
      BUY              : Delta > config.SCREENING_DISCOUNT_THRESHOLD
                          AND anomaly z-score < config.SCREENING_ANOMALY_Z_THRESHOLD
      MANUAL_APPRAISAL : Delta > threshold BUT anomaly z-score >= threshold
                          (a real discount, but on a physically unusual
                          property -- OOD features mean the model's fair-
                          value estimate is less trustworthy)
      REJECT           : Delta <= threshold
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config
from data_pipeline import RatioFeatureBuilder, MacroSeriesResampler
import models as _models  # PIPELINE_CONTEXT: real macro/geo data set once by main.py

logger = logging.getLogger("real_estate_valuation.screening_engine")


class DealScreener:
    """
    Scores real candidate listings through a frozen, already-fitted
    valuation pipeline. Never refits any transformer -- mirrors
    `stress_test.StressTester`'s frozen-artifact philosophy, since a
    screening engine in production runs against a deployed model, not one
    being retrained per request.
    """

    def __init__(self, model, fitted_transformers: Dict, feature_cols: List[str],
                 baseline_anomaly_scores: pd.Series,
                 discount_threshold: float = config.SCREENING_DISCOUNT_THRESHOLD,
                 anomaly_z_threshold: float = config.SCREENING_ANOMALY_Z_THRESHOLD):
        """
        model                   : fitted regressor (.predict) -- the winning model from main.py
        fitted_transformers     : dict from models.preprocess_fold() (winsorizer,
                                   isolation_forest, target_encoder)
        feature_cols            : raw property feature columns the pipeline expects
                                   (data_pipeline.get_raw_feature_columns())
        baseline_anomaly_scores : the real training-fold isolation-forest anomaly
                                   scores, used to standardize new listings' z-scores
                                   against the same real distribution the model was
                                   fit on (never refit here)
        """
        self.model = model
        self.winsorizer = fitted_transformers["winsorizer"]
        self.isolation_forest = fitted_transformers["isolation_forest"]
        self.target_encoder = fitted_transformers["target_encoder"]
        self.feature_cols = feature_cols
        self.discount_threshold = discount_threshold
        self.anomaly_z_threshold = anomaly_z_threshold
        self._baseline_mean = float(baseline_anomaly_scores.mean())
        self._baseline_std = float(baseline_anomaly_scores.std()) + 1e-9

    # ------------------------------------------------------------------ #
    def _prep(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Frozen preprocessing chain, identical in structure to
        `models.preprocess_fold()` / `stress_test.StressTester._prep()`:
        ratio features -> fold-safe real macro alignment -> optional live
        geo enrichment -> frozen winsorizer/isolation-forest/target
        encoder. No transformer is fit here; all are reused as-is.
        """
        ratio_builder = RatioFeatureBuilder()
        X = ratio_builder.transform(X_raw[self.feature_cols + [config.DATE_COL]])

        macro_frame = _models.PIPELINE_CONTEXT["macro_frame"]
        if macro_frame is None:
            raise RuntimeError(
                "models.PIPELINE_CONTEXT['macro_frame'] is not set -- call "
                "models.set_pipeline_context(...) in main.py before screening."
            )
        X = MacroSeriesResampler(macro_frame).transform(X)
        X = X.drop(columns=[config.DATE_COL])

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

    # ------------------------------------------------------------------ #
    def screen_listings(self, listings: pd.DataFrame, list_price_col: str = "ListPrice",
                         listing_id_col: Optional[str] = None) -> pd.DataFrame:
        """
        Scores a real batch of active/incoming listings and returns an
        Investment Ranking Table sorted by discount percentage (best deals
        first).

        Parameters
        ----------
        listings       : real DataFrame of candidate properties. Must contain
                          every column in `self.feature_cols` (the raw
                          property features the model was trained on),
                          `config.DATE_COL`, and a real asking-price column.
        list_price_col : name of the real asking-price column in `listings`.
        listing_id_col : optional column to use as the row identifier in the
                          output table (e.g. an MLS ID); defaults to the
                          input row position.

        Returns
        -------
        Investment Ranking Table with columns: ListingIndex/id, Neighborhood,
        ListPrice_$, FairValue_$, Discount_%, AnomalyZScore, Decision.
        """
        missing = [c for c in self.feature_cols + [config.DATE_COL, list_price_col] if c not in listings.columns]
        if missing:
            raise ValueError(f"Listings batch is missing required columns: {missing}")

        listings = listings.reset_index(drop=True)
        X = self._prep(listings)

        log_fair_value = self.model.predict(X)
        fair_value = np.expm1(log_fair_value)
        list_price = listings[list_price_col].to_numpy(dtype=float)

        discount_pct = (fair_value - list_price) / np.clip(list_price, 1.0, None)

        anomaly_score = X["is_outlier_score"].to_numpy()
        anomaly_z = (anomaly_score - self._baseline_mean) / self._baseline_std

        is_discounted = discount_pct > self.discount_threshold
        is_anomalous = anomaly_z >= self.anomaly_z_threshold
        decision = np.select(
            [~is_discounted, is_discounted & ~is_anomalous, is_discounted & is_anomalous],
            ["REJECT", "BUY", "MANUAL_APPRAISAL"],
            default="REJECT",
        )

        result = pd.DataFrame({
            "ListingID": listings[listing_id_col] if listing_id_col else listings.index,
            "Neighborhood": listings[config.GROUP_COL] if config.GROUP_COL in listings.columns else np.nan,
            "ListPrice_$": list_price,
            "FairValue_$": fair_value,
            "Discount_%": discount_pct * 100.0,
            "AnomalyZScore": anomaly_z,
            "Decision": decision,
        })
        result = result.sort_values("Discount_%", ascending=False).reset_index(drop=True)

        n_buy = int((decision == "BUY").sum())
        n_appraise = int((decision == "MANUAL_APPRAISAL").sum())
        n_reject = int((decision == "REJECT").sum())
        logger.info(
            "Screened %d listings: %d BUY, %d MANUAL_APPRAISAL, %d REJECT (discount>%.0f%%, anomaly_z<%.1f).",
            len(result), n_buy, n_appraise, n_reject,
            self.discount_threshold * 100, self.anomaly_z_threshold,
        )
        return result
