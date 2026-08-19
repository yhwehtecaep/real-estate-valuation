"""
api_service.py
================
Asynchronous production REST API for the Real Estate Valuation &
Underwriting system, built on FastAPI + uvicorn.

Run (dev):
    python export_pipeline.py                 # produces the model bundle, once
    uvicorn api_service:app --host 0.0.0.0 --port 8000 --reload

Run (prod, see Dockerfile):
    uvicorn api_service:app --host 0.0.0.0 --port 8000

Endpoints
---------
GET  /health        Model artifact load state + FRED API reachability.
POST /predict        Fair value, anomaly z-score, and feature contributions
                      for one or more real property listings.
POST /underwrite      Full deal-screening decision (BUY / MANUAL_APPRAISAL /
                      REJECT), recommended bid range, and a stress-tested
                      LTV limit, for one or more real listings with a real
                      asking price.

Model artifacts are loaded ONCE at startup into a module-level, read-only
ModelState object; every request thereafter only reads from it (frozen
transformers, frozen model), so concurrent requests across uvicorn's
thread pool are safe without any locking. Every request is logged to the
database (database.py) for audit.
"""

import datetime as dt
import logging
import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

import config
import database
from data_pipeline import FredMacroClient, MacroSeriesResampler, RatioFeatureBuilder
from schema_utils import unseen_neighborhood_warning, validate_payload

logger = logging.getLogger("real_estate_valuation.api_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

app = FastAPI(title=config.API_TITLE, version=config.API_VERSION)


# --------------------------------------------------------------------------- #
# Global, thread-safe model state -- populated ONCE at startup, read-only
# thereafter. FastAPI/uvicorn serve requests from a thread pool; nothing
# here is mutated per-request, so concurrent reads are safe without locks.
# --------------------------------------------------------------------------- #
class ModelState:
    bundle: Optional[Dict[str, Any]] = None
    loaded_at: Optional[str] = None
    load_error: Optional[str] = None


model_state = ModelState()


# --------------------------------------------------------------------------- #
# Pydantic request/response schemas
# --------------------------------------------------------------------------- #
class PropertyListingRequest(BaseModel):
    """One real property listing -- physical dimensions, quality grades, location, and date."""

    OverallQual: float = Field(..., ge=1, le=10, description="Overall material/finish quality, 1-10")
    OverallCond: float = Field(..., ge=1, le=10, description="Overall condition rating, 1-10")
    LivingAreaSqFt: float = Field(..., gt=0, description="Above-grade living area, sq ft")
    LotAreaSqFt: float = Field(..., gt=0, description="Lot size, sq ft")
    GarageAreaSqFt: float = Field(0.0, ge=0, description="Garage area, sq ft (0 if none)")
    YearBuilt: float = Field(..., ge=1800, le=2100)
    YearRenovated: float = Field(..., ge=1800, le=2100,
                                  description="Year of last remodel; same as YearBuilt if never renovated")
    BedroomAbvGr: float = Field(..., ge=0)
    FullBath: float = Field(..., ge=0)
    TotRmsAbvGrd: float = Field(..., ge=0)
    NearRailroad: int = Field(0, ge=0, le=1)
    NearArtery: int = Field(0, ge=0, le=1)
    NearPositiveFeature: int = Field(0, ge=0, le=1)
    Neighborhood: str = Field(..., description="Real Ames neighborhood code, e.g. 'CollgCr'")
    ListingDate: str = Field(..., description="ISO date (YYYY-MM-DD)")
    ListPrice: Optional[float] = Field(None, gt=0, description="Real asking price; required for /underwrite")

    @field_validator("ListingDate")
    @classmethod
    def _valid_date(cls, v: str) -> str:
        try:
            pd.Timestamp(v)
        except Exception as e:
            raise ValueError(f"ListingDate {v!r} is not a valid date") from e
        return v


class PredictBatchRequest(BaseModel):
    listings: List[PropertyListingRequest] = Field(..., min_length=1, max_length=config.API_MAX_BATCH_SIZE)


class PredictionResult(BaseModel):
    index: int
    Neighborhood: str
    FairValue: float
    AnomalyZScore: float
    FeatureContributions: Dict[str, float]
    Warnings: List[str] = []


class UnderwriteResult(BaseModel):
    index: int
    Neighborhood: str
    ListPrice: float
    FairValue: float
    DiscountPct: float
    AnomalyZScore: float
    Decision: str
    RecommendedBidLow: float
    RecommendedBidHigh: float
    StressedLTVLimit: float
    Warnings: List[str] = []


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str]
    model_bundle_version: Optional[str]
    model_loaded_at: Optional[str]
    fred_api_configured: bool
    fred_api_reachable: Optional[bool]
    load_error: Optional[str]


# --------------------------------------------------------------------------- #
# Startup: load the serialized bundle ONCE, initialize the database
# --------------------------------------------------------------------------- #
@app.on_event("startup")
def load_model_artifacts() -> None:
    logger.info("Loading model bundle from %s ...", config.MODEL_ARTIFACT_PATH)
    try:
        bundle = joblib.load(config.MODEL_ARTIFACT_PATH)
        model_state.bundle = bundle
        model_state.loaded_at = dt.datetime.now(dt.timezone.utc).isoformat()
        model_state.load_error = None
        logger.info("Model bundle loaded: %s (exported %s)", bundle["model_name"], bundle["exported_at"])
    except Exception as e:
        model_state.bundle = None
        model_state.load_error = str(e)
        logger.error(
            "FAILED to load model bundle from %s: %s. Run `python export_pipeline.py` first.",
            config.MODEL_ARTIFACT_PATH, e,
        )
    database.init_db()


def _require_model() -> Dict[str, Any]:
    if model_state.bundle is None:
        raise HTTPException(status_code=503, detail=f"Model artifacts not loaded: {model_state.load_error}")
    return model_state.bundle


# --------------------------------------------------------------------------- #
# Shared frozen-pipeline inference helper -- no transformer is ever refit
# here; every one is the exact fitted object from export_pipeline.py.
# --------------------------------------------------------------------------- #
def _prep_and_score(listings: List[PropertyListingRequest], bundle: Dict[str, Any]):
    schema = bundle["schema"]
    feature_cols = bundle["feature_cols"]

    records = [l.model_dump() for l in listings]
    warnings_per_row: List[List[str]] = []
    for rec in records:
        errs = validate_payload(rec, schema)
        if errs:
            raise HTTPException(status_code=422, detail=errs)
        w = []
        nbhd_warning = unseen_neighborhood_warning(rec, schema)
        if nbhd_warning:
            w.append(nbhd_warning)
        warnings_per_row.append(w)

    df = pd.DataFrame(records).rename(columns={"ListingDate": config.DATE_COL})
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])

    X = RatioFeatureBuilder().transform(df[feature_cols + [config.DATE_COL]])
    X = MacroSeriesResampler(bundle["macro_frame"]).transform(X)
    X = X.drop(columns=[config.DATE_COL])
    X = bundle["winsorizer"].transform(X)
    X = bundle["isolation_forest"].transform(X)
    X = bundle["target_encoder"].transform(X)

    model = bundle["model"]
    numeric_cols = bundle["numeric_cols"]
    if bundle["model_name"] == "ridge_spatial":
        log_fair_value = model.predict(X[numeric_cols])
    else:
        log_fair_value = model.predict(X)
    fair_value = np.expm1(log_fair_value)

    anomaly_score = X["is_outlier_score"].to_numpy()
    anomaly_z = (anomaly_score - bundle["baseline_anomaly_mean"]) / bundle["baseline_anomaly_std"]

    return fair_value, anomaly_z, X, warnings_per_row


def _feature_contributions(x_row: pd.Series, bundle: Dict[str, Any]) -> Dict[str, float]:
    """
    Per-request linear feature contributions for a fitted Ridge pipeline
    (coefficient * transformed feature value). Falls back to the model's
    global feature_importances_ (e.g. XGBoost/LightGBM) if the exported
    model isn't linear -- that fallback is a GLOBAL importance, not a
    per-instance explanation.
    """
    model = bundle["model"]
    numeric_cols = bundle["numeric_cols"]
    if bundle["model_name"] == "ridge_spatial":
        try:
            ct = model.named_steps["features"]
            ridge = model.named_steps["ridge"]
            feature_names = ct.get_feature_names_out()
            x_transformed = ct.transform(pd.DataFrame([x_row[numeric_cols]]))[0]
            contributions = ridge.coef_ * x_transformed
            return {name: float(val) for name, val in zip(feature_names, contributions)}
        except Exception as e:
            logger.warning("Could not compute per-instance Ridge contributions: %s", e)
            return {}
    if hasattr(model, "feature_importances_"):
        return {col: float(imp) for col, imp in zip(numeric_cols, model.feature_importances_)}
    return {}


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    fred_configured = bool(os.environ.get(config.FRED_API_KEY_ENV_VAR))
    fred_reachable: Optional[bool] = None
    if fred_configured:
        try:
            client = FredMacroClient()
            client.fetch_series("MORTGAGE30US", "2020-01-01", "2020-01-31")
            fred_reachable = True
        except Exception:
            fred_reachable = False
    return HealthResponse(
        status="ok" if model_state.bundle is not None else "degraded",
        model_loaded=model_state.bundle is not None,
        model_name=model_state.bundle["model_name"] if model_state.bundle else None,
        model_bundle_version=model_state.bundle["bundle_version"] if model_state.bundle else None,
        model_loaded_at=model_state.loaded_at,
        fred_api_configured=fred_configured,
        fred_api_reachable=fred_reachable,
        load_error=model_state.load_error,
    )


@app.post("/predict", response_model=List[PredictionResult])
def predict(batch: PredictBatchRequest, db: Session = Depends(database.get_db)) -> List[PredictionResult]:
    bundle = _require_model()
    try:
        fair_value, anomaly_z, X, warnings_per_row = _prep_and_score(batch.listings, bundle)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("/predict pipeline error")
        raise HTTPException(status_code=500, detail=f"Inference pipeline error: {e}")

    results = []
    for i, listing in enumerate(batch.listings):
        payload = listing.model_dump()
        listing_row = database.log_listing(db, payload)
        database.log_underwriting_evaluation(
            db, listing_id=listing_row.id, endpoint="predict", model_name=bundle["model_name"],
            model_bundle_version=bundle["bundle_version"], fair_value=float(fair_value[i]),
            anomaly_z_score=float(anomaly_z[i]),
        )
        results.append(PredictionResult(
            index=i, Neighborhood=listing.Neighborhood, FairValue=round(float(fair_value[i]), 2),
            AnomalyZScore=round(float(anomaly_z[i]), 3),
            FeatureContributions={k: round(v, 2) for k, v in _feature_contributions(X.iloc[i], bundle).items()},
            Warnings=warnings_per_row[i],
        ))
    return results


@app.post("/underwrite", response_model=List[UnderwriteResult])
def underwrite(batch: PredictBatchRequest, db: Session = Depends(database.get_db)) -> List[UnderwriteResult]:
    bundle = _require_model()
    for listing in batch.listings:
        if listing.ListPrice is None:
            raise HTTPException(status_code=422, detail="ListPrice is required for /underwrite requests.")

    try:
        fair_value, anomaly_z, X, warnings_per_row = _prep_and_score(batch.listings, bundle)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("/underwrite pipeline error")
        raise HTTPException(status_code=500, detail=f"Inference pipeline error: {e}")

    # Real macro shock (same shock config.py uses in stress_test.py),
    # applied to this SAME frozen pipeline, to derive a stress-tested LTV
    # limit. This is a documented heuristic for a demo underwriting
    # workflow -- not a real credit/lending standard; see README.
    model = bundle["model"]
    numeric_cols = bundle["numeric_cols"]
    shocked_X = X.copy()
    shocked_X["MortgageRate30Y"] = shocked_X["MortgageRate30Y"] + config.MACRO_SHOCK_RATE_BPS / 100.0
    shocked_X["LocalUnemploymentRate"] = shocked_X["LocalUnemploymentRate"] + config.MACRO_SHOCK_UNEMPLOYMENT_PP
    if bundle["model_name"] == "ridge_spatial":
        shocked_log_fv = model.predict(shocked_X[numeric_cols])
    else:
        shocked_log_fv = model.predict(shocked_X)
    stressed_fair_value = np.expm1(shocked_log_fv)

    results = []
    for i, listing in enumerate(batch.listings):
        list_price = listing.ListPrice
        discount_pct = (fair_value[i] - list_price) / max(list_price, 1.0)
        is_discounted = discount_pct > config.SCREENING_DISCOUNT_THRESHOLD
        is_anomalous = anomaly_z[i] >= config.SCREENING_ANOMALY_Z_THRESHOLD
        if not is_discounted:
            decision = "REJECT"
        elif not is_anomalous:
            decision = "BUY"
        else:
            decision = "MANUAL_APPRAISAL"

        bid_low = fair_value[i] * config.UNDERWRITING_BID_LOW_PCT
        bid_high = fair_value[i] * config.UNDERWRITING_BID_HIGH_PCT
        # Stressed LTV limit: how much loan the STRESS-SHOCKED valuation
        # supports relative to the asking price, capped at the
        # conventional ceiling -- never higher than config.UNDERWRITING_MAX_LTV.
        coverage_ratio = max(0.0, stressed_fair_value[i] / list_price)
        stressed_ltv = min(config.UNDERWRITING_MAX_LTV, coverage_ratio * config.UNDERWRITING_MAX_LTV)

        payload = listing.model_dump()
        listing_row = database.log_listing(db, payload)
        database.log_underwriting_evaluation(
            db, listing_id=listing_row.id, endpoint="underwrite", model_name=bundle["model_name"],
            model_bundle_version=bundle["bundle_version"], fair_value=float(fair_value[i]),
            anomaly_z_score=float(anomaly_z[i]), discount_pct=float(discount_pct), decision=decision,
            recommended_bid_low=float(bid_low), recommended_bid_high=float(bid_high),
            stressed_ltv_limit=float(stressed_ltv),
        )

        results.append(UnderwriteResult(
            index=i, Neighborhood=listing.Neighborhood, ListPrice=list_price,
            FairValue=round(float(fair_value[i]), 2), DiscountPct=round(float(discount_pct) * 100, 2),
            AnomalyZScore=round(float(anomaly_z[i]), 3), Decision=decision,
            RecommendedBidLow=round(float(bid_low), 2), RecommendedBidHigh=round(float(bid_high), 2),
            StressedLTVLimit=round(float(stressed_ltv), 4), Warnings=warnings_per_row[i],
        ))
    return results
