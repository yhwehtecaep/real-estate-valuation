"""
database.py
============
SQLAlchemy database interface for the Real Estate Valuation API.
SQLite by default (zero-setup, file-based); set DATABASE_URL to point at
Postgres or any other SQLAlchemy-supported backend in production --
nothing else in this module changes.

Tables
------
Listings                 : raw incoming property payloads + metadata, one
                            row per /predict or /underwrite request.
UnderwritingEvaluations  : model outputs for each listing (fair value,
                            anomaly z-score, decision, recommended bid
                            range, stressed LTV limit), one row per
                            /underwrite request, foreign-keyed to Listings.

Every write goes through `log_listing()` / `log_underwriting_evaluation()`,
called from api_service.py on every request for a complete audit trail.
"""

import datetime as dt
import json
import logging
import os
from typing import Any, Dict, Generator, Optional

from sqlalchemy import (
    Column, Integer, Float, String, DateTime, ForeignKey, Text, create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

import config

logger = logging.getLogger("real_estate_valuation.database")

Base = declarative_base()


class Listing(Base):
    """One row per property payload received by /predict or /underwrite."""
    __tablename__ = "listings"

    id = Column(Integer, primary_key=True, index=True)
    received_at = Column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc), index=True)

    neighborhood = Column(String, index=True, nullable=False)
    listing_date = Column(String, nullable=False)   # ISO date string, as received
    overall_qual = Column(Float, nullable=False)
    overall_cond = Column(Float, nullable=False)
    living_area_sqft = Column(Float, nullable=False)
    lot_area_sqft = Column(Float, nullable=False)
    garage_area_sqft = Column(Float, nullable=False)
    year_built = Column(Float, nullable=False)
    year_renovated = Column(Float, nullable=False)
    bedroom_abvgr = Column(Float, nullable=False)
    full_bath = Column(Float, nullable=False)
    totrms_abvgrd = Column(Float, nullable=False)
    near_railroad = Column(Integer, nullable=False)
    near_artery = Column(Integer, nullable=False)
    near_positive_feature = Column(Integer, nullable=False)
    list_price = Column(Float, nullable=True)   # only present for /underwrite requests

    raw_payload_json = Column(Text, nullable=False)   # full original JSON, for audit

    evaluations = relationship("UnderwritingEvaluation", back_populates="listing")


class UnderwritingEvaluation(Base):
    """One row per model evaluation (both /predict-only and /underwrite calls log here)."""
    __tablename__ = "underwriting_evaluations"

    id = Column(Integer, primary_key=True, index=True)
    listing_id = Column(Integer, ForeignKey("listings.id"), nullable=False, index=True)
    evaluated_at = Column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc), index=True)

    endpoint = Column(String, nullable=False)          # "predict" or "underwrite"
    model_name = Column(String, nullable=False)
    model_bundle_version = Column(String, nullable=False)

    fair_value = Column(Float, nullable=False)
    anomaly_z_score = Column(Float, nullable=False)

    discount_pct = Column(Float, nullable=True)         # /underwrite only
    decision = Column(String, nullable=True)             # BUY / MANUAL_APPRAISAL / REJECT
    recommended_bid_low = Column(Float, nullable=True)
    recommended_bid_high = Column(Float, nullable=True)
    stressed_ltv_limit = Column(Float, nullable=True)

    listing = relationship("Listing", back_populates="evaluations")


# --------------------------------------------------------------------------- #
# Engine / session management
# --------------------------------------------------------------------------- #
def _resolve_database_url() -> str:
    """
    Falls back to config.DEFAULT_DATABASE_URL when DATABASE_URL is either
    absent OR present-but-empty (os.environ.get alone only handles the
    absent case -- an empty string is a valid env var value and would
    otherwise reach SQLAlchemy's create_engine() and fail to parse).
    """
    url = os.environ.get(config.DATABASE_URL_ENV_VAR, "").strip()
    return url or config.DEFAULT_DATABASE_URL


def build_engine(database_url: Optional[str] = None):
    """
    Builds a SQLAlchemy engine. For SQLite, `check_same_thread=False` is
    required because FastAPI/uvicorn serve requests from a thread pool --
    each request still gets its own Session (see `get_db`), so this is
    safe: SQLite's own file locking still serializes concurrent writes.
    """
    url = database_url or _resolve_database_url()
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    if url.startswith("sqlite:///") and url != "sqlite:///:memory:":
        db_path = url.replace("sqlite:///", "", 1)
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    engine = create_engine(url, connect_args=connect_args, future=True)
    return engine


engine = build_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    """Creates all tables if they don't exist. Called once on API startup."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized at %s", _resolve_database_url())


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yields a request-scoped Session, always closed after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# Logging helpers -- called from api_service.py on every request
# --------------------------------------------------------------------------- #
def log_listing(db: Session, payload: Dict[str, Any]) -> Listing:
    """Persists one incoming property payload. Returns the created Listing row (with id)."""
    listing = Listing(
        neighborhood=payload["Neighborhood"],
        listing_date=str(payload[config.DATE_COL]),
        overall_qual=float(payload["OverallQual"]),
        overall_cond=float(payload["OverallCond"]),
        living_area_sqft=float(payload["LivingAreaSqFt"]),
        lot_area_sqft=float(payload["LotAreaSqFt"]),
        garage_area_sqft=float(payload["GarageAreaSqFt"]),
        year_built=float(payload["YearBuilt"]),
        year_renovated=float(payload["YearRenovated"]),
        bedroom_abvgr=float(payload["BedroomAbvGr"]),
        full_bath=float(payload["FullBath"]),
        totrms_abvgrd=float(payload["TotRmsAbvGrd"]),
        near_railroad=int(payload["NearRailroad"]),
        near_artery=int(payload["NearArtery"]),
        near_positive_feature=int(payload["NearPositiveFeature"]),
        list_price=float(payload["ListPrice"]) if payload.get("ListPrice") is not None else None,
        raw_payload_json=json.dumps(payload, default=str),
    )
    db.add(listing)
    db.commit()
    db.refresh(listing)
    return listing


def log_underwriting_evaluation(
    db: Session, listing_id: int, endpoint: str, model_name: str, model_bundle_version: str,
    fair_value: float, anomaly_z_score: float, discount_pct: Optional[float] = None,
    decision: Optional[str] = None, recommended_bid_low: Optional[float] = None,
    recommended_bid_high: Optional[float] = None, stressed_ltv_limit: Optional[float] = None,
) -> UnderwritingEvaluation:
    """Persists one model evaluation, foreign-keyed to its Listing row."""
    evaluation = UnderwritingEvaluation(
        listing_id=listing_id, endpoint=endpoint, model_name=model_name,
        model_bundle_version=model_bundle_version, fair_value=fair_value,
        anomaly_z_score=anomaly_z_score, discount_pct=discount_pct, decision=decision,
        recommended_bid_low=recommended_bid_low, recommended_bid_high=recommended_bid_high,
        stressed_ltv_limit=stressed_ltv_limit,
    )
    db.add(evaluation)
    db.commit()
    db.refresh(evaluation)
    return evaluation
