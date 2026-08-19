"""
config.py
=========
Centralized configuration for the Real Estate Valuation & Pricing Model.

Every tunable constant (data paths, column names, hyperparameter search
bounds, CV settings) lives here so that `data_pipeline.py`, `models.py`,
`stress_test.py`, and `main.py` never hard-code magic numbers.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import os

from dotenv import load_dotenv

# Loads variables from a local .env file (if present) into os.environ,
# BEFORE anything below reads an env var. This is the one place it's
# called -- every other module gets it for free by importing config first.
# Safe to call even with no .env file (no-op); never overwrites a variable
# already set in the real environment (e.g. by `docker run -e ...` or
# `export ...`) -- see .env.example for the full list of supported keys.
load_dotenv()

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
PLOT_DIR = os.path.join(ARTIFACT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Real, historical De Cock (2011) Ames, Iowa housing dataset -- 2,930 actual
# residential property sales, 2006-2010. Bundled as a local CSV (fetched
# once from a public GitHub mirror of the original dataset). No synthetic
# rows, no synthetic columns -- see data_pipeline.py for the exact source.
AMES_CSV_PATH = os.path.join(BASE_DIR, "ames_raw.csv")

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Schema (all real columns from the Ames, Iowa Assessor's Office dataset,
# De Cock 2011 -- https://jse.amstat.org/v19n3/decock.pdf)
# --------------------------------------------------------------------------- #
TARGET_RAW = "SalePrice"            # dollars, right-skewed (real, as sold)
TARGET_LOG = "LogSalePrice"         # log1p(SalePrice) -- actual model target

ID_COL = "Order"                    # real row identifier from the dataset
GROUP_COL = "Neighborhood"          # real Ames neighborhood -- spatial grouping key
DATE_COL = "ListingDate"            # built from real Yr Sold / Mo Sold fields

# Real per-property numeric features, always present directly from the raw
# Ames data (renamed to snake-free identifiers; original De Cock column
# names are mapped in data_pipeline.py)
PROPERTY_NUMERIC_FEATURES = [
    "OverallQual", "OverallCond",
    "LivingAreaSqFt", "LotAreaSqFt", "GarageAreaSqFt",
    "YearBuilt", "YearRenovated",
    "BedroomAbvGr", "FullBath", "TotRmsAbvGrd",
    "NearRailroad", "NearArtery", "NearPositiveFeature",   # from real Condition1/2
]

# Real macro features, joined fold-safely by MacroSeriesResampler from
# live-or-cached-fallback FRED data (see data_pipeline.get_macro_series)
MACRO_FEATURES = ["MortgageRate30Y", "CPI", "LocalUnemploymentRate"]

# Real live-geocoded spatial features (OpenStreetMap Nominatim + osmnx).
# Only included in the actual model feature set for a given run if they
# come back non-NaN (i.e. live network access was available) -- see
# main.py's feature-availability probe. Never fabricated as a fallback.
OPTIONAL_GEO_FEATURES = [
    "DistToCommercialCenterKm", "DistToISUKm",
    "OSMNearHighwayM", "OSMNearRailM", "OSMNearParkM",
]

# Guaranteed numeric feature set (always available, no live network needed)
NUMERIC_FEATURES = PROPERTY_NUMERIC_FEATURES + MACRO_FEATURES

CATEGORICAL_HIGH_CARD = ["Neighborhood"]   # target-encoded, real 28-category field
CATEGORICAL_LOW_CARD: List[str] = []

# Engineered ratio / interaction features created in data_pipeline.py from
# the real columns above (arithmetic only -- no new information injected)
ENGINEERED_FEATURES = [
    "LivingToLotRatio",
    "AgeAtSale",
    "YearsSinceRenovation",
]

OPTIONAL_ATTRS_FOR_MASKING = ["GarageAreaSqFt", "YearRenovated", "LotAreaSqFt"]

# --------------------------------------------------------------------------- #
# Outlier handling
# --------------------------------------------------------------------------- #
WINSOR_LOWER_PCT = 0.01
WINSOR_UPPER_PCT = 0.99
ISOLATION_FOREST_CONTAMINATION = 0.02

# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
N_SPATIAL_FOLDS = 5          # grouped by Neighborhood cluster
N_TIME_SPLITS = 5            # for GroupTimeSeriesSplit ordering by ListingDate

# --------------------------------------------------------------------------- #
# Optuna / model hyperparameter search spaces
# --------------------------------------------------------------------------- #
N_OPTUNA_TRIALS = 25         # kept modest for reproducible runtime; raise for production

@dataclass
class SearchSpace:
    max_depth: Tuple[int, int] = (3, 6)                # forced conservative depth
    learning_rate: Tuple[float, float] = (0.01, 0.1)
    n_estimators: Tuple[int, int] = (200, 800)
    reg_alpha: Tuple[float, float] = (1e-3, 10.0)       # L1
    reg_lambda: Tuple[float, float] = (1e-3, 10.0)      # L2
    subsample: Tuple[float, float] = (0.6, 0.8)         # <= 0.8 per spec
    colsample_bytree: Tuple[float, float] = (0.6, 0.8)  # <= 0.8 per spec
    min_child_weight: Tuple[int, int] = (1, 10)

SEARCH_SPACE = SearchSpace()

EARLY_STOPPING_ROUNDS = 50

RIDGE_ALPHA_GRID = [0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]
# No lat/lon in the raw Ames data (only later derivative packages geocode
# it), so the Ridge baseline's polynomial expansion is applied to the real
# OverallQual/OverallCond quality grades instead of spatial coordinates.
RIDGE_POLY_COLUMNS = ["OverallQual", "OverallCond"]
RIDGE_POLY_DEGREE = 2

# --------------------------------------------------------------------------- #
# Stress testing
# --------------------------------------------------------------------------- #
MACRO_SHOCK_RATE_BPS = 300          # +300 bps mortgage rate shock
MACRO_SHOCK_UNEMPLOYMENT_PP = 3.0   # +3 percentage-point unemployment shock
                                     # (recession scenario; the real Ames data
                                     # has no per-property local-income series,
                                     # so the income-shock leg of the spec is
                                     # implemented via the real national
                                     # unemployment-rate series instead -- see
                                     # README for this substitution)
MASK_FRACTION = 0.30                # fraction of rows with an attribute masked
N_MASK_TRIALS = 20                  # repetitions for stable degradation estimate

# Synthetic EDGE-CASE PROBES for stress testing only (never used as training
# data): the spec explicitly calls for "out-of-bounds physical features" such
# as a 10,000 sqft home with 1 bedroom, constructed by overriding a real
# template row to test extrapolation behavior.
OOD_TEST_CASES: List[Dict] = [
    {"name": "huge_house_one_bed", "LivingAreaSqFt": 10000, "BedroomAbvGr": 1, "TotRmsAbvGrd": 3},
    {"name": "tiny_house_many_beds", "LivingAreaSqFt": 300, "BedroomAbvGr": 8, "TotRmsAbvGrd": 10},
    {"name": "zero_lot_area", "LotAreaSqFt": 1},
    {"name": "ancient_never_renovated", "YearBuilt": 1875, "YearRenovated": 1875},
]

# Prediction-interval half-width (in log space) beyond which a prediction is
# flagged as "high uncertainty" during OOD testing. Derived from residual
# std on the validation folds at runtime; this is just the multiplier.
OOD_FLAG_SIGMA_MULTIPLIER = 2.0

# --------------------------------------------------------------------------- #
# Live data source configuration
# --------------------------------------------------------------------------- #
# FRED (Federal Reserve Economic Data) -- free, zero-cost API. Register a key
# at https://fred.stlouisfed.org/docs/api/api_key.html and export it as:
#     export FRED_API_KEY="your_key_here"
# If unset or unreachable, data_pipeline.py falls back to a cached table of
# the same real, published FRED series (never simulated values) so the
# pipeline still runs -- see FredMacroClient in data_pipeline.py.
FRED_API_KEY_ENV_VAR = "FRED_API_KEY"
FRED_SERIES = {
    "MortgageRate30Y": "MORTGAGE30US",   # weekly, Freddie Mac PMMS
    "LocalUnemploymentRate": "UNRATE",    # monthly, BLS, seasonally adjusted
    "CPI": "CPIAUCNS",                    # monthly, BLS, not seasonally adjusted
}
FRED_REQUEST_TIMEOUT_S = 15
FRED_MAX_RETRIES = 3
FRED_RETRY_BACKOFF_S = 2.0

# OpenStreetMap geocoding (Nominatim, via geopy) -- free, zero-cost, subject
# to Nominatim's usage policy (max ~1 request/sec, valid User-Agent required).
NOMINATIM_USER_AGENT = "real-estate-valuation-model (contact: analytics@example.com)"
NOMINATIM_TIMEOUT_S = 5
NOMINATIM_MIN_DELAY_S = 1.0   # politeness delay between requests

# Real, verified reference landmarks in Ames, Iowa (public, documented
# coordinates -- Wikipedia / NRHP listings), used as the fixed "commercial
# center" / "university" anchor points for geodesic distance features.
AMES_CBD_LATLON = (42.02556, -93.61361)      # Ames Main Street Historic District (NRHP)
ISU_CAMPUS_LATLON = (42.02662, -93.64647)    # Iowa State University campus center

# osmnx / OSM infrastructure proximity search radius around each reference
# point when pulling real highway/rail/park geometries.
OSM_SEARCH_RADIUS_M = 3000

# --------------------------------------------------------------------------- #
# Production holdout
# --------------------------------------------------------------------------- #
HOLDOUT_FRACTION = 0.10   # immutable final test set, split off before any
                           # tuning or CV; never touched until final reporting

# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
PRICE_DECILES = 10

# --------------------------------------------------------------------------- #
# Live deal screening / automated underwriting (screening_engine.py)
# --------------------------------------------------------------------------- #
SCREENING_DISCOUNT_THRESHOLD = 0.12    # Delta = (fair_value - list_price) / list_price
SCREENING_ANOMALY_Z_THRESHOLD = 2.0    # isolation-forest anomaly z-score cutoff

# --------------------------------------------------------------------------- #
# Model artifact persistence (export_pipeline.py / api_service.py)
# --------------------------------------------------------------------------- #
MODEL_ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts", "model_bundle")
MODEL_ARTIFACT_PATH = os.path.join(MODEL_ARTIFACT_DIR, "pipeline_bundle.joblib")
MODEL_BUNDLE_VERSION = "1.0.0"

# --------------------------------------------------------------------------- #
# REST API service (api_service.py)
# --------------------------------------------------------------------------- #
API_TITLE = "Real Estate Valuation & Underwriting API"
API_VERSION = MODEL_BUNDLE_VERSION
API_MAX_BATCH_SIZE = 200

# Underwriting heuristics for /underwrite (config.py is the single place
# these live, so they're auditable/tunable, not buried in endpoint code).
# These are simple, documented heuristics for a demo underwriting workflow,
# not a real credit/lending standard -- see README disclaimer.
UNDERWRITING_BID_LOW_PCT = 0.95    # recommended bid range: [fair_value * LOW, fair_value * HIGH]
UNDERWRITING_BID_HIGH_PCT = 1.00
UNDERWRITING_MAX_LTV = 0.80        # conventional conforming-loan ceiling, used as the cap

# --------------------------------------------------------------------------- #
# Database (database.py) -- SQLAlchemy, SQLite by default, Postgres-compatible
# --------------------------------------------------------------------------- #
DATABASE_URL_ENV_VAR = "DATABASE_URL"
DEFAULT_SQLITE_PATH = os.path.join(BASE_DIR, "artifacts", "underwriting.db")
DEFAULT_DATABASE_URL = f"sqlite:///{DEFAULT_SQLITE_PATH}"

