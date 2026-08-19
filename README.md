# Real Estate Valuation, Automated Screening & Production API

An enterprise-grade, production-ready pipeline **and containerized REST
API** for real-estate investment evaluation and live deal screening, built
on **real historical and live data only** (no synthetic rows or columns
anywhere in the training data), with leakage-safe cross-validation, an
immutable pre-deployment holdout set, an automated underwriting engine,
model artifact persistence, and a SQL-backed audit trail.

## Configuration (.env)

```bash
cp .env.example .env
# then edit .env and fill in what you want -- both are optional, see below
```

`config.py` loads `.env` automatically (via `python-dotenv`) before anything
reads an environment variable, so every script below (`main.py`,
`export_pipeline.py`, `api_service.py`) picks it up for free -- nothing
else to configure. `.env` is gitignored/dockerignored, so real keys never
get committed or baked into the image.

| Variable | Required? | Effect if unset |
|---|---|---|
| `FRED_API_KEY` | No | Falls back to real cached historical FRED figures (2006-2010) instead of live data. Free key: https://fred.stlouisfed.org/docs/api/api_key.html |
| `DATABASE_URL` | No | Defaults to a local SQLite file (`artifacts/underwriting.db`). Set to a Postgres URL to use that instead. |

OpenStreetMap Nominatim (neighborhood geocoding) and the OSM Overpass API
(highway/rail/park proximity) need no key at all -- just outbound network
access.

## Quick start -- training pipeline

```bash
pip install -r requirements.txt
python main.py
```

Live geocoding (OpenStreetMap Nominatim) and OSM infrastructure buffers
(osmnx) need outbound network access to `nominatim.openstreetmap.org` and
the Overpass API; no separate API key is required for those, just network
access and a polite request rate (already built in). If either live
source is unreachable, the pipeline logs one clear warning per source and
degrades gracefully -- it never fabricates a value in place of a live one.
See **Live data sources & graceful degradation** below.

Runtime is a few minutes on the full dataset with the default 25 Optuna
trials / 5 spatial-temporal folds per model.

## Quick start -- production API

```bash
pip install -r requirements.txt
python export_pipeline.py                 # trains + serializes the model bundle (once)
uvicorn api_service:app --host 0.0.0.0 --port 8000

# in another shell:
python test_client.py
```

Or via Docker (see **Deployment** below).

## Files

| File | Responsibility |
|---|---|
| `ames_raw.csv` | Real De Cock (2011) Ames, Iowa housing dataset -- 2,930 actual property sales, 2006-2010 |
| `config.py` | All paths, schema, live-API settings, hyperparameter search bounds, CV/holdout/screening/underwriting/API/DB settings |
| `data_pipeline.py` | Real-data ingestion; live FRED macro client with cached-real fallback; live OSM geocoding + infrastructure enrichment with graceful degradation; leakage-safe transformers |
| `models.py` | `GroupTimeSeriesSplit` (spatial + temporal CV), fold-safe preprocessing (incl. macro/geo alignment), Optuna tuning, Ridge baseline, metrics |
| `stress_test.py` | `StressTester`: macro shocks, missing-attribute sensitivity, out-of-distribution physical features, residual diagnostics |
| `screening_engine.py` | `DealScreener`: live deal screening & automated underwriting against a frozen, already-fitted model |
| `main.py` | Executable end-to-end training workflow, including the immutable holdout split and a screening demo |
| `export_pipeline.py` | Trains the winning model on the full dev set and serializes every fitted artifact into a single joblib bundle |
| `schema_utils.py` | Shared input-schema contract (`FeatureSchema`) and payload validation, used by both `export_pipeline.py` and `api_service.py` |
| `api_service.py` | FastAPI service: `/health`, `/predict`, `/underwrite`, backed by the exported bundle and the database audit trail |
| `database.py` | SQLAlchemy ORM (`Listings`, `UnderwritingEvaluations`) + session management, SQLite by default / Postgres-compatible |
| `Dockerfile` | Production container: Python 3.11, trains+exports the bundle at build time if missing, serves via uvicorn on port 8000 |
| `test_client.py` | Standalone script exercising `/health`, `/predict`, and `/underwrite` against a running API instance |
| `requirements.txt` | Full dependency list for the training pipeline and the API service |

## What's new in this upgrade

### 1. Dynamic live macro ingestion (`FredMacroClient`, `MacroSeriesResampler`)

`data_pipeline.FredMacroClient` wraps `fredapi` to pull `MORTGAGE30US`,
`UNRATE`, and `CPIAUCNS` live from the free FRED API, with retries and
exponential backoff (`config.FRED_MAX_RETRIES` / `FRED_RETRY_BACKOFF_S`).
If `FRED_API_KEY` is unset or the API is unreachable,
`get_macro_series()` logs a clear warning and falls back to a cached table
of the *same real, published* FRED figures (never a simulated value) so
the pipeline keeps running.

Frequency alignment (the mortgage-rate series is weekly, CPI/unemployment
are monthly, and properties sell on arbitrary days) is handled by
`MacroSeriesResampler`, an as-of (backward-fill) join with linear
interpolation for any remaining gaps. It's invoked from inside
`models.preprocess_fold()` on every CV fold, matching the leakage-safe
architecture of the target encoder and winsorizer -- see the note in
`data_pipeline.py`'s module docstring on why this particular join carries
no actual leakage risk regardless of fold-scoping (the macro series is
exogenous public data, independent of y or of fold membership), but is
fold-scoped anyway for architectural consistency.

### 2. Live geo-spatial enrichment (`GeoEnricher`, `SpatialInfrastructureEngineer`)

The raw Ames dataset has no per-property Latitude/Longitude (only later
derivative R packages geocode it, as compiled binary data with no plain-CSV
mirror we could verify as authentic). Instead, `GeoEnricher` live-geocodes
the real `Neighborhood` field (28 real Ames neighborhoods) via OpenStreetMap
Nominatim (through `geopy`) and computes real geodesic distances to two
fixed, verified landmarks -- downtown Ames (NRHP-listed Main Street
Historic District) and the Iowa State University campus
(`config.AMES_CBD_LATLON` / `ISU_CAMPUS_LATLON`). `SpatialInfrastructureEngineer`
then queries live OpenStreetMap data via `osmnx` for the real nearest
highway, railway, and park within `config.OSM_SEARCH_RADIUS_M` of each
geocoded centroid.

Both are invoked from inside `models.preprocess_fold()` per the spec, with
results cached in-process (geocoding a fixed place name is invariant
across folds). If live network access to Nominatim/OSM is unavailable,
each logs **one** clear warning, trips an internal fast-fail latch (so it
doesn't retry per-neighborhood), and returns `NaN` for those columns --
never a fabricated coordinate. `main.py` probes availability once at
startup and only wires these columns into the feature set if they came
back with real values; otherwise the real per-property `Condition
1`/`Condition 2` proximity flags (`NearRailroad`, `NearArtery`,
`NearPositiveFeature` -- the Assessor's own real adjacency encoding) carry
the spatial-infrastructure signal instead, so the pipeline is 100%
executable with or without live network access.

### 3. Live deal screening & automated underwriting (`screening_engine.py`)

`DealScreener` scores a real batch of candidate listings through the
frozen, already-fitted pipeline (no transformer is refit -- same
philosophy as `StressTester`) and produces an Investment Ranking Table:

- **Fair Value** ($\hat P_{fair}$, from the model) vs. **Listing Price** ($P_{list}$)
- **Discount %**: $\Delta = (\hat P_{fair} - P_{list}) / P_{list}$
- **Anomaly Z-Score**, from the frozen isolation forest, standardized
  against the real training-fold anomaly-score distribution
- **Decision**:
  - `BUY` — Discount > 12% **and** anomaly z-score < 2.0
  - `MANUAL_APPRAISAL` — Discount > 12% **but** anomaly z-score ≥ 2.0
    (a real discount, but on a physically unusual property, so the
    fair-value estimate itself is less trustworthy)
  - `REJECT` — Discount ≤ 12%

`main.py` demonstrates this on a real sample batch drawn from the holdout
set (see the inline note in `main.py`: their real recorded `SalePrice`
stands in for "current asking price" purely for the demo, since this repo
has no live MLS feed; in production, `listings` would be a real feed of
active/incoming asking prices).

### 4. Immutable holdout test set (`main.py`)

Before any Optuna tuning or cross-validation runs, `main.py` splits off the
most recent **10%** of real sales by date (`config.HOLDOUT_FRACTION`) as an
immutable holdout set. It is never touched again until the very end, when
the winning model (selected purely from dev-set CV) is refit once on the
*full* dev set and scored exactly once on the untouched holdout — the
final, unbiased, pre-deployment metric.

### 5. Model artifact serialization (`export_pipeline.py`)

`export_pipeline.py` trains the winning model on the full real dev set
(same `GroupTimeSeriesSplit` procedure) and serializes a single joblib
bundle (`config.MODEL_ARTIFACT_PATH`) containing: the fitted regressor,
the fold-fitted `Winsorizer` / `IsolationForestFlagger` /
`KFoldTargetEncoder`, the real macro series used at export time,
reference anomaly-score statistics (mean/std) for live z-score
standardization, the final unbiased holdout metrics, and a
`schema_utils.FeatureSchema` — the frozen input-schema contract built from
the real training data (which real columns, their types, and the 28 real
neighborhoods actually observed).

```bash
python export_pipeline.py                    # exports the verified winner (Ridge)
python export_pipeline.py --model xgboost     # force a specific model type
python export_pipeline.py --reselect          # re-run the full CV comparison and export the winner
```

### 6. Production REST API (`api_service.py`)

An async FastAPI service that loads the exported bundle **once** at
startup (`@app.on_event("startup")`) into a module-level, read-only
`ModelState` — safe for concurrent requests across uvicorn's thread pool
without any locking, since nothing in the bundle is mutated per-request.

- **`GET /health`** — model load state, bundle version, and (if
  `FRED_API_KEY` is set) a live FRED reachability check.
- **`POST /predict`** — batch of `PropertyListingRequest`s in, real fair
  value + anomaly z-score + per-request feature contributions out. For
  the exported Ridge model, contributions are exact linear
  `coefficient × transformed feature value` decompositions; for a
  tree-model export, this falls back to the model's global
  `feature_importances_` (labeled as such, since that's a global, not
  per-instance, explanation).
- **`POST /underwrite`** — same as `/predict` plus a real `ListPrice`,
  returns the full deal-screening decision (`BUY` / `MANUAL_APPRAISAL` /
  `REJECT`), a recommended bid range (`config.UNDERWRITING_BID_LOW_PCT`/
  `HIGH_PCT` of fair value), and a **stress-tested LTV limit** — the
  conventional LTV ceiling (`config.UNDERWRITING_MAX_LTV`) scaled down by
  how much the model's fair value *after* the same real macro shock used
  in `stress_test.py` (+300bps mortgage rate, +3pp unemployment) still
  covers the asking price. **This is a documented heuristic for a demo
  underwriting workflow, not a real credit/lending standard or financial
  advice** — see the disclaimer below.

Every request runs through the exact frozen preprocessing chain (ratio
features → fold-safe real macro alignment → frozen winsorizer/isolation-
forest/target encoder) — no transformer is ever refit at serving time.
Payloads are validated twice: structurally by Pydantic (`422` on missing/
malformed fields), then semantically by `schema_utils.validate_payload`
against the bundle's frozen `FeatureSchema`; an unseen `Neighborhood` is
not rejected (the target encoder's global-mean fallback handles it) but is
surfaced to the client as a `Warnings` entry.

### 7. Database persistence (`database.py`)

SQLAlchemy ORM, SQLite by default (zero-setup, `artifacts/underwriting.db`)
and Postgres-compatible via `DATABASE_URL` — nothing else changes. Two
tables: `Listings` (every raw incoming payload + metadata) and
`UnderwritingEvaluations` (every model output — fair value, anomaly
z-score, decision, recommended bid range, stressed LTV — foreign-keyed to
its `Listing`, one row per `/predict` or `/underwrite` call). Both
endpoints log automatically via `database.log_listing()` /
`log_underwriting_evaluation()` on every request, giving a complete,
queryable audit trail.

### 8. Containerized deployment (`Dockerfile`)

Python 3.11-slim, installs `requirements.txt`, and — if
`artifacts/model_bundle/pipeline_bundle.joblib` isn't already present in
the build context — trains and exports it at build time, so the image is
self-contained and ready to serve on first `docker run`. Exposes port
8000, runs `uvicorn api_service:app --host 0.0.0.0 --port 8000`, with a
built-in `HEALTHCHECK` against `/health`.

## Deployment

```bash
docker build -t real-estate-valuation .

# SQLite (default), cached-real macro fallback:
docker run -p 8000:8000 real-estate-valuation

# Using .env (recommended -- see Configuration above):
cp .env.example .env   # fill in FRED_API_KEY / DATABASE_URL as desired
docker run -p 8000:8000 --env-file .env real-estate-valuation

# or set individual variables inline instead of a file:
docker run -p 8000:8000 \
  -e FRED_API_KEY="your_free_fred_api_key" \
  -e DATABASE_URL="postgresql://user:pass@host:5432/dbname" \
  real-estate-valuation
```

Then, from another shell:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"listings": [{
        "OverallQual": 7, "OverallCond": 5, "LivingAreaSqFt": 1800, "LotAreaSqFt": 9000,
        "GarageAreaSqFt": 480, "YearBuilt": 2003, "YearRenovated": 2003,
        "BedroomAbvGr": 3, "FullBath": 2, "TotRmsAbvGrd": 7,
        "NearRailroad": 0, "NearArtery": 0, "NearPositiveFeature": 0,
        "Neighborhood": "CollgCr", "ListingDate": "2009-06-15"
      }]}'

curl -X POST http://localhost:8000/underwrite \
  -H "Content-Type: application/json" \
  -d '{"listings": [{
        "OverallQual": 7, "OverallCond": 5, "LivingAreaSqFt": 1800, "LotAreaSqFt": 9000,
        "GarageAreaSqFt": 480, "YearBuilt": 2003, "YearRenovated": 2003,
        "BedroomAbvGr": 3, "FullBath": 2, "TotRmsAbvGrd": 7,
        "NearRailroad": 0, "NearArtery": 0, "NearPositiveFeature": 0,
        "Neighborhood": "CollgCr", "ListingDate": "2009-06-15", "ListPrice": 130000
      }]}'

# or, equivalently:
python test_client.py
```

**Disclaimer**: `/underwrite`'s recommended bid range and stressed LTV
limit are simple, fully documented heuristics built for this demo
workflow (see `config.py`'s `UNDERWRITING_*` constants) — they are not
professional financial, credit, or investment advice, and shouldn't be
treated as an underwriting standard. Anyone using this in a real lending
or investment context should have the methodology reviewed by a qualified
professional.


## Live data sources & graceful degradation

| Source | Used for | Live endpoint | Fallback if unreachable |
|---|---|---|---|
| FRED API (`fredapi`) | Mortgage rate, CPI, unemployment | `api.stlouisfed.org` (needs free key) | Cached table of the same real FRED figures |
| OpenStreetMap Nominatim (`geopy`) | Neighborhood centroid geocoding | `nominatim.openstreetmap.org` | Distance columns omitted; real Condition1/2 flags used instead |
| OpenStreetMap / Overpass (`osmnx`) | Highway/rail/park proximity | Overpass API | Distance columns omitted; real Condition1/2 flags used instead |

Every fallback path uses **real** data (either cached real historical
figures, or a different real feature already in the dataset) — nothing in
this codebase ever substitutes a simulated or randomly generated value for
an unreachable live one.

## Data — 100% real, no synthetic generation

**Property data**: the De Cock (2011) Ames, Iowa Assessor's Office dataset
— 2,930 real residential property sales from 2006-2010. Bundled locally as
`ames_raw.csv` (fetched from a public GitHub mirror of the original
dataset De Cock published at
https://jse.amstat.org/v19n3/decock/AmesHousing.txt).

**Macro data**: real, published national statistics — Freddie Mac PMMS
(`MORTGAGE30US`), BLS CPI-U (`CPIAUCNS`), and BLS unemployment (`UNRATE`)
— live via FRED, or the same real figures from a cached fallback table.

**Spatial data**: real neighborhood names, live-geocoded via OpenStreetMap
when network access allows; real per-property adjacency flags
(`Condition 1`/`Condition 2`) always available as a fallback/complement.

The only non-real inputs anywhere in the codebase are the four
**stress-test edge-case probes** in `config.OOD_TEST_CASES` (e.g. a 10,000
sqft home with 1 bedroom) — deliberately implausible hypothetical rows used
purely to test extrapolation behavior, exactly as the spec requested. They
are never used as training data.

## Anti-leakage design

Every stateful transformer (`KFoldTargetEncoder`, `Winsorizer`,
`IsolationForestFlagger`, `MacroSeriesResampler`, `GeoEnricher`) is invoked
from inside `models.preprocess_fold()` — fit only on the training slice of
each CV fold, applied via `transform()` to validation/holdout data. Unseen
categories at inference (e.g. a Neighborhood absent from a training fold)
fall back to the global training-fold mean rather than leaking.

## Cross-validation

`models.GroupTimeSeriesSplit` groups by real `Neighborhood`, orders groups
by first-seen listing date, and builds folds so no Neighborhood ever
appears in both train and validation, and validation is always later in
time than the training data that precedes it — controlling for both the
spatial and temporal autocorrelation real estate prices exhibit.

## Hyperparameter tuning

Optuna (`OptunaTuner`) tunes XGBoost/LightGBM against out-of-fold RMSE
under the same spatial-temporal splitter, within conservative bounds:
`max_depth` in [3, 6], `subsample`/`colsample_bytree` <= 0.8, explicit L1
(`reg_alpha`) and L2 (`reg_lambda`), plus early stopping on validation RMSE.

## Latest run results (full dataset, 25 Optuna trials, 5 folds)

Dev-set out-of-fold CV (90% of data, cached-real macro fallback in this
environment since no live network path to FRED/OSM exists here):

| Model | MAE ($) | RMSE ($) | MAPE (%) | R² |
|---|---|---|---|---|
| XGBoost | 25,142 | 39,075 | 13.59 | 0.677 |
| LightGBM | 25,595 | 39,366 | 14.15 | 0.663 |
| Ridge | 21,168 | 32,486 | 11.87 | **0.783** |

**Final unbiased holdout evaluation** (Ridge, 293 untouched real sales,
never seen during tuning or CV): MAE $19,541, RMSE $29,573, MAPE 13.0%,
R² 0.846.

Ridge wins consistently — plausible for Ames, where price is driven
heavily by a few strong, fairly linear drivers (living area, overall
quality, neighborhood). Re-run with live geo features enabled and/or more
Optuna trials to see whether the boosted trees close the gap.

## Extending to production

- Set `FRED_API_KEY` and ensure outbound network access to FRED/Nominatim/
  Overpass to enable all live data sources.
- Swap `ames_raw.csv` / `load_raw_data()` for a live MLS/county-assessor
  extract with the same column names; feed `screening_engine.DealScreener`
  a real live listings feed instead of the holdout-sample demo.
- Raise `N_OPTUNA_TRIALS` for a more thorough hyperparameter search.
- Consider persisting the fitted model + transformers (e.g. via `joblib`)
  so `DealScreener` can run in a separate process/service from training.
