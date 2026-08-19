"""
data_pipeline.py
=================
Ingestion and leakage-safe feature engineering for the Real Estate
Valuation & Pricing Model.

Data sources (all real, no synthetic rows or columns anywhere in this file)
-----------------------------------------------------------------------------
1. Property-level data: the De Cock (2011) Ames, Iowa Assessor's Office
   dataset -- 2,930 real residential property sales, 2006-2010. Loaded from
   a bundled local CSV (`config.AMES_CSV_PATH`), originally fetched from a
   public GitHub mirror of the dataset
   (https://github.com/STATCowboy/pbidataflowstalk, itself a copy of the
   dataset De Cock published at
   https://jse.amstat.org/v19n3/decock/AmesHousing.txt).
2. Macro time series: LIVE via the free FRED API (`FredMacroClient`, using
   `fredapi`) -- 30-year mortgage rate (MORTGAGE30US), CPI-U (CPIAUCNS), and
   unemployment rate (UNRATE). Requires a free FRED API key
   (config.FRED_API_KEY_ENV_VAR). If the key is missing or the API is
   unreachable, the pipeline falls back to a cached table of the SAME real,
   published FRED figures (never simulated values) so it keeps running --
   see `_CACHED_FALLBACK_MACRO_*` below.
3. Spatial data: LIVE geocoding of the real Ames `Neighborhood` field via
   OpenStreetMap Nominatim (`GeoEnricher`, using `geopy`), producing real
   geodesic distances to two fixed, verified Ames landmarks (downtown CBD,
   Iowa State University campus -- see config.AMES_CBD_LATLON /
   config.ISU_CAMPUS_LATLON). Real infrastructure-proximity buffers
   (highway/rail/park) are pulled live from OpenStreetMap via `osmnx`
   (`SpatialInfrastructureEngineer`). If live geocoding/OSM access is
   unavailable, these enrichments are skipped (columns omitted, never
   fabricated) and the pipeline falls back to the real per-property
   `Condition 1`/`Condition 2` proximity flags already in the Assessor
   data, which is real ground truth in its own right.

Design principles
------------------
1. **No look-ahead / no target leakage.** Every stateful transformer here
   (target encoder, winsorizer, isolation-forest outlier flagger) exposes a
   standard `fit` / `transform` interface and is only ever `fit()` on the
   training fold inside cross-validation (see `models.py`). Nothing in this
   module computes statistics over the full dataset and reuses them at
   inference time.
2. **No synthetic data.** Every feature is either a real column from the
   Ames dataset, a real published/live macro statistic, a real live-geocoded
   coordinate, or a deterministic arithmetic combination of real columns
   (e.g. age at sale = sale year - year built). Nothing here is randomly
   generated. The only exception anywhere in the codebase is the stress-test
   edge-case probes in `config.OOD_TEST_CASES`, which are deliberately
   synthetic hypotheticals used purely to test extrapolation behavior and
   are never used as training data.
3. **Fold-safe spatial/macro alignment.** `MacroSeriesResampler` and
   `GeoEnricher` are invoked from inside `models.preprocess_fold()` (per
   fold), matching the leakage-safe architecture of the target encoder and
   winsorizer. Note that macro/geo data is exogenous public information
   that does not depend on y or on fold membership, so there is no actual
   leakage vector here (unlike the target encoder) -- fold-scoping is done
   for architectural consistency and because results are cached, so it
   costs nothing extra to do it this way.
"""

import logging
import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest

import config

logger = logging.getLogger("real_estate_valuation.data_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


# --------------------------------------------------------------------------- #
# Cached fallback macro data (real, published FRED figures -- Freddie Mac
# PMMS / BLS via FRED) used ONLY when the live FRED API is unreachable
# (missing API key or no network path). This is real historical data, not a
# synthetic estimate -- see `FredMacroClient` below for the live path.
# --------------------------------------------------------------------------- #
# 30-year fixed mortgage rate: monthly average of Freddie Mac's weekly PMMS
# series (MORTGAGE30US), rounded to 2 decimals.
_CACHED_FALLBACK_MORTGAGE30US_MONTHLY = {
    (2006, 1): 6.15, (2006, 2): 6.25, (2006, 3): 6.32, (2006, 4): 6.51, (2006, 5): 6.60,
    (2006, 6): 6.68, (2006, 7): 6.76, (2006, 8): 6.52, (2006, 9): 6.40, (2006, 10): 6.36,
    (2006, 11): 6.24, (2006, 12): 6.14,
    (2007, 1): 6.22, (2007, 2): 6.29, (2007, 3): 6.16, (2007, 4): 6.18, (2007, 5): 6.21,
    (2007, 6): 6.66, (2007, 7): 6.70, (2007, 8): 6.57, (2007, 9): 6.38, (2007, 10): 6.38,
    (2007, 11): 6.21, (2007, 12): 6.10,
    (2008, 1): 5.68, (2008, 2): 5.92, (2008, 3): 5.97, (2008, 4): 5.92, (2008, 5): 6.04,
    (2008, 6): 6.32, (2008, 7): 6.43, (2008, 8): 6.48, (2008, 9): 6.04, (2008, 10): 6.20,
    (2008, 11): 6.09, (2008, 12): 5.33,
    (2009, 1): 5.05, (2009, 2): 5.13, (2009, 3): 5.00, (2009, 4): 4.81, (2009, 5): 4.86,
    (2009, 6): 5.42, (2009, 7): 5.22, (2009, 8): 5.19, (2009, 9): 5.06, (2009, 10): 4.95,
    (2009, 11): 4.88, (2009, 12): 4.93,
    (2010, 1): 5.03, (2010, 2): 4.99, (2010, 3): 4.97, (2010, 4): 5.10, (2010, 5): 4.89,
    (2010, 6): 4.74, (2010, 7): 4.56, (2010, 8): 4.43, (2010, 9): 4.35, (2010, 10): 4.23,
    (2010, 11): 4.30, (2010, 12): 4.69,
}

# CPI-U, all items, U.S. city average, not seasonally adjusted (CPIAUCNS)
_CACHED_FALLBACK_CPIAUCNS_MONTHLY = {
    (2006, 1): 198.300, (2006, 2): 198.700, (2006, 3): 199.800, (2006, 4): 201.500,
    (2006, 5): 202.500, (2006, 6): 202.900, (2006, 7): 203.500, (2006, 8): 203.900,
    (2006, 9): 202.900, (2006, 10): 201.800, (2006, 11): 201.500, (2006, 12): 201.800,
    (2007, 1): 202.416, (2007, 2): 203.499, (2007, 3): 205.352, (2007, 4): 206.686,
    (2007, 5): 207.949, (2007, 6): 208.352, (2007, 7): 208.299, (2007, 8): 207.917,
    (2007, 9): 208.490, (2007, 10): 208.936, (2007, 11): 210.177, (2007, 12): 210.036,
    (2008, 1): 211.080, (2008, 2): 211.693, (2008, 3): 213.528, (2008, 4): 214.823,
    (2008, 5): 216.632, (2008, 6): 218.815, (2008, 7): 219.964, (2008, 8): 219.086,
    (2008, 9): 218.783, (2008, 10): 216.573, (2008, 11): 212.425, (2008, 12): 210.228,
    (2009, 1): 211.143, (2009, 2): 212.193, (2009, 3): 212.709, (2009, 4): 213.240,
    (2009, 5): 213.856, (2009, 6): 215.693, (2009, 7): 215.351, (2009, 8): 215.834,
    (2009, 9): 215.969, (2009, 10): 216.177, (2009, 11): 216.330, (2009, 12): 215.949,
    (2010, 1): 216.687, (2010, 2): 216.741, (2010, 3): 217.631, (2010, 4): 218.009,
    (2010, 5): 218.178, (2010, 6): 217.965, (2010, 7): 218.011, (2010, 8): 218.312,
    (2010, 9): 218.439, (2010, 10): 218.711, (2010, 11): 218.803, (2010, 12): 219.179,
}

# Civilian unemployment rate, seasonally adjusted (UNRATE)
_CACHED_FALLBACK_UNRATE_MONTHLY = {
    (2006, 1): 4.7, (2006, 2): 4.8, (2006, 3): 4.7, (2006, 4): 4.7, (2006, 5): 4.6,
    (2006, 6): 4.6, (2006, 7): 4.7, (2006, 8): 4.7, (2006, 9): 4.5, (2006, 10): 4.4,
    (2006, 11): 4.5, (2006, 12): 4.4,
    (2007, 1): 4.6, (2007, 2): 4.5, (2007, 3): 4.4, (2007, 4): 4.5, (2007, 5): 4.4,
    (2007, 6): 4.6, (2007, 7): 4.7, (2007, 8): 4.6, (2007, 9): 4.7, (2007, 10): 4.7,
    (2007, 11): 4.7, (2007, 12): 5.0,
    (2008, 1): 5.0, (2008, 2): 4.9, (2008, 3): 5.1, (2008, 4): 5.0, (2008, 5): 5.4,
    (2008, 6): 5.6, (2008, 7): 5.8, (2008, 8): 6.1, (2008, 9): 6.1, (2008, 10): 6.5,
    (2008, 11): 6.8, (2008, 12): 7.3,
    (2009, 1): 7.8, (2009, 2): 8.3, (2009, 3): 8.7, (2009, 4): 9.0, (2009, 5): 9.4,
    (2009, 6): 9.5, (2009, 7): 9.5, (2009, 8): 9.6, (2009, 9): 9.8, (2009, 10): 10.0,
    (2009, 11): 9.9, (2009, 12): 9.9,
    (2010, 1): 9.8, (2010, 2): 9.8, (2010, 3): 9.9, (2010, 4): 9.9, (2010, 5): 9.6,
    (2010, 6): 9.4, (2010, 7): 9.4, (2010, 8): 9.5, (2010, 9): 9.5, (2010, 10): 9.4,
    (2010, 11): 9.8, (2010, 12): 9.3,
}


def _cached_fallback_macro_frame() -> pd.DataFrame:
    """Real (year, month)-indexed macro frame used only as a live-API fallback."""
    rows = []
    for (year, month), rate in _CACHED_FALLBACK_MORTGAGE30US_MONTHLY.items():
        rows.append({
            "Date": pd.Timestamp(year=year, month=month, day=1),
            "MortgageRate30Y": rate,
            "CPI": _CACHED_FALLBACK_CPIAUCNS_MONTHLY[(year, month)],
            "LocalUnemploymentRate": _CACHED_FALLBACK_UNRATE_MONTHLY[(year, month)],
        })
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two real lat/lon points."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return float(2 * 6371.0 * np.arcsin(np.sqrt(a)))


class MacroDataUnavailableError(RuntimeError):
    """Raised when the live FRED API cannot be reached after all retries."""


class FredMacroClient:
    """
    Thin, robust wrapper around `fredapi.Fred` for pulling real, live macro
    series (30-year mortgage rate, CPI, unemployment rate) at their native
    publication frequency.

    Requires a free FRED API key (register at
    https://fred.stlouisfed.org/docs/api/api_key.html), read from the
    `config.FRED_API_KEY_ENV_VAR` environment variable unless passed
    explicitly. Retries transient network/HTTP errors up to
    `config.FRED_MAX_RETRIES` times with exponential backoff before raising
    `MacroDataUnavailableError` -- callers (see `get_macro_series`) should
    catch this and fall back to the cached real historical table rather
    than crash the pipeline.
    """

    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.environ.get(config.FRED_API_KEY_ENV_VAR)
        self._client = None
        if self.api_key:
            try:
                from fredapi import Fred
                self._client = Fred(api_key=self.api_key)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("Could not initialize fredapi client: %s", e)
                self._client = None

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    def fetch_series(self, series_id: str, start: str, end: str) -> pd.Series:
        """Fetches one real FRED series with retries. Raises MacroDataUnavailableError on failure."""
        if not self.is_configured:
            raise MacroDataUnavailableError(
                f"FRED API key not configured (set the {config.FRED_API_KEY_ENV_VAR} "
                "environment variable to enable live macro data)."
            )
        last_err = None
        for attempt in range(1, config.FRED_MAX_RETRIES + 1):
            try:
                series = self._client.get_series(
                    series_id, observation_start=start, observation_end=end
                )
                if series is None or len(series) == 0:
                    raise ValueError(f"FRED returned no observations for {series_id}")
                return series
            except Exception as e:
                last_err = e
                logger.warning(
                    "FRED fetch of %s failed (attempt %d/%d): %s",
                    series_id, attempt, config.FRED_MAX_RETRIES, e,
                )
                if attempt < config.FRED_MAX_RETRIES:
                    time.sleep(config.FRED_RETRY_BACKOFF_S * attempt)
        raise MacroDataUnavailableError(f"Could not fetch {series_id} from FRED after retries: {last_err}")

    def fetch_all(self, start: str, end: str) -> pd.DataFrame:
        """
        Fetches all three real macro series (config.FRED_SERIES) at their
        native frequency and returns a long, date-indexed frame with columns
        [Date, MortgageRate30Y, CPI, LocalUnemploymentRate] (outer-joined,
        NaN where a series has no observation on a given date -- frequency
        alignment to property sale dates happens downstream in
        `MacroSeriesResampler`, not here).
        """
        frames = []
        for our_name, fred_id in config.FRED_SERIES.items():
            s = self.fetch_series(fred_id, start, end).rename(our_name)
            frames.append(s)
        wide = pd.concat(frames, axis=1)
        wide.index.name = "Date"
        return wide.reset_index()


def get_macro_series(start: str, end: str, api_key: Optional[str] = None) -> Tuple[pd.DataFrame, bool]:
    """
    Real macro series orchestration: tries the LIVE FRED API first; on any
    failure (missing key, network error, rate limit), logs a clear warning
    and falls back to the cached table of the same real, published FRED
    figures. Never fabricates a value.

    Returns (macro_frame, was_live) where macro_frame has columns
    [Date, MortgageRate30Y, CPI, LocalUnemploymentRate] at native/monthly
    frequency and was_live indicates which source was used.
    """
    client = FredMacroClient(api_key=api_key)
    try:
        frame = client.fetch_all(start, end)
        logger.info("Fetched LIVE macro data from FRED API for %s..%s.", start, end)
        return frame, True
    except MacroDataUnavailableError as e:
        logger.warning(
            "Live FRED fetch unavailable (%s). Falling back to cached real "
            "historical FRED figures (Freddie Mac PMMS / BLS, 2006-2010) -- "
            "not synthetic data, just not live for this run.", e,
        )
        return _cached_fallback_macro_frame(), False


class MacroSeriesResampler(BaseEstimator, TransformerMixin):
    """
    Aligns real macro time series (fetched once at their native
    weekly/monthly frequency via `get_macro_series`) to each property's
    exact real sale date, using an as-of (backward) merge -- i.e. forward-
    filling the most recently published macro observation as of the sale
    date -- with linear interpolation for any remaining internal gaps.

    Implemented as a fit/transform pipeline stage so it can be invoked from
    inside `models.preprocess_fold()`, matching the leakage-safe pattern
    used by the target encoder and winsorizer. Because the macro series
    itself is exogenous (public economic data independent of y and of
    which properties land in which fold), fitting it per-fold carries no
    actual leakage risk -- this is architectural consistency, not a
    correction for a real leakage vector.
    """

    def __init__(self, macro_frame: pd.DataFrame):
        self.macro_frame = macro_frame.sort_values("Date").reset_index(drop=True)

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        original_index = X.index
        X_sorted = X.sort_values(config.DATE_COL)
        merged = pd.merge_asof(
            X_sorted, self.macro_frame,
            left_on=config.DATE_COL, right_on="Date", direction="backward",
        )
        merged.index = X_sorted.index
        merged = merged.reindex(original_index)
        macro_cols = ["MortgageRate30Y", "CPI", "LocalUnemploymentRate"]
        for col in macro_cols:
            # backward-fill covers sale dates earlier than the first live/
            # cached macro observation; interpolate covers any internal gaps
            merged[col] = merged[col].interpolate(limit_direction="both")
        merged = merged.drop(columns=["Date"])
        return merged


class GeoEnricher:
    """
    Live geocoding of the real Ames `Neighborhood` field via OpenStreetMap
    Nominatim (through `geopy`), producing real geodesic distances from
    each neighborhood's geocoded centroid to two fixed, verified Ames
    landmarks: downtown (config.AMES_CBD_LATLON) and the Iowa State
    University campus (config.ISU_CAMPUS_LATLON).

    Results are cached in-process by neighborhood name -- geocoding a fixed
    place name is invariant across CV folds, so there is no leakage risk in
    reusing the cache (unlike the target encoder, whose statistics must
    differ per fold).

    If Nominatim is unreachable, this degrades gracefully: it logs one
    clear warning and returns NaN distance columns rather than fabricating
    coordinates. `models.preprocess_fold()` drops these columns from the
    feature set for that run if they come back entirely NaN, so the
    pipeline stays 100% executable without live network access -- it just
    runs with the real `Condition 1`/`Condition 2` proximity flags as the
    spatial-infrastructure signal instead.
    """

    _cache: Dict[str, Optional[Tuple[float, float]]] = {}
    _network_confirmed_down: bool = False   # class-level fast-fail latch, shared across instances

    def __init__(self, user_agent: str = config.NOMINATIM_USER_AGENT,
                 timeout: int = config.NOMINATIM_TIMEOUT_S,
                 min_delay: float = config.NOMINATIM_MIN_DELAY_S):
        try:
            from geopy.geocoders import Nominatim
            from geopy.extra.rate_limiter import RateLimiter
            self._geolocator = Nominatim(user_agent=user_agent, timeout=timeout)
            # max_retries=0, swallow_exceptions=False: let the FIRST failure raise
            # immediately so our own fast-fail latch (_network_confirmed_down) can
            # trip after exactly one round-trip, instead of geopy silently retrying
            # and swallowing the error internally on every neighborhood.
            self._geocode = RateLimiter(
                self._geolocator.geocode, min_delay_seconds=min_delay,
                max_retries=0, swallow_exceptions=False,
            )
            self._available = True
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Could not initialize geopy/Nominatim client: %s", e)
            self._available = False

    def geocode_neighborhood(self, neighborhood_name: str) -> Optional[Tuple[float, float]]:
        if neighborhood_name in self._cache:
            return self._cache[neighborhood_name]
        if not self._available or GeoEnricher._network_confirmed_down:
            self._cache[neighborhood_name] = None
            return None
        query = f"{neighborhood_name}, Ames, Iowa, USA"
        try:
            location = self._geocode(query)
            if location is None:
                logger.warning("Live geocoding: Nominatim returned no result for %r.", query)
                coords = None
            else:
                coords = (location.latitude, location.longitude)
        except Exception as e:
            logger.warning(
                "Live geocoding unreachable (%s). No coordinates will be fabricated; "
                "treating the network as down for the rest of this run (no further "
                "Nominatim calls will be attempted) -- distance-to-CBD/ISU will be NaN "
                "and the real Condition1/2 proximity flags will carry the spatial "
                "signal instead.", e,
            )
            coords = None
            GeoEnricher._network_confirmed_down = True
        self._cache[neighborhood_name] = coords
        return coords

    def enrich_with_distances(self, X: pd.DataFrame) -> pd.DataFrame:
        """Adds real geodesic DistToCommercialCenterKm / DistToISUKm columns (NaN if ungeocodable)."""
        X = X.copy()
        dist_cbd, dist_isu = {}, {}
        for nbhd in X[config.GROUP_COL].unique():
            coords = self.geocode_neighborhood(nbhd)
            if coords is None:
                dist_cbd[nbhd], dist_isu[nbhd] = np.nan, np.nan
            else:
                dist_cbd[nbhd] = _haversine_km(*coords, *config.AMES_CBD_LATLON)
                dist_isu[nbhd] = _haversine_km(*coords, *config.ISU_CAMPUS_LATLON)
        X["DistToCommercialCenterKm"] = X[config.GROUP_COL].map(dist_cbd)
        X["DistToISUKm"] = X[config.GROUP_COL].map(dist_isu)
        return X


class SpatialInfrastructureEngineer:
    """
    Live proximity buffers to real OpenStreetMap infrastructure (highways,
    parks, rail) around each geocoded neighborhood centroid, via `osmnx`.

    Queries OSM's Overpass API (through osmnx) for real infrastructure
    geometries within `config.OSM_SEARCH_RADIUS_M` of each centroid and
    computes the real distance to the nearest feature of each type. This
    requires live network access to OSM; if unavailable, it logs one clear
    warning and returns NaN columns (never fabricated), matching the same
    graceful-degradation pattern as `GeoEnricher`.
    """

    _cache: Dict[str, Dict[str, float]] = {}
    _network_confirmed_down: bool = False   # class-level fast-fail latch, shared across instances

    def __init__(self, search_radius_m: float = config.OSM_SEARCH_RADIUS_M):
        self.search_radius_m = search_radius_m
        try:
            import osmnx  # noqa: F401
            self._available = True
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("osmnx not usable in this environment: %s", e)
            self._available = False

    def _nearest_feature_distance_m(self, lat: float, lon: float, tags: Dict) -> float:
        import osmnx as ox
        from shapely.geometry import Point
        import geopandas as gpd

        gdf = ox.features_from_point((lat, lon), tags=tags, dist=self.search_radius_m)
        if gdf.empty:
            return np.nan
        point_gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
        gdf_m = gdf.to_crs(gdf.estimate_utm_crs())
        point_m = point_gdf.to_crs(gdf_m.crs)
        return float(gdf_m.distance(point_m.geometry.iloc[0]).min())

    def enrich_with_infrastructure_distances(self, X: pd.DataFrame, geo_enricher: "GeoEnricher") -> pd.DataFrame:
        """Adds real OSMNearHighwayM / OSMNearRailM / OSMNearParkM columns (NaN if unavailable)."""
        X = X.copy()
        cols = {"OSMNearHighwayM": {"highway": ["motorway", "trunk", "primary"]},
                "OSMNearRailM": {"railway": "rail"},
                "OSMNearParkM": {"leisure": "park"}}
        for out_col in cols:
            X[out_col] = np.nan
        if not self._available:
            return X
        for nbhd in X[config.GROUP_COL].unique():
            if nbhd in self._cache:
                distances = self._cache[nbhd]
            elif SpatialInfrastructureEngineer._network_confirmed_down:
                distances = {c: np.nan for c in cols}
            else:
                coords = geo_enricher.geocode_neighborhood(nbhd)
                if coords is None:
                    distances = {c: np.nan for c in cols}
                else:
                    distances = {}
                    for out_col, tags in cols.items():
                        try:
                            distances[out_col] = self._nearest_feature_distance_m(*coords, tags)
                        except Exception as e:
                            logger.warning(
                                "Live OSM infrastructure query unreachable for %s (%s). No "
                                "distance will be fabricated; treating OSM as down for the "
                                "rest of this run.", out_col, e,
                            )
                            distances[out_col] = np.nan
                            SpatialInfrastructureEngineer._network_confirmed_down = True
                self._cache[nbhd] = distances
            for out_col, val in distances.items():
                X.loc[X[config.GROUP_COL] == nbhd, out_col] = val
        return X


# --------------------------------------------------------------------------- #
# 1. Ingestion
# --------------------------------------------------------------------------- #
def load_raw_data(csv_path: str = config.AMES_CSV_PATH) -> pd.DataFrame:
    """
    Loads the real De Cock (2011) Ames, Iowa housing dataset. Does NOT join
    macro data here -- that now happens fold-safely via
    `MacroSeriesResampler` inside `models.preprocess_fold()`, using the real
    (live-or-cached-fallback) series from `get_macro_series()`.

    Every column that reaches the model is either:
      (a) a real column from the Ames Assessor's Office data, renamed for
          readability (e.g. "Gr Liv Area" -> "LivingAreaSqFt"),
      (b) a real macro/geospatial statistic joined downstream, or
      (c) a deterministic arithmetic function of (a) -- e.g. age at sale --
          computed later in `RatioFeatureBuilder`.
    No column is randomly generated.
    """
    logger.info("Loading real Ames, Iowa housing dataset from %s ...", csv_path)
    raw = pd.read_csv(csv_path)
    logger.info("Loaded %d real property sales, %d raw columns.", *raw.shape)

    df = pd.DataFrame({
        config.ID_COL: raw["Order"],
        "LivingAreaSqFt": raw["Gr Liv Area"].astype(float),
        "LotAreaSqFt": raw["Lot Area"].astype(float),
        "GarageAreaSqFt": raw["Garage Area"].astype(float),
        "YearBuilt": raw["Year Built"].astype(float),
        "YearRenovated": raw["Year Remod/Add"].astype(float),
        "OverallQual": raw["Overall Qual"].astype(float),
        "OverallCond": raw["Overall Cond"].astype(float),
        "BedroomAbvGr": raw["Bedroom AbvGr"].astype(float),
        "FullBath": raw["Full Bath"].astype(float),
        "TotRmsAbvGrd": raw["TotRms AbvGrd"].astype(float),
        config.GROUP_COL: raw["Neighborhood"].astype(str),
        "Mo Sold": raw["Mo Sold"].astype(int),
        "Yr Sold": raw["Yr Sold"].astype(int),
        config.TARGET_RAW: raw["SalePrice"].astype(float),
    })

    # Real proximity-to-infrastructure flags, derived from the real
    # Condition 1 / Condition 2 fields (De Cock's own encoding of adjacency
    # to a railroad, arterial/feeder street, or an off-site positive
    # feature such as a park or greenbelt). This is the real, always-
    # available substitute for the live OSM buffers in
    # `SpatialInfrastructureEngineer` above, used whenever live OSM access
    # isn't available.
    cond = raw[["Condition 1", "Condition 2"]]
    df["NearRailroad"] = cond.isin(["RRAn", "RRAe", "RRNn", "RRNe"]).any(axis=1).astype(int)
    df["NearArtery"] = cond.isin(["Artery", "Feedr"]).any(axis=1).astype(int)
    df["NearPositiveFeature"] = cond.isin(["PosN", "PosA"]).any(axis=1).astype(int)

    # A handful of rows (1 in this dataset) have a missing Garage Area,
    # meaning "no garage was recorded" in the assessor data -- impute with
    # 0, the same convention De Cock used for basement/garage absence
    # elsewhere in the dataset. This is a real, documented data-cleaning
    # convention, not a synthetic value.
    df["GarageAreaSqFt"] = df["GarageAreaSqFt"].fillna(0.0)

    # Real listing date, built from the real Yr Sold / Mo Sold fields
    df[config.DATE_COL] = pd.to_datetime(
        df["Yr Sold"].astype(str) + "-" + df["Mo Sold"].astype(str) + "-01"
    )
    df = df.drop(columns=["Mo Sold", "Yr Sold"])

    logger.info("Final dataset: %d rows, %d columns. Macro/geo enrichment happens fold-safely downstream.",
                *df.shape)
    return df


# --------------------------------------------------------------------------- #
# 2. Leakage-safe transformers
# --------------------------------------------------------------------------- #
class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target-encodes high-cardinality categorical columns using ONLY statistics
    computed on the fold passed to `fit()`.

    Critically:
    - `fit(X_train, y_train)` computes per-category mean(y) + a global prior,
      smoothed by category count, using ONLY X_train/y_train.
    - `transform(X_any)` maps categories to the fitted encoding and falls
      back to the global prior for categories unseen during `fit` (new
      neighborhoods at inference time never see training-fold target
      leakage).
    - Called from inside each CV fold in `models.py`, never on the full
      dataset before splitting — this is what prevents target leakage.
    """

    def __init__(self, columns, smoothing: float = 10.0):
        self.columns = columns
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y: pd.Series):
        y = pd.Series(np.asarray(y), index=X.index)
        self.global_mean_ = y.mean()
        self.encodings_ = {}
        for col in self.columns:
            stats = y.groupby(X[col]).agg(["mean", "count"])
            smoothed = (stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing) / (
                stats["count"] + self.smoothing
            )
            self.encodings_[col] = smoothed.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            mapping = self.encodings_[col]
            X[col + "_te"] = X[col].map(mapping).fillna(self.global_mean_)
            X = X.drop(columns=[col])
        return X


class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Clips numeric columns to the [lower_pct, upper_pct] quantiles computed
    strictly from the training distribution passed to `fit()`.
    """

    def __init__(self, columns, lower_pct=config.WINSOR_LOWER_PCT, upper_pct=config.WINSOR_UPPER_PCT):
        self.columns = columns
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def fit(self, X: pd.DataFrame, y=None):
        self.bounds_ = {
            col: (X[col].quantile(self.lower_pct), X[col].quantile(self.upper_pct))
            for col in self.columns
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, (lo, hi) in self.bounds_.items():
            X[col] = X[col].clip(lo, hi)
        return X


class IsolationForestFlagger(BaseEstimator, TransformerMixin):
    """
    Fits an Isolation Forest on the training fold's numeric features and adds
    an `is_outlier_score` column (higher = more anomalous) at transform time.
    Does NOT drop rows itself — flagging is left to the caller so that
    train-time row removal and inference-time behavior stay clearly separate.
    """

    def __init__(self, columns, contamination=config.ISOLATION_FOREST_CONTAMINATION,
                 random_state=config.RANDOM_STATE):
        self.columns = columns
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None):
        self.model_ = IsolationForest(
            contamination=self.contamination, random_state=self.random_state, n_estimators=200
        )
        self.model_.fit(X[self.columns])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # score_samples: higher = more normal. Flip sign so higher = more anomalous.
        X["is_outlier_score"] = -self.model_.score_samples(X[self.columns])
        return X

    def train_mask(self, X: pd.DataFrame) -> np.ndarray:
        """Boolean mask of INLIERS for the fold X was fit on (predict==1)."""
        return self.model_.predict(X[self.columns]) == 1


class RatioFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Stateless domain-ratio engineering (safe to apply before or after
    splitting since it never references the target or cross-row statistics):
      - LivingToLotRatio = LivingAreaSqFt / LotAreaSqFt
      - AgeAtSale = ListingDate.year - YearBuilt
      - YearsSinceRenovation = ListingDate.year - YearRenovated
    All three are deterministic arithmetic on real columns.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        sale_year = pd.to_datetime(X[config.DATE_COL]).dt.year
        X["LivingToLotRatio"] = X["LivingAreaSqFt"] / X["LotAreaSqFt"].replace(0, np.nan)
        X["LivingToLotRatio"] = X["LivingToLotRatio"].fillna(X["LivingToLotRatio"].median())
        X["AgeAtSale"] = (sale_year - X["YearBuilt"]).clip(lower=0)
        X["YearsSinceRenovation"] = (sale_year - X["YearRenovated"]).clip(lower=0)
        return X


def apply_log_target(df: pd.DataFrame) -> pd.DataFrame:
    """log1p transform of SalePrice to stabilize variance / handle right skew."""
    df = df.copy()
    df[config.TARGET_LOG] = np.log1p(df[config.TARGET_RAW])
    return df


def get_raw_feature_columns() -> list:
    """
    Raw columns that must exist in the source dataframe immediately after
    `load_raw_data()` -- i.e. real per-property columns only. Macro
    (config.MACRO_FEATURES) and optional live-geo (config.OPTIONAL_GEO_FEATURES)
    columns are joined downstream, fold-safely, inside `models.preprocess_fold()`.
    Does NOT include ENGINEERED_FEATURES -- those are created downstream by
    `RatioFeatureBuilder.transform()`.
    """
    return config.PROPERTY_NUMERIC_FEATURES + config.CATEGORICAL_HIGH_CARD


def get_model_feature_columns() -> list:
    """Full feature column list the models train on (post ratio-engineering, pre target-encoding)."""
    return (
        config.NUMERIC_FEATURES
        + config.ENGINEERED_FEATURES
        + config.CATEGORICAL_HIGH_CARD
    )
