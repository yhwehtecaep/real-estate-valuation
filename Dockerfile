# syntax=docker/dockerfile:1
FROM python:3.11-slim

# osmnx/geopandas pull in a few native geo libraries; keep the image lean
# but include what's needed for shapely/pyproj wheels to build/run cleanly.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gdal-bin \
        libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Real data + real model bundle. If artifacts/model_bundle/pipeline_bundle.joblib
# isn't already present in the build context, train and export it at build
# time so the image is self-contained and ready to serve immediately.
RUN test -f artifacts/model_bundle/pipeline_bundle.joblib || python export_pipeline.py

ENV PYTHONUNBUFFERED=1
# .env is gitignored/dockerignored on purpose -- never baked into the image.
# Pass real secrets at `docker run` time instead:
#   docker run -p 8000:8000 --env-file .env real-estate-valuation
# or individually:
#   docker run -p 8000:8000 -e FRED_API_KEY=your_key real-estate-valuation
# Both default to empty/unset here, matching config.py's graceful fallback
# (cached real macro data, local SQLite) when unset.
ENV FRED_API_KEY=""
ENV DATABASE_URL=""

EXPOSE 8000

# Shell form (not exec form) so ${PORT} is actually expanded. Render (and
# several other PaaS platforms) inject a PORT env var at runtime and expect
# the container to listen on it; falls back to 8000 for plain `docker run`.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python healthcheck.py

CMD ["sh", "-c", "uvicorn api_service:app --host 0.0.0.0 --port ${PORT:-8000}"]
