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
# Set at `docker run` time to enable live FRED data, e.g.:
#   docker run -e FRED_API_KEY=your_key -p 8000:8000 real-estate-valuation
ENV FRED_API_KEY=""
# Defaults to a SQLite file inside the container; override to point at
# Postgres or another SQLAlchemy-supported backend, e.g.:
#   docker run -e DATABASE_URL=postgresql://user:pass@host:5432/db ...
ENV DATABASE_URL=""

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "api_service:app", "--host", "0.0.0.0", "--port", "8000"]
