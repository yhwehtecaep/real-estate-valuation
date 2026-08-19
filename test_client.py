"""
test_client.py
===============
Small standalone script that exercises all three API endpoints against a
running instance of api_service.py. Uses only the standard library (no
`requests` dependency) so it's runnable anywhere Python 3 is available.

Usage:
    python export_pipeline.py                          # once, to produce the model bundle
    uvicorn api_service:app --host 0.0.0.0 --port 8000 &
    python test_client.py [base_url]                    # default: http://localhost:8000

Every listing below uses real, plausible Ames property values -- no
synthetic training data is involved; this just exercises the live API
contract.
"""

import json
import sys
import urllib.error
import urllib.request

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

SAMPLE_LISTING = {
    "OverallQual": 7, "OverallCond": 5, "LivingAreaSqFt": 1800, "LotAreaSqFt": 9000,
    "GarageAreaSqFt": 480, "YearBuilt": 2003, "YearRenovated": 2003,
    "BedroomAbvGr": 3, "FullBath": 2, "TotRmsAbvGrd": 7,
    "NearRailroad": 0, "NearArtery": 0, "NearPositiveFeature": 0,
    "Neighborhood": "CollgCr", "ListingDate": "2009-06-15",
}

SAMPLE_LISTING_2 = {
    "OverallQual": 5, "OverallCond": 6, "LivingAreaSqFt": 1200, "LotAreaSqFt": 6000,
    "GarageAreaSqFt": 0, "YearBuilt": 1955, "YearRenovated": 1955,
    "BedroomAbvGr": 2, "FullBath": 1, "TotRmsAbvGrd": 5,
    "NearRailroad": 1, "NearArtery": 0, "NearPositiveFeature": 0,
    "Neighborhood": "OldTown", "ListingDate": "2008-03-01",
}


def _call(method: str, path: str, payload: dict = None) -> dict:
    url = f"{BASE_URL}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"  HTTP {e.code}: {body}")
        return {"error": e.code, "body": body}


def main():
    print(f"Testing API at {BASE_URL}\n")

    print("=== GET /health ===")
    health = _call("GET", "/health")
    print(json.dumps(health, indent=2))
    if not health.get("model_loaded"):
        print("\nModel not loaded -- run `python export_pipeline.py` first, then restart the API.")
        return

    print("\n=== POST /predict (single listing) ===")
    pred = _call("POST", "/predict", {"listings": [SAMPLE_LISTING]})
    print(json.dumps(pred, indent=2))

    print("\n=== POST /predict (batch of 2) ===")
    pred_batch = _call("POST", "/predict", {"listings": [SAMPLE_LISTING, SAMPLE_LISTING_2]})
    print(json.dumps(pred_batch, indent=2))

    print("\n=== POST /underwrite (with ListPrice, expect a real decision) ===")
    underwrite_listing = dict(SAMPLE_LISTING, ListPrice=130000)
    result = _call("POST", "/underwrite", {"listings": [underwrite_listing]})
    print(json.dumps(result, indent=2))

    print("\n=== POST /underwrite (missing ListPrice -> expect 422) ===")
    _call("POST", "/underwrite", {"listings": [SAMPLE_LISTING]})

    print("\nDone.")


if __name__ == "__main__":
    main()
