"""
schema_utils.py
================
Shared feature-schema definition and payload validation, used by both
`export_pipeline.py` (to freeze the model's expected input contract
alongside the serialized artifacts) and `api_service.py` (to validate
incoming JSON payloads against that exact contract before they ever reach
the model pipeline).

Kept dependency-light (stdlib + config only) so it can be imported by
either module without pulling in FastAPI or the training stack.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import config

# Real per-property fields the model expects, with their expected Python
# types. NearRailroad/NearArtery/NearPositiveFeature are real binary flags
# derived from the Assessor's own Condition1/Condition2 fields (see
# data_pipeline.load_raw_data) -- clients send them as 0/1 directly.
FEATURE_TYPES: Dict[str, str] = {
    "OverallQual": "float", "OverallCond": "float",
    "LivingAreaSqFt": "float", "LotAreaSqFt": "float", "GarageAreaSqFt": "float",
    "YearBuilt": "float", "YearRenovated": "float",
    "BedroomAbvGr": "float", "FullBath": "float", "TotRmsAbvGrd": "float",
    "NearRailroad": "int", "NearArtery": "int", "NearPositiveFeature": "int",
    "Neighborhood": "str",
}


@dataclass
class FeatureSchema:
    """
    Frozen, joblib-serializable description of the model's expected input
    contract. Built once by `export_pipeline.py` from the real training
    data and shipped inside the model bundle so `api_service.py` validates
    against exactly what the deployed model was trained on -- never a
    hand-maintained, driftable copy.
    """
    feature_types: Dict[str, str]        # column -> type name
    known_neighborhoods: List[str]        # real neighborhoods seen in training
    required_columns: List[str] = field(default_factory=list)
    date_col: str = config.DATE_COL

    @classmethod
    def build(cls, known_neighborhoods: List[str]) -> "FeatureSchema":
        return cls(
            feature_types=dict(FEATURE_TYPES),
            known_neighborhoods=sorted(known_neighborhoods),
            required_columns=list(FEATURE_TYPES.keys()) + [config.DATE_COL],
        )


def validate_payload(record: Dict[str, Any], schema: FeatureSchema) -> List[str]:
    """
    Validates a single incoming JSON record against `schema`. Returns a
    list of human-readable error strings (empty list if valid). Never
    raises on its own -- callers (api_service.py) decide whether a
    non-empty error list means HTTP 422.
    """
    errors: List[str] = []
    for col in schema.required_columns:
        if col not in record or record[col] is None:
            errors.append(f"missing required field: {col!r}")

    for col, type_name in schema.feature_types.items():
        if col in record and record[col] is not None:
            val = record[col]
            if type_name in ("float", "int"):
                try:
                    float(val)
                except (TypeError, ValueError):
                    errors.append(f"field {col!r} must be numeric, got {val!r}")
            elif type_name == "str" and not isinstance(val, str):
                errors.append(f"field {col!r} must be a string, got {val!r}")
    return errors


def unseen_neighborhood_warning(record: Dict[str, Any], schema: FeatureSchema) -> Optional[str]:
    """
    Non-fatal check: flags a real Neighborhood value that was never seen
    during training. The target encoder still scores it fine (via its
    global-mean fallback -- see data_pipeline.KFoldTargetEncoder), so this
    is a warning for the client, not a rejection.
    """
    nbhd = record.get("Neighborhood")
    if nbhd is not None and nbhd not in schema.known_neighborhoods:
        return (
            f"Neighborhood {nbhd!r} was not observed during training -- scored using "
            f"the target encoder's global-mean fallback, which may reduce accuracy."
        )
    return None
