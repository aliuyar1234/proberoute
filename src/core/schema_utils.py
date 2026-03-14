from __future__ import annotations

from pathlib import Path
from typing import Any

import jsonschema

from .constants import SCHEMA_PATHS
from .io_utils import read_json


def validate_payload(payload: dict[str, Any], schema_name: str) -> None:
    schema_path = SCHEMA_PATHS[schema_name]
    schema = read_json(schema_path)
    jsonschema.validate(payload, schema)

