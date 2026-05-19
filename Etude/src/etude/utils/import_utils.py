from __future__ import annotations

from importlib import import_module
from typing import Any


def load_symbol(path: str) -> Any:
    if ":" in path:
        module_path, attr_path = path.split(":", maxsplit=1)
    else:
        module_path, _, attr_path = path.rpartition(".")
    if not module_path or not attr_path:
        raise ValueError(f"Import path must look like 'package.module:symbol' or 'package.module.symbol': {path}")
    module = import_module(module_path)
    value: Any = module
    for attr in attr_path.split("."):
        value = getattr(value, attr)
    return value
