from typing import Any, Dict

def safe_get(d: Dict[str, Any], keys: list, default: Any = None) -> Any:
    """Safely get a nested key from a dictionary."""
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return default
    return d
