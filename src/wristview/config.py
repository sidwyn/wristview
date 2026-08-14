"""Run configuration.

One YAML file per run. The file is hashed into every stage's meta.json, so a
run directory always records the settings that produced it.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge override into base. Nested dicts merge, everything else replaces."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


@dataclass
class Config:
    """A resolved run configuration."""

    data: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    @classmethod
    def load(cls, path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
        """Load a config file, layered over the packaged default."""
        with open(DEFAULT_CONFIG_PATH) as handle:
            data = yaml.safe_load(handle) or {}

        source = None
        if path is not None:
            source = Path(path).resolve()
            if source != DEFAULT_CONFIG_PATH.resolve():
                with open(source) as handle:
                    data = _deep_merge(data, yaml.safe_load(handle) or {})

        if overrides:
            data = _deep_merge(data, overrides)

        return cls(data=data, source_path=source)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Read a nested value with a dotted path, for example `ingest.scan_fps`."""
        node: Any = self.data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict[str, Any]:
        """Return one top-level section as a plain dict."""
        value = self.data.get(name, {})
        return copy.deepcopy(value) if isinstance(value, dict) else {}

    @property
    def hash(self) -> str:
        """A stable hash of the resolved config. Recorded in every meta.json."""
        canonical = json.dumps(self.data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def write(self, path: str | Path) -> Path:
        """Write the resolved config next to the run, so the run is reproducible."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as handle:
            yaml.safe_dump(self.data, handle, sort_keys=True, default_flow_style=False)
        return target
