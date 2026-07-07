"""
Module 7: Model Registry & Versioning

Tracks all adaptive weight configurations and scoring models.
Supports:
  - Saving named model versions
  - Rollback to previous version
  - A/B comparison of model performance
  - Automatic archiving of underperforming models
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelVersion:
    name:         str
    version:      int
    created_at:   float
    weights:      dict          # expert weights per regime
    performance:  dict          # metrics at time of save
    description:  str = ""
    is_active:    bool = False
    trades_since_save: int = 0


class ModelRegistry:
    """Stores and manages weight model versions."""

    def __init__(self, registry_path: str = "model_registry.json"):
        self.registry_path = registry_path
        self._models: Dict[str, List[ModelVersion]] = {}
        self._active: Dict[str, str] = {}  # name -> version label
        self._load()

    def save_model(
        self,
        name:        str,
        weights:     dict,
        performance: dict,
        description: str = "",
    ) -> ModelVersion:
        versions = self._models.get(name, [])
        version  = len(versions) + 1
        model    = ModelVersion(
            name=name,
            version=version,
            created_at=time.time(),
            weights=weights,
            performance=performance,
            description=description,
            is_active=True,
        )
        # Deactivate previous
        for m in versions:
            m.is_active = False
        versions.append(model)
        self._models[name] = versions
        self._active[name] = f"{name}_v{version}"
        self._save()
        return model

    def get_active_weights(self, name: str) -> Optional[dict]:
        models = self._models.get(name, [])
        for m in reversed(models):
            if m.is_active:
                return m.weights
        return None

    def rollback(self, name: str, version: int) -> bool:
        """Activate a specific version, deactivate others."""
        models = self._models.get(name, [])
        for m in models:
            m.is_active = (m.version == version)
        success = any(m.is_active for m in models)
        if success:
            self._save()
        return success

    def list_versions(self, name: str) -> list:
        return [
            {
                "version":    m.version,
                "created_at": m.created_at,
                "description":m.description,
                "is_active":  m.is_active,
                "performance":m.performance,
            }
            for m in self._models.get(name, [])
        ]

    def compare_models(self, name: str) -> dict:
        """Show performance diff between current and previous version."""
        models = self._models.get(name, [])
        if len(models) < 2:
            return {"message": "Need at least 2 versions to compare"}
        curr = models[-1]; prev = models[-2]
        diff: dict = {}
        for key in curr.performance:
            if key in prev.performance:
                curr_v = curr.performance[key]
                prev_v = prev.performance[key]
                if isinstance(curr_v, (int, float)) and isinstance(prev_v, (int, float)):
                    diff[key] = {
                        "current": curr_v, "previous": prev_v,
                        "delta": round(float(curr_v) - float(prev_v), 4),
                    }
        return {"current_v": curr.version, "previous_v": prev.version, "diff": diff}

    def _load(self):
        if not os.path.exists(self.registry_path):
            return
        try:
            with open(self.registry_path) as f:
                data = json.load(f)
            for name, versions in data.get("models", {}).items():
                self._models[name] = [ModelVersion(**v) for v in versions]
            self._active = data.get("active", {})
        except Exception:
            pass

    def _save(self):
        try:
            data = {
                "models": {name: [asdict(m) for m in versions]
                           for name, versions in self._models.items()},
                "active": self._active,
            }
            with open(self.registry_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
