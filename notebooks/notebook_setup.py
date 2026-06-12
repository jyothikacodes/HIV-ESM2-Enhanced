"""
Helpers imported by notebook setup cells to load shared pipeline configuration.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_pipeline_config(project_root: Path) -> ModuleType:
    """Load ``notebooks/pipeline_config.py`` as a module."""
    config_path = project_root / 'notebooks' / 'pipeline_config.py'
    if not config_path.exists():
        raise FileNotFoundError(f'Pipeline config not found: {config_path}')

    spec = importlib.util.spec_from_file_location('pipeline_config', config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
