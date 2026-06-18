"""
Shared notebook bootstrap for local runs and Google Colab.

Import ``bootstrap_notebook_environment()`` from the first code cell of every
notebook (01–07). Legacy cells that use ``Path.cwd().parent`` remain valid
because bootstrap sets the working directory to ``<repo>/notebooks``.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Optional, Tuple


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    notebooks: Path
    data: Path
    raw: Path
    processed: Path
    embeddings: Path
    rare_mutations: Path
    results: Path
    figures: Path
    in_colab: bool


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def mount_google_drive(force: bool = True) -> bool:
    """Mount Google Drive when running inside Colab."""
    if not is_colab():
        return False
    if not force and Path('/content/drive/MyDrive').exists():
        return True
    from google.colab import drive

    drive.mount('/content/drive', force_remount=False)
    return True


def resolve_repo_root(start: Optional[Path] = None) -> Path:
    """Resolve repository root from REPO_DIR or filesystem heuristics."""
    env_root = os.environ.get('REPO_DIR')
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if (root / 'src' / 'improved_pipeline.py').exists():
            return root

    candidates = []
    if start is not None:
        candidates.append(start.resolve())
    candidates.extend([
        Path.cwd().resolve(),
        Path.cwd().resolve().parent,
        Path(__file__).resolve().parent.parent,
    ])
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / 'src' / 'improved_pipeline.py').exists():
            return candidate
        if (candidate / 'notebooks' / 'pipeline_config.py').exists():
            return candidate
    raise FileNotFoundError(
        'Could not locate HIV-ESM-2 repository root. '
        'Set REPO_DIR to your clone path (e.g. /content/drive/MyDrive/HIV-ESM-2).'
    )


def _pip_install_colab_dependencies(repo_root: Path) -> None:
    """Install Colab-compatible dependencies without fighting the preinstalled torch."""
    req_colab = repo_root / 'requirements-colab.txt'
    req_default = repo_root / 'requirements.txt'
    req_file = req_colab if req_colab.exists() else req_default
    cmd = [
        sys.executable, '-m', 'pip', 'install', '-q',
        '-r', str(req_file),
        'optuna>=3.0.0',
    ]
    subprocess.check_call(cmd)


def build_project_paths(repo_root: Path, in_colab: bool) -> ProjectPaths:
    data = repo_root / 'data'
    for directory in (
        data / 'raw',
        data / 'processed',
        data / 'embeddings',
        data / 'rare_mutations',
        repo_root / 'results',
        repo_root / 'figures',
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return ProjectPaths(
        root=repo_root,
        notebooks=repo_root / 'notebooks',
        data=data,
        raw=data / 'raw',
        processed=data / 'processed',
        embeddings=data / 'embeddings',
        rare_mutations=data / 'rare_mutations',
        results=repo_root / 'results',
        figures=repo_root / 'figures',
        in_colab=in_colab,
    )


def configure_sys_path(paths: ProjectPaths) -> None:
    for entry in (paths.root, paths.root / 'src', paths.notebooks):
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)


def load_pipeline_config(project_root: Path) -> ModuleType:
    """Load ``notebooks/pipeline_config.py`` as a module."""
    config_path = project_root / 'notebooks' / 'pipeline_config.py'
    if not config_path.exists():
        raise FileNotFoundError(f'Pipeline config not found: {config_path}')

    spec = importlib.util.spec_from_file_location('pipeline_config', config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bootstrap_notebook_environment(
    install_deps: bool = False,
    mount_drive: bool = True,
    repo_root: Optional[Path] = None,
) -> Tuple[ProjectPaths, ModuleType]:
    """
    One-call setup for notebooks 01–07.

    - Mounts Drive on Colab (optional)
    - Resolves repo root and exports REPO_DIR
    - Creates standard data/results directories
    - Adds src/ and notebooks/ to sys.path
    - Sets cwd to notebooks/ so legacy cells using Path.cwd().parent work
    """
    in_colab = is_colab()
    if in_colab and mount_drive:
        mount_google_drive()

    root = repo_root or resolve_repo_root()
    os.environ['REPO_DIR'] = str(root)
    paths = build_project_paths(root, in_colab=in_colab)
    configure_sys_path(paths)

    if install_deps and in_colab:
        _pip_install_colab_dependencies(root)

    os.chdir(paths.notebooks)
    pipeline_config = load_pipeline_config(root)
    return paths, pipeline_config


def export_notebook_globals(paths: ProjectPaths, pipeline_config: ModuleType) -> dict:
    """Dictionary of names injected into notebook namespace."""
    return {
        'PROJECT_ROOT': paths.root,
        'REPO_DIR': str(paths.root),
        'DATA_DIR': paths.data,
        'RAW_DIR': paths.raw,
        'PROCESSED_DIR': paths.processed,
        'EMBEDDINGS_DIR': paths.embeddings,
        'RARE_MUTATIONS_DIR': paths.rare_mutations,
        'RESULTS_DIR': paths.results,
        'FIGURES_DIR': paths.figures,
        'IN_COLAB': paths.in_colab,
        'pipeline_config': pipeline_config,
        'ENABLE_IMPROVED_PIPELINE': pipeline_config.ENABLE_IMPROVED_PIPELINE,
    }
