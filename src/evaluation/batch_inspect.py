"""Run model inspection across multiple experiments."""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from src.evaluation.inspect_model import reload_from_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "src" / "config" / "cfg.yaml"

# Populate this list with experiment names or paths.
EXPERIMENTS = [
    "example_experiment",
]

# Alternatively, set BASE_DATE to process every experiment folder containing this
# substring in its name (e.g., "2024-07-01"). When BASE_DATE is provided,
# EXPERIMENTS is ignored.
BASE_DATE: str | None = '2025-09-26'


def _prepare_temp_config(experiment: str) -> Path:
    cfg_data = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8")) or {}
    evaluation_cfg = cfg_data.get("evaluation") or {}
    evaluation_cfg["experiment"] = experiment
    cfg_data["evaluation"] = evaluation_cfg

    temp_file = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    try:
        yaml.safe_dump(cfg_data, temp_file)
        return Path(temp_file.name)
    finally:
        temp_file.close()


def _resolve_experiment_names() -> list[str]:
    if BASE_DATE is None:
        return EXPERIMENTS

    experiments_dir = PROJECT_ROOT / "outputs" / "experiments"
    if not experiments_dir.exists():
        raise FileNotFoundError(
            f"No experiments directory found at {experiments_dir}. "
            "Cannot expand experiments by base date."
        )

    matches = [
        path.name for path in experiments_dir.iterdir() if path.is_dir() and BASE_DATE in path.name
    ]

    if not matches:
        raise FileNotFoundError(
            f"No experiment folders containing '{BASE_DATE}' found in {experiments_dir}."
        )

    return sorted(matches)


def main() -> None:
    experiments = _resolve_experiment_names()

    total = len(experiments)
    for index, experiment in enumerate(experiments, start=1):
        remaining = total - index
        print(
            f"[Batch Inspect] Processing {index}/{total}: {experiment} "
            f"({remaining} remaining)"
        )
        temp_config = _prepare_temp_config(experiment)
        try:
            reload_from_config(str(temp_config))
        finally:
            temp_config.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
