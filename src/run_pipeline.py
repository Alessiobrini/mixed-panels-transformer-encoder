import platform
import subprocess
import sys
from pathlib import Path

# Set project root
project_root = Path(__file__).resolve().parents[1]

# Define script paths
scripts = {
    "convert_fred": project_root / "src" / "data" / "convert_fred_to_long.py",
    "simulate": project_root / "src" / "data" / "simulate_to_long.py",
    "train": project_root / "src" / "train.py",
    "ar": project_root / "src" / "models" / "ar.py",
    "midas": project_root / "src" / "models" / "midas.R",
    "evaluate": project_root / "src" / "evaluation" / "evaluate_forecasts.py",
}

sys.path.append(str(project_root))
from src.utils.config import Config  # noqa: E402
from src.utils.data_paths import is_simulation_enabled  # noqa: E402

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config", default=None, help="Path to config YAML (default: src/config/cfg.yaml)")
parser.add_argument("--rscript", default=None, help="Path to Rscript binary (overrides the platform default)")
_args, _ = parser.parse_known_args()

# Resolve to an ABSOLUTE path: midas.R setwd()s away from the project root, so a relative
# config path passed to it cannot be opened. Absolute fixes the data step, AR, MIDAS, evaluate.
cfg_path = (Path(_args.config).resolve() if _args.config
            else project_root / "src" / "config" / "cfg.yaml")
config = Config(cfg_path)
use_simulation = is_simulation_enabled(config)

if _args.rscript:
    rscript_path = _args.rscript
elif platform.system() == "Windows":
    rscript_path = r"C:\\Program Files\\R\\R-4.5.1\\bin\\Rscript.exe"
else:
    rscript_path = "/hpc/group/darec/ab978/miniconda3/envs/tsa-dev/bin/Rscript"

# Commands for each script. The Python steps take --config; midas.R takes the config path
# as its first positional argument. Threading the config through makes the pipeline safe to
# run as a parallel SLURM array (each task points at its own per-target config).
cfg_arg = ["--config", str(cfg_path)]
data_step = "simulate" if use_simulation else "convert_fred"
pipeline_steps = [
    (data_step, [sys.executable, str(scripts[data_step]), *cfg_arg]),
    ("train", [sys.executable, str(scripts["train"]), *cfg_arg]),
    ("ar", [sys.executable, str(scripts["ar"]), *cfg_arg]),
    ("midas", [rscript_path, str(scripts["midas"]), str(cfg_path)]),
    ("evaluate", [sys.executable, str(scripts["evaluate"]), *cfg_arg])
]

# Run scripts in order and capture output
for name, cmd in pipeline_steps:
    print(f"\n=== Running {name.upper()} ===")
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"ERROR in {name.upper()}:\n{e.stderr}")
        break
