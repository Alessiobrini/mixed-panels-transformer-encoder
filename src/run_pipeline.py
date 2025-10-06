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

cfg_path = project_root / "src" / "config" / "cfg.yaml"
config = Config(cfg_path)
use_simulation = is_simulation_enabled(config)

if platform.system() == "Windows":
    rscript_path = r"C:\\Program Files\\R\\R-4.5.1\\bin\\Rscript.exe"
else:
    rscript_path = "/hpc/group/darec/ab978/miniconda3/envs/tsa-dev/bin/Rscript"

# Commands for each script
data_step = "simulate" if use_simulation else "convert_fred"
pipeline_steps = [
    (data_step, [sys.executable, str(scripts[data_step])]),
    ("train", [sys.executable, str(scripts["train"])]),
    ("ar", [sys.executable, str(scripts["ar"])]),
    ("midas", [rscript_path, str(scripts["midas"])]),  # <-- Update the path to your local Rscript binary
    ("evaluate", [sys.executable, str(scripts["evaluate"])])
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
