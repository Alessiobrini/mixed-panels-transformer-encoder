import subprocess
import sys
from pathlib import Path

# Set project root
project_root = Path(__file__).resolve().parents[1]

# Define script paths
scripts = {
    "convert_fred": project_root / "src" / "data" / "convert_fred_to_long.py",
    "train": project_root / "src" / "train.py",
    "ar": project_root / "src" / "models" / "ar.py",
    "midas": project_root / "src" / "models" / "midas.R",
    "evaluate": project_root / "src" / "evaluation" / "evaluate_forecasts.py"
}

# Commands for each script
commands = [
    [sys.executable, str(scripts["convert_fred"])],
    [sys.executable, str(scripts["train"])],
    [sys.executable, str(scripts["ar"])],
    [r"C:\Program Files\R\R-4.5.1\bin\Rscript.exe", str(scripts["midas"])],  # <-- Full path here
    [sys.executable, str(scripts["evaluate"])]
]

# Run scripts in order and capture output
for name, cmd in zip(scripts.keys(), commands):
    print(f"\n=== Running {name.upper()} ===")
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"ERROR in {name.upper()}:\n{e.stderr}")
        break
