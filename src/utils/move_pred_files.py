import shutil
from pathlib import Path

# Compute base project directory from script location
project_root = Path(__file__).resolve().parents[2]
experiments_dir = project_root / "outputs" / "experiments"

old_date = "2025-07-24"
new_date = "2025-09-26"
prefixes = ("xgb", "nn", "ols")

# Loop over all folders ending with old_date
for src_folder in experiments_dir.glob(f"*_{old_date}"):
    target_name = src_folder.name.split("_")[0]
    dst_folder = experiments_dir / f"{target_name}_{new_date}"

    if not dst_folder.exists():
        print(f"Skipping missing destination: {dst_folder}")
        continue

    # Copy matching files
    for file in src_folder.glob("*.csv"):
        if file.name.startswith(prefixes):
            shutil.copy(file, dst_folder / file.name)
            print(f"Copied {file.name} → {dst_folder.name}")
