import json
from pathlib import Path

# Path to your saved file
p = Path(r"G:\other computers\dell duke\workfiles\postdoc_file\peer_review_research\timeseriesattention\mixed-frequency-attention\outputs\experiments\GDPC1_2025-09-26\model_inspection\attention_analysis\attention_summary.json")

with p.open("r", encoding="utf-8") as f:
    data = json.load(f)

# Print top-level structure
print("Top-level keys:")
for k in data.keys():
    print("  -", k)

print("\nSummary fields:")
print("experiment:", data["experiment"])
print("num_sequences:", data["num_sequences"])
print("num_heads:", data["num_heads"])
print("n_monthly:", data["n_monthly"])
print("n_quarterly:", data["n_quarterly"])

# Show structure of one per-sequence entry
print("\nStructure of data['per_sequence'][0]:")
example = data["per_sequence"][0]
for k, v in example.items():
    if isinstance(v, list):
        print(f"  {k}: list[{len(v)}]")
    elif isinstance(v, dict):
        print(f"  {k}: dict with keys {list(v.keys())}")
    else:
        print(f"  {k}: {type(v).__name__}")

# Show shape of mean matrices
print("\nShape info:")
print("mean_by_sequence Ax size:", len(data["mean_by_sequence"]["Ax"]), "x", len(data["mean_by_sequence"]["Ax"][0]))
print("overall_mean Ax size:", len(data["overall_mean"]["Ax"]), "x", len(data["overall_mean"]["Ax"][0]))
