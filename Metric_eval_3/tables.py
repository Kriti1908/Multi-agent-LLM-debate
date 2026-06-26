import json
from pathlib import Path
import pandas as pd

# Folders (same directory as this script)
FOLDERS = [
    "Reasoning vs Non-reasoning",
    "Runs/Untitled",
    "Small vs Large",
]

OUTPUT_DIR = Path("tables")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_metrics(path):
    with open(path, "r") as f:
        return json.load(f)


def normalize_pair(m1, m2):
    # ensures (A,B) and (B,A) are treated same
    return tuple(sorted([m1, m2]))


def process_folder(folder_name):
    folder_path = Path(folder_name)
    metrics_path = folder_path / "metrics.json"

    data = load_metrics(metrics_path)

    rows = []

    for entry in data:
        m1 = entry["Model1"]
        m2 = entry["Model2"]

        pair = normalize_pair(m1, m2)

        rows.append({
            "model_pair": f"{pair[0]} vs {pair[1]}",
            "TokenCost_Model1": entry["TokenCost_Model1"],
            "TokenCost_Model2": entry["TokenCost_Model2"],
            "Avg_Turns_To_Consensus": entry["Avg_Turns_To_Consensus"],
        })

    df = pd.DataFrame(rows)

    # Average over all value pairs for each model pair
    grouped = df.groupby("model_pair", as_index=False).mean(numeric_only=True)

    return grouped


def main():
    for folder in FOLDERS:
        result_df = process_folder(folder)

        # clean filename
        safe_name = safe_name = folder.replace(" ", "_").replace("/", "_").lower()
        output_path = OUTPUT_DIR / f"{safe_name}_table.csv"

        result_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()