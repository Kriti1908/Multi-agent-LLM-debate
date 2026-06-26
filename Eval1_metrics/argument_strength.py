from collections import defaultdict
from pathlib import Path
import json
from utils import group_metric


ARG_ANALYSIS_DIR = Path("../argument_analysis")

SKIP_FOLDERS = {
    "reasoning_llama3.1-8b_gemma3-12b"
}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_all_argument_data():
    """
    Scans all folders in argument_analysis (except skipped ones),
    finds argument files and their corresponding debate files.
    Returns list of tuples: (folder_name, argument_data, debate_data)
    """
    results = []
    
    for folder in ARG_ANALYSIS_DIR.iterdir():
        if not folder.is_dir():
            continue
        
        if folder.name in SKIP_FOLDERS:
            continue
        
        arguments_dir = folder / "arguments"
        
        if not arguments_dir.exists():
            continue
        
        # Find all argument files
        for arg_file in arguments_dir.glob("*.arguments.json"):
            # Extract debate ID from filename (e.g., "3.arguments.json" -> "3")
            debate_id = arg_file.stem.replace(".arguments", "")
            
            # Find corresponding debate file
            debate_file = folder / f"{debate_id}.json"
            
            if not debate_file.exists():
                continue
            
            try:
                arg_data = load_json(arg_file)
                debate_data = load_json(debate_file)
                results.append((folder.name, arg_data, debate_data))
            except Exception as e:
                print(f"Error loading {arg_file} or {debate_file}: {e}")
                continue
    
    return results


def compute_all_argument_strength():
    """
    Computes argument strength scores for all debates.
    Returns: defaultdict[model][value] -> list of scores
    """
    stats = defaultdict(lambda: defaultdict(list))
    
    all_data = get_all_argument_data()
    
    print(f"\nFound {len(all_data)} debate-argument pairs to process")
    
    for folder_name, arg_data, debate_data in all_data:
        
        # Extract model and value info from debate data
        model1 = debate_data.get("model_1", "unknown")
        model2 = debate_data.get("model_2", "unknown")
        
        values1 = debate_data.get("values1", [])
        values2 = debate_data.get("values2", [])
        
        # Ensure values are lists
        if isinstance(values1, str):
            values1 = [values1]
        if isinstance(values2, str):
            values2 = [values2]
        
        # Process each argument
        for arg in arg_data.get("arguments", []):
            
            # Skip if arg is not a dictionary
            if not isinstance(arg, dict):
                continue
            
            arg_type = arg.get("type", "").lower()
            speaker = arg.get("agent", "")
            
            # Convert type to score: strong = 1, weak = 0
            if arg_type == "strong":
                strength_score = 1
            elif arg_type == "weak":
                strength_score = 0
            else:
                # Skip unknown types
                continue
            
            # Determine which model/values based on agent name
            if "Agent1" in speaker:
                model = model1
                values = values1
            elif "Agent2" in speaker:
                model = model2
                values = values2
            else:
                # Skip if agent not recognized
                continue
            
            # Record score for each value associated with this model
            for v in values:
                stats[model][v].append(strength_score)
    
    return stats


def compute_strength(run, arg_analysis):
    """
    Legacy function for compatibility with metric_runner.py
    Computes argument strength for a single debate.
    """
    model1 = run["model_1"]
    model2 = run["model_2"]

    values1 = run["values1"]
    values2 = run["values2"]

    if isinstance(values1, str):
        values1 = [values1]

    if isinstance(values2, str):
        values2 = [values2]

    stats = defaultdict(lambda: defaultdict(list))

    for arg in arg_analysis.get("arguments", []):

        # Skip if arg is not a dictionary
        if not isinstance(arg, dict):
            continue

        arg_type = arg.get("type", "").lower()
        strength = 1 if arg_type == "strong" else 0

        speaker = arg.get("agent", "")

        if "Agent1" in speaker:
            for v in values1:
                stats[model1][v].append(strength)
        elif "Agent2" in speaker:
            for v in values2:
                stats[model2][v].append(strength)

    return stats


if __name__ == "__main__":
    # Standalone execution for testing
    import pandas as pd
    
    stats = compute_all_argument_strength()
    
    # Aggregate results
    rows = []
    for model in stats:
        for value in stats[model]:
            scores = stats[model][value]
            if len(scores) == 0:
                continue
            rows.append({
                "model": model,
                "value": value,
                "avg_strength": sum(scores) / len(scores),
                "total_arguments": len(scores),
                "strong_count": sum(scores),
                "weak_count": len(scores) - sum(scores),
                "min": min(scores),
                "max": max(scores)
            })
    
    df = pd.DataFrame(rows)
    
    if not df.empty:
        df = df.sort_values(["model", "value"])
        print("\n" + "="*60)
        print("ARGUMENT STRENGTH RESULTS")
        print("="*60)
        print(df.to_string(index=False))
        
        # Save results
        output_dir = Path("metric_outputs")
        output_dir.mkdir(exist_ok=True)
        
        df.to_csv(output_dir / "argument_strength.csv", index=False)
        df.to_json(output_dir / "argument_strength.json", orient="records", indent=2)
        
        print(f"\nSaved results to {output_dir}/argument_strength.csv")