#!/usr/bin/env python3
"""
Generate 100 random annotation rows from debate argument analysis.

Randomly samples individual arguments (not full debates) across different
value pairs and model configurations from the argument_analysis folder.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd

# Configuration
ARGUMENT_ANALYSIS_ROOT = Path(__file__).parent / "argument_analysis"
OUTPUT_EXCEL = Path(__file__).parent / "annotation_sample_100.xlsx"
NUM_ROWS = 100
RANDOM_SEED = 42

def find_all_argument_files(root: Path) -> List[Tuple[Path, Path]]:
    """
    Find all (debate_file, argument_file) pairs.
    
    Returns: List of tuples (debate_json_path, argument_json_path)
    """
    pairs = []
    
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
            
        arguments_dir = run_dir / "arguments"
        if not arguments_dir.exists():
            continue
        
        for arg_file in arguments_dir.glob("*.arguments.json"):
            # Extract debate ID from filename (e.g., "9.arguments.json" -> "9")
            debate_id = arg_file.stem.replace(".arguments", "")
            debate_file = run_dir / f"{debate_id}.json"
            
            if debate_file.exists():
                pairs.append((debate_file, arg_file))
    
    return pairs

def extract_turn_number(argument_text: str, messages: List[Dict]) -> int:
    """
    Find which turn contains the argument text.
    Returns turn number (1-indexed, counting only agent turns, excluding UserProxy).
    """
    agent_turn = 0
    for msg in messages:
        name = msg.get("name", "")
        content = msg.get("content", "")
        
        # Skip UserProxy messages
        if "UserProxy" in name or "user" in name.lower():
            continue
            
        agent_turn += 1
        
        # Check if argument text appears in this message
        if argument_text in content:
            return agent_turn
    
    return 0  # Not found

def load_argument_row(debate_file: Path, argument_file: Path, arg_index: int) -> Dict[str, Any]:
    """
    Load a single argument row from the files.
    
    Args:
        debate_file: Path to debate JSON
        argument_file: Path to arguments JSON
        arg_index: Index of the argument to extract
    
    Returns: Dictionary with annotation row data
    """
    with open(debate_file) as f:
        debate = json.load(f)
    
    with open(argument_file) as f:
        arg_analysis = json.load(f)
    
    arguments = arg_analysis.get("arguments", [])
    if arg_index >= len(arguments):
        return None
    
    arg = arguments[arg_index]
    
    # USE FILENAME AS DEBATE_ID (not the JSON field)
    debate_id = debate_file.stem  # e.g., "9", "17", "1000"
    
    # Extract metadata
    model_1 = debate.get("model_1", "")
    model_2 = debate.get("model_2", "")
    values1 = debate.get("values1", "")
    values2 = debate.get("values2", "")
    
    value_pair = f"{values1} vs {values2}"
    model_pair = f"{model_1} vs {model_2}"
    
    # Get argument details
    argument_text = arg.get("argument", "")
    agent = arg.get("agent", "")
    llm_type = arg.get("type", "")
    llm_response = arg.get("response", "")
    
    # Find turn number
    messages = debate.get("messages", [])
    turn_number = extract_turn_number(argument_text, messages)
    
    # Build row
    row = {
        # Metadata columns
        "debate_id": debate_id,
        "value_pair": value_pair,
        "sample_id": "",  # To be filled by annotator
        "annotator": "",  # To be filled by annotator
        "model_pair": model_pair,
        
        # Argument-level columns
        "agent": agent,
        "turn_number": turn_number,
        "argument_text": argument_text,
        
        # LLM prediction columns (pre-filled)
        "LLM_type": llm_type,
        "LLM_response": llm_response,
        
        # Human annotation columns (empty for annotator to fill)
        "human_type": "",
        "human_response": "",
        "argument_strength_score": "",
        "persuasiveness_score": "",
        
        # Debate-level evaluation (empty for annotator to fill)
        "human_final_verdict": "",
        "confidence": "",
        "notes": ""
    }
    
    return row

def collect_all_argument_indices(root: Path) -> List[Tuple[Path, Path, int]]:
    """
    Create a list of (debate_file, argument_file, argument_index) tuples.
    Each tuple represents one extractable argument row.
    
    Returns: List of tuples that can be randomly sampled
    """
    pairs = find_all_argument_files(root)
    
    all_indices = []
    
    for debate_file, arg_file in pairs:
        try:
            with open(arg_file) as f:
                arg_data = json.load(f)
            
            num_args = len(arg_data.get("arguments", []))
            
            # Add one tuple for each argument in this file
            for i in range(num_args):
                all_indices.append((debate_file, arg_file, i))
        
        except Exception as e:
            print(f"⚠️  Warning: Could not read {arg_file}: {e}")
            continue
    
    return all_indices

def main():
    print(f"🔍 Scanning {ARGUMENT_ANALYSIS_ROOT} for argument files...")
    
    # Collect all possible argument rows
    all_indices = collect_all_argument_indices(ARGUMENT_ANALYSIS_ROOT)
    
    print(f"📊 Found {len(all_indices)} total argument rows across all debates")
    
    if len(all_indices) == 0:
        print("❌ No argument files found! Check the directory structure.")
        return
    
    # Randomly sample
    random.seed(RANDOM_SEED)
    sample_size = min(NUM_ROWS, len(all_indices))
    sampled_indices = random.sample(all_indices, sample_size)
    
    print(f"🎲 Randomly sampled {sample_size} argument rows")
    
    # Load sampled rows
    rows = []
    failed = 0
    
    print("\n📝 Loading argument data...")
    for debate_file, arg_file, arg_idx in sampled_indices:
        try:
            row = load_argument_row(debate_file, arg_file, arg_idx)
            if row:
                rows.append(row)
            else:
                failed += 1
        except Exception as e:
            print(f"⚠️  Failed to load argument {arg_idx} from {arg_file.name}: {e}")
            failed += 1
    
    print(f"\n✅ Successfully loaded {len(rows)} argument rows")
    if failed > 0:
        print(f"❌ Failed to load {failed} rows")
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Sort by debate_id numerically for better readability
    df['debate_id_numeric'] = pd.to_numeric(df['debate_id'], errors='coerce')
    df = df.sort_values('debate_id_numeric').drop('debate_id_numeric', axis=1)
    
    # Reorder columns
    column_order = [
        "debate_id", "value_pair", "sample_id", "annotator", "model_pair",
        "agent", "turn_number", "argument_text",
        "LLM_type", "LLM_response",
        "human_type", "human_response",
        "argument_strength_score", "persuasiveness_score",
        "human_final_verdict", "confidence", "notes"
    ]
    
    df = df[column_order]
    
    # Write to Excel
    print(f"\n💾 Writing to {OUTPUT_EXCEL}...")
    
    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
        # Main annotation sheet
        df.to_excel(writer, sheet_name='Annotations', index=False)
        
        # Summary sheet
        summary_data = {
            "Metric": [
                "Total Argument Rows",
                "Unique Debates",
                "Unique Value Pairs",
                "Unique Model Pairs",
                "Failed Loads"
            ],
            "Value": [
                len(rows),
                df['debate_id'].nunique(),
                df['value_pair'].nunique(),
                df['model_pair'].nunique(),
                failed
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Value pair distribution
        value_dist = df['value_pair'].value_counts().reset_index()
        value_dist.columns = ['Value Pair', 'Count']
        value_dist.to_excel(writer, sheet_name='Value Distribution', index=False)
        
        # Model pair distribution
        model_dist = df['model_pair'].value_counts().reset_index()
        model_dist.columns = ['Model Pair', 'Count']
        model_dist.to_excel(writer, sheet_name='Model Distribution', index=False)
        
        # Debate distribution
        debate_dist = df['debate_id'].value_counts().reset_index()
        debate_dist.columns = ['Debate ID', 'Argument Count']
        debate_dist = debate_dist.sort_values('Debate ID')
        debate_dist.to_excel(writer, sheet_name='Debate Distribution', index=False)
    
    print(f"✨ Successfully created annotation sheet: {OUTPUT_EXCEL}")
    print(f"\n📋 Sheet contains {len(df)} rows ready for annotation")
    print(f"   - {df['debate_id'].nunique()} unique debates")
    print(f"   - {df['value_pair'].nunique()} unique value pairs")
    print(f"   - {df['model_pair'].nunique()} unique model pairs")
    
    print("\n📈 Sample debate IDs (first 10):")
    print(df['debate_id'].unique()[:10])
    
    print("\n📈 Top 5 value pairs by sample count:")
    print(df['value_pair'].value_counts().head())
    
    print("\n🎯 Next steps:")
    print("   1. Open the Excel file")
    print("   2. Fill in the empty columns (human_type, human_response, etc.)")
    print("   3. Assign sample_id for each debate")
    print("   4. Add your annotator ID")
    print("   5. Save and use for analysis!")

if __name__ == "__main__":
    main()
