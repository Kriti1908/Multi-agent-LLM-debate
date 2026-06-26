#!/usr/bin/env python3
"""
Generate annotation Excel sheet from debate argument analysis.

Randomly samples ~100 debate pairs from the argument_analysis folder
and creates an Excel file with pre-filled LLM predictions for human annotation.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from collections import defaultdict

# Configuration
ARGUMENT_ANALYSIS_ROOT = Path(__file__).parent / "argument_analysis"
OUTPUT_EXCEL = Path(__file__).parent / "annotation_sheet.xlsx"
SAMPLE_SIZE = 100
RANDOM_SEED = 42

def find_debate_pairs(root: Path) -> List[Dict[str, Path]]:
    """
    Find all debate files that have corresponding argument analysis.
    
    Returns list of dicts with 'debate_file' and 'argument_file' paths.
    """
    pairs = []
    
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
            
        arguments_dir = run_dir / "arguments"
        if not arguments_dir.exists():
            continue
        
        # Find all debate JSON files
        for debate_file in run_dir.glob("*.json"):
            debate_id = debate_file.stem
            argument_file = arguments_dir / f"{debate_id}.arguments.json"
            
            if argument_file.exists():
                pairs.append({
                    "run_name": run_dir.name,
                    "debate_file": debate_file,
                    "argument_file": argument_file,
                    "debate_id": debate_id
                })
    
    return pairs

def extract_turn_number(argument_text: str, messages: List[Dict]) -> int:
    """
    Find which turn contains the argument text.
    Returns turn number (1-indexed, counting only agent turns).
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
    
    # If not found, return 0 (indicates error)
    return 0

def load_debate_data(debate_file: Path, argument_file: Path) -> List[Dict[str, Any]]:
    """
    Load debate and argument analysis, return list of annotation rows.
    """
    with open(debate_file) as f:
        debate = json.load(f)
    
    with open(argument_file) as f:
        arg_analysis = json.load(f)
    
    debate_id = debate.get("debate_id", debate_file.stem)
    
    # Extract metadata
    model_1 = debate.get("model_1", "")
    model_2 = debate.get("model_2", "")
    values1 = debate.get("values1", "")
    values2 = debate.get("values2", "")
    
    value_pair = f"{values1} vs {values2}"
    model_pair = f"{model_1} vs {model_2}"
    
    # Get messages for turn number mapping
    messages = debate.get("messages", [])
    
    rows = []
    for arg in arg_analysis.get("arguments", []):
        argument_text = arg.get("argument", "")
        agent = arg.get("agent", "")
        llm_type = arg.get("type", "")
        llm_response = arg.get("response", "")
        
        # Find turn number
        turn_number = extract_turn_number(argument_text, messages)
        
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
        
        rows.append(row)
    
    return rows

def main():
    print(f"🔍 Scanning {ARGUMENT_ANALYSIS_ROOT} for debate pairs...")
    
    # Find all valid debate pairs
    all_pairs = find_debate_pairs(ARGUMENT_ANALYSIS_ROOT)
    print(f"📊 Found {len(all_pairs)} debate pairs with argument analysis")
    
    if len(all_pairs) == 0:
        print("❌ No debate pairs found! Check the directory structure.")
        return
    
    # Sample randomly
    random.seed(RANDOM_SEED)
    sample_size = min(SAMPLE_SIZE, len(all_pairs))
    sampled_pairs = random.sample(all_pairs, sample_size)
    
    print(f"🎲 Randomly sampled {sample_size} debate pairs")
    
    # Group by run for summary
    runs_summary = defaultdict(int)
    for pair in sampled_pairs:
        runs_summary[pair["run_name"]] += 1
    
    print("\n📈 Sample distribution by run:")
    for run_name, count in sorted(runs_summary.items()):
        print(f"   {run_name}: {count} debates")
    
    # Load all annotation rows
    all_rows = []
    failed = []
    
    print("\n📝 Loading debate data...")
    for pair in sampled_pairs:
        try:
            rows = load_debate_data(pair["debate_file"], pair["argument_file"])
            all_rows.extend(rows)
        except Exception as e:
            failed.append({
                "debate_id": pair["debate_id"],
                "run": pair["run_name"],
                "error": str(e)
            })
            print(f"⚠️  Failed to load {pair['debate_id']} from {pair['run_name']}: {e}")
    
    print(f"\n✅ Successfully loaded {len(all_rows)} argument rows from {sample_size - len(failed)} debates")
    
    if failed:
        print(f"❌ Failed to load {len(failed)} debates")
    
    # Create DataFrame
    df = pd.DataFrame(all_rows)
    
    # Reorder columns to match annotation guide
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
        
        # Create a summary sheet
        summary_data = {
            "Metric": [
                "Total Debates Sampled",
                "Total Arguments to Annotate",
                "Average Arguments per Debate",
                "Unique Value Pairs",
                "Unique Model Pairs",
                "Failed Loads"
            ],
            "Value": [
                sample_size - len(failed),
                len(all_rows),
                len(all_rows) / (sample_size - len(failed)) if sample_size - len(failed) > 0 else 0,
                df['value_pair'].nunique(),
                df['model_pair'].nunique(),
                len(failed)
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Distribution by run
        run_dist_df = pd.DataFrame([
            {"Run Name": run, "Count": count}
            for run, count in sorted(runs_summary.items())
        ])
        run_dist_df.to_excel(writer, sheet_name='Run Distribution', index=False)
        
        # Failed loads (if any)
        if failed:
            failed_df = pd.DataFrame(failed)
            failed_df.to_excel(writer, sheet_name='Failed Loads', index=False)
    
    print(f"✨ Successfully created annotation sheet: {OUTPUT_EXCEL}")
    print(f"\n📋 Sheet contains {len(df)} rows ready for annotation")
    print(f"   - {df['debate_id'].nunique()} unique debates")
    print(f"   - {df['value_pair'].nunique()} unique value pairs")
    print(f"   - {df['model_pair'].nunique()} unique model pairs")
    
    print("\n🎯 Next steps:")
    print("   1. Open the Excel file")
    print("   2. Fill in the empty columns (human_type, human_response, etc.)")
    print("   3. Assign sample_id for each debate")
    print("   4. Add your annotator ID")
    print("   5. Save and use for analysis!")

if __name__ == "__main__":
    main()
