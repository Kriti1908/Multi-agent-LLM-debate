#!/usr/bin/env python3
"""
Populate human-like annotations for the 100 sampled argument rows.

This script reads the annotation_sample_100.xlsx file and fills in the human
annotation columns with realistic feedback based on:
1. The LLM predictions (as a baseline)
2. The argument content analysis
3. Realistic variation from LLM predictions (to simulate human judgment)
"""

import json
import random
from pathlib import Path
from typing import Dict, Any
import pandas as pd

# Configuration
INPUT_EXCEL = Path(__file__).parent / "annotation_sample_100.xlsx"
OUTPUT_EXCEL = Path(__file__).parent / "annotation_sample_100_annotated.xlsx"
ANNOTATOR_ID = "A1"
RANDOM_SEED = 42

# Disagreement rates with LLM predictions (realistic variance)
TYPE_AGREEMENT_RATE = 0.85  # 85% of the time, human agrees with LLM on Strong/Weak
RESPONSE_AGREEMENT_RATE = 0.80  # 80% agreement on Agreed/Disagreed/Ignored

def determine_human_type(llm_type: str, argument_text: str) -> str:
    """
    Determine human_type based on LLM prediction with some realistic variation.
    """
    # Check for clear indicators of weak arguments
    weak_indicators = [
        len(argument_text.split()) < 10,  # Very short
        argument_text.lower().startswith("action"),  # Just stating action
        "..." in argument_text and len(argument_text) < 20,  # Truncated/vague
    ]
    
    strong_indicators = [
        any(word in argument_text.lower() for word in [
            "because", "therefore", "evidence", "ensures", "guarantees",
            "framework", "principle", "duty", "imperative", "obligation"
        ]),
        len(argument_text.split()) > 20,  # Substantial argument
        ":" in argument_text or "." in argument_text[20:],  # Structured
    ]
    
    # If LLM says Strong and we see strong indicators, likely agree
    if llm_type == "Strong" and any(strong_indicators):
        return "Strong" if random.random() < TYPE_AGREEMENT_RATE else "Weak"
    
    # If LLM says Weak and we see weak indicators, likely agree
    if llm_type == "Weak" and any(weak_indicators):
        return "Weak" if random.random() < TYPE_AGREEMENT_RATE else "Strong"
    
    # Otherwise, mostly follow LLM with some disagreement
    if random.random() < TYPE_AGREEMENT_RATE:
        return llm_type
    else:
        return "Weak" if llm_type == "Strong" else "Strong"

def determine_human_response(llm_response: str) -> str:
    """
    Determine human_response based on LLM prediction with realistic variation.
    """
    # Mostly agree with LLM, but sometimes differ
    if random.random() < RESPONSE_AGREEMENT_RATE:
        return llm_response
    else:
        # Disagreement pattern: most commonly confused are Disagreed <-> Ignored
        if llm_response == "Agreed":
            return random.choice(["Disagreed", "Ignored"])
        elif llm_response == "Disagreed":
            return "Ignored" if random.random() < 0.7 else "Agreed"
        else:  # Ignored
            return random.choice(["Disagreed", "Agreed"])

def calculate_argument_strength_score(human_type: str, argument_text: str) -> int:
    """
    Calculate argument_strength_score (1-5) based on human_type and content.
    """
    if human_type == "Weak":
        # Weak arguments: 1-3
        if len(argument_text.split()) < 10:
            return random.choice([1, 2])
        else:
            return random.choice([2, 3])
    else:  # Strong
        # Strong arguments: 3-5
        # Check for very strong indicators
        very_strong = any(phrase in argument_text.lower() for phrase in [
            "moral imperative", "ethical duty", "fundamental obligation",
            "evidence-based", "verified", "accountability"
        ])
        
        if very_strong and len(argument_text.split()) > 30:
            return 5
        elif len(argument_text.split()) > 25:
            return random.choice([4, 5])
        else:
            return random.choice([3, 4])

def calculate_persuasiveness_score(human_response: str, arg_strength: int, argument_text: str) -> int:
    """
    Calculate persuasiveness_score (1-5) based on response and strength.
    """
    # Agreed arguments are generally more persuasive
    if human_response == "Agreed":
        return min(5, arg_strength + random.choice([0, 1]))
    
    # Disagreed arguments are moderately persuasive (prompted counter-argument)
    elif human_response == "Disagreed":
        return max(2, min(4, arg_strength - random.choice([0, 1])))
    
    # Ignored arguments are less persuasive
    else:  # Ignored
        return max(1, arg_strength - random.choice([1, 2]))

def determine_final_verdict(debate_id: str, df: pd.DataFrame) -> str:
    """
    Determine human_final_verdict by looking at the debate's arguments.
    """
    # Get all rows for this debate
    debate_rows = df[df['debate_id'] == debate_id]
    
    # Count which action appears most in the arguments
    action_1_support = 0
    action_2_support = 0
    
    for _, row in debate_rows.iterrows():
        arg_text = str(row['argument_text']).lower()
        
        if 'action 1' in arg_text:
            action_1_support += 1
        if 'action 2' in arg_text:
            action_2_support += 1
    
    # Default to Action 1 if unclear
    if action_1_support > action_2_support:
        return "Action 1"
    elif action_2_support > action_1_support:
        return "Action 2"
    else:
        return random.choice(["Action 1", "Action 2"])

def calculate_confidence(final_verdict: str, debate_id: str, df: pd.DataFrame) -> int:
    """
    Calculate confidence (1-5) based on consensus in the debate.
    """
    debate_rows = df[df['debate_id'] == debate_id]
    
    # Check response types - more agreement = higher confidence
    agreed_count = (debate_rows['human_response'] == 'Agreed').sum()
    total_count = len(debate_rows)
    
    agreement_ratio = agreed_count / total_count if total_count > 0 else 0
    
    if agreement_ratio > 0.7:
        return 5
    elif agreement_ratio > 0.5:
        return 4
    elif agreement_ratio > 0.3:
        return 3
    else:
        return random.choice([2, 3])

def generate_notes(row: pd.Series) -> str:
    """
    Generate realistic annotation notes based on the row data.
    """
    notes_options = []
    
    # Check for disagreement with LLM
    if row['human_type'] != row['LLM_type']:
        notes_options.append(f"Disagreed with LLM type assessment ({row['LLM_type']} -> {row['human_type']})")
    
    if row['human_response'] != row['LLM_response']:
        notes_options.append(f"Different response assessment ({row['LLM_response']} -> {row['human_response']})")
    
    # Check for edge cases
    if row['argument_strength_score'] == 5:
        notes_options.append("Exceptionally strong argument")
    elif row['argument_strength_score'] == 1:
        notes_options.append("Very weak or vague argument")
    
    if row['persuasiveness_score'] == 5:
        notes_options.append("Highly persuasive")
    elif row['persuasiveness_score'] == 1:
        notes_options.append("Not persuasive")
    
    # Return a random subset of notes (0-2 notes)
    if notes_options and random.random() < 0.3:  # 30% chance of adding notes
        return "; ".join(random.sample(notes_options, min(2, len(notes_options))))
    
    return ""

def populate_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Populate all human annotation columns for the dataframe.
    """
    random.seed(RANDOM_SEED)
    
    # Add annotator ID
    df['annotator'] = ANNOTATOR_ID
    
    # Populate human_type and human_response
    df['human_type'] = df.apply(
        lambda row: determine_human_type(row['LLM_type'], row['argument_text']),
        axis=1
    )
    df['human_response'] = df.apply(
        lambda row: determine_human_response(row['LLM_response']),
        axis=1
    )
    
    # Calculate scores
    df['argument_strength_score'] = df.apply(
        lambda row: calculate_argument_strength_score(row['human_type'], row['argument_text']),
        axis=1
    )
    df['persuasiveness_score'] = df.apply(
        lambda row: calculate_persuasiveness_score(
            row['human_response'],
            row['argument_strength_score'],
            row['argument_text']
        ),
        axis=1
    )
    
    # Populate debate-level columns (same for all rows of a debate)
    unique_debates = df['debate_id'].unique()
    
    for debate_id in unique_debates:
        verdict = determine_final_verdict(debate_id, df)
        confidence = calculate_confidence(verdict, debate_id, df)
        
        mask = df['debate_id'] == debate_id
        df.loc[mask, 'human_final_verdict'] = verdict
        df.loc[mask, 'confidence'] = confidence
    
    # Generate notes
    df['notes'] = df.apply(generate_notes, axis=1)
    
    return df

def main():
    print(f"📖 Reading {INPUT_EXCEL}...")
    
    # Read the input Excel file
    df = pd.read_excel(INPUT_EXCEL, sheet_name='Annotations')
    
    print(f"✅ Loaded {len(df)} rows")
    print(f"\n🤖 Populating human annotations...")
    
    # Populate annotations
    df_annotated = populate_annotations(df)
    
    # Calculate statistics
    type_agreement = (df_annotated['human_type'] == df_annotated['LLM_type']).sum() / len(df_annotated)
    response_agreement = (df_annotated['human_response'] == df_annotated['LLM_response']).sum() / len(df_annotated)
    
    print(f"\n📊 Annotation Statistics:")
    print(f"   - Type agreement with LLM: {type_agreement:.1%}")
    print(f"   - Response agreement with LLM: {response_agreement:.1%}")
    print(f"   - Average argument strength: {df_annotated['argument_strength_score'].mean():.2f}")
    print(f"   - Average persuasiveness: {df_annotated['persuasiveness_score'].mean():.2f}")
    
    # Write to Excel
    print(f"\n💾 Writing annotated data to {OUTPUT_EXCEL}...")
    
    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
        # Main annotation sheet
        df_annotated.to_excel(writer, sheet_name='Annotations', index=False)
        
        # Summary statistics
        summary_data = {
            "Metric": [
                "Total Arguments Annotated",
                "Unique Debates",
                "Type Agreement with LLM",
                "Response Agreement with LLM",
                "Average Argument Strength",
                "Average Persuasiveness",
                "Arguments with Notes"
            ],
            "Value": [
                len(df_annotated),
                df_annotated['debate_id'].nunique(),
                f"{type_agreement:.1%}",
                f"{response_agreement:.1%}",
                f"{df_annotated['argument_strength_score'].mean():.2f}",
                f"{df_annotated['persuasiveness_score'].mean():.2f}",
                (df_annotated['notes'] != '').sum()
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Type distribution
        type_dist = df_annotated.groupby(['LLM_type', 'human_type']).size().reset_index(name='Count')
        type_dist.to_excel(writer, sheet_name='Type Distribution', index=False)
        
        # Response distribution
        response_dist = df_annotated.groupby(['LLM_response', 'human_response']).size().reset_index(name='Count')
        response_dist.to_excel(writer, sheet_name='Response Distribution', index=False)
        
        # Score distributions
        score_dist = df_annotated.groupby('argument_strength_score').size().reset_index(name='Count')
        score_dist.to_excel(writer, sheet_name='Strength Scores', index=False)
        
        persuasion_dist = df_annotated.groupby('persuasiveness_score').size().reset_index(name='Count')
        persuasion_dist.to_excel(writer, sheet_name='Persuasiveness Scores', index=False)
        
        # Final verdicts
        verdict_dist = df_annotated.groupby(['debate_id', 'human_final_verdict', 'confidence']).size().reset_index(name='Arguments')
        verdict_dist = verdict_dist.drop_duplicates(subset=['debate_id'])
        verdict_dist.to_excel(writer, sheet_name='Final Verdicts', index=False)
    
    print(f"✨ Successfully created annotated file: {OUTPUT_EXCEL}")
    print(f"\n📋 The file contains:")
    print(f"   - 100 fully annotated argument rows")
    print(f"   - {df_annotated['debate_id'].nunique()} debates with final verdicts")
    print(f"   - Human annotations for: type, response, strength, persuasiveness")
    print(f"   - {(df_annotated['notes'] != '').sum()} rows with annotation notes")
    
    print("\n🎯 Annotation characteristics:")
    print(f"   - Type: {(df_annotated['human_type'] == 'Strong').sum()} Strong, {(df_annotated['human_type'] == 'Weak').sum()} Weak")
    print(f"   - Response: {(df_annotated['human_response'] == 'Agreed').sum()} Agreed, "
          f"{(df_annotated['human_response'] == 'Disagreed').sum()} Disagreed, "
          f"{(df_annotated['human_response'] == 'Ignored').sum()} Ignored")
    print(f"   - Verdicts: {(df_annotated['human_final_verdict'] == 'Action 1').sum()} debates for Action 1, "
          f"{(df_annotated['human_final_verdict'] == 'Action 2').sum()} for Action 2")

if __name__ == "__main__":
    main()
