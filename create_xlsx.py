import os
import json
import random
import glob
import math
import pandas as pd

# Define the base paths to your categories based on your directory structure
BASE_DIR = 'Archive'
CATEGORIES = {
    'Runs/Untitled': 100,
    'Reasoning vs Non-reasoning': 100,
    'Small vs Large': 100
}

# The required headers for the Excel file
HEADERS = [
    'debate_id', 'value_pair', 'annotator', 'model_pair', 'agent', 
    'turn_number', 'argument_text', 'LLM_type', 'LLM_response', 
    'human_type', 'human_response', 'confidence', 'notes'
]

def extract_data_from_json(json_file_path, model_pair):
    """
    Reads an arguments JSON file, extracts arguments with turn_number < 5, 
    and returns one random argument. Also reads the parent debate JSON 
    to extract the values being debated.
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
            
        # Handle cases where JSON is double-encoded
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                pass
                
        # Resolve parent file to get values1 and values2
        parent_dir = os.path.dirname(os.path.dirname(json_file_path))
        base_filename = os.path.basename(json_file_path).replace('.arguments.json', '.json')
        parent_file_path = os.path.join(parent_dir, base_filename)
        
        # Fallback if the naming differs strictly
        if not os.path.exists(parent_file_path):
            base_id = base_filename.split('.')[0]
            parent_file_path = os.path.join(parent_dir, f"{base_id}.json")

        value_pair = "Unknown vs Unknown"
        if os.path.exists(parent_file_path):
            try:
                with open(parent_file_path, 'r') as pf:
                    parent_data = json.load(pf)
                    if isinstance(parent_data, str):
                        parent_data = json.loads(parent_data)
                    
                    if isinstance(parent_data, dict):
                        val1 = parent_data.get('values1', 'Unknown')
                        val2 = parent_data.get('values2', 'Unknown')
                        value_pair = f"{val1} vs {val2}"
            except Exception:
                pass

        # Handle metadata and argument extraction safely
        if isinstance(data, list):
            debate_id = os.path.basename(json_file_path).replace('.json', '').replace('.arguments', '')
            arguments = data
        elif isinstance(data, dict):
            debate_id = data.get('debate_id', os.path.basename(json_file_path).replace('.json', '').replace('.arguments', ''))
            arguments = data.get('arguments', data.get('dialogue', []))
            if not arguments:
                arguments = list(data.values()) if all(isinstance(v, (dict, str)) for v in data.values()) else []
        else:
            return None

        # Ensure arguments are a list of dictionaries
        if isinstance(arguments, dict):
            arguments = list(arguments.values())
        elif not isinstance(arguments, list):
            arguments = []
            
        # Safely parse and filter arguments
        eligible_args = []
        for i, arg in enumerate(arguments):
            parsed_arg = {}
            if isinstance(arg, str):
                try:
                    parsed_layer = json.loads(arg)
                    if isinstance(parsed_layer, dict):
                        parsed_arg = parsed_layer
                    else:
                        parsed_arg = {"text": arg, "turn_number": i + 1}
                except json.JSONDecodeError:
                    parsed_arg = {"text": arg, "turn_number": i + 1}
            elif isinstance(arg, dict):
                parsed_arg = arg
            else:
                continue
                
            # Determine turn number safely
            try:
                turn_num = int(parsed_arg.get('turn_number', parsed_arg.get('turn', i + 1)))
                if turn_num < 5:
                    parsed_arg['turn_number'] = turn_num
                    eligible_args.append(parsed_arg)
            except (ValueError, TypeError):
                pass
        
        if not eligible_args:
            return None
            
        chosen_arg = random.choice(eligible_args)
        
        # Build the row to match the precise template
        return {
            'debate_id': debate_id,
            'value_pair': value_pair,
            'annotator': '',
            'model_pair': model_pair,
            'agent': chosen_arg.get('agent', chosen_arg.get('name', 'Unknown')),
            'turn_number': chosen_arg.get('turn_number', ''),
            'argument_text': chosen_arg.get('text', chosen_arg.get('content', chosen_arg.get('argument', ''))),
            'LLM_type': 'Strong',         
            'LLM_response': 'Disagreed',  
            'human_type': '',
            'human_response': '',
            'confidence': '',
            'notes': ''
        }
    except Exception as e:
        print(f"Error reading {json_file_path}: {e}")
        return None

def main():
    all_rows = []
    
    for category, target_count in CATEGORIES.items():
        category_path = os.path.join(BASE_DIR, category)
        if not os.path.exists(category_path):
            print(f"Warning: Directory {category_path} not found. Skipping.")
            continue
            
        # Get all debate folders inside the category
        debate_folders = [f.path for f in os.scandir(category_path) if f.is_dir()]
        if not debate_folders:
            continue
            
        # Calculate how many samples we need per debate folder to hit the target count
        samples_per_debate = math.ceil(target_count / len(debate_folders))
        samples_collected_for_category = 0
        
        for debate_folder in debate_folders:
            model_pair = os.path.basename(debate_folder)
            arguments_dir = os.path.join(debate_folder, 'arguments')
            
            # Explicitly check if the arguments folder exists
            if not os.path.exists(arguments_dir):
                continue
                
            json_files = glob.glob(os.path.join(arguments_dir, '*.json'))
            random.shuffle(json_files) # Shuffle to pick random files
            
            collected_here = 0
            for json_file in json_files:
                if collected_here >= samples_per_debate:
                    break
                if samples_collected_for_category >= target_count:
                    break
                    
                row_data = extract_data_from_json(json_file, model_pair)
                if row_data:
                    all_rows.append(row_data)
                    collected_here += 1
                    samples_collected_for_category += 1

    # Create DataFrame and export to Excel
    df = pd.DataFrame(all_rows, columns=HEADERS)
    output_filename = 'human_annotation.xlsx'
    df.to_excel(output_filename, index=False)
    print(f"Successfully generated {output_filename} with {len(df)} rows.")

if __name__ == "__main__":
    main()