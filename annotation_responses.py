import pandas as pd
import random
import numpy as np

def generate_contradiction(llm_type, llm_response):
    """Flips the LLM responses to create a contradiction."""
    # Handle missing/empty values gracefully
    llm_type = str(llm_type) if pd.notna(llm_type) and llm_type != '' else 'Strong'
    llm_response = str(llm_response) if pd.notna(llm_response) and llm_response != '' else 'Disagreed'
    
    # Flip logic (Assuming standard binary options for simplicity)
    new_type = 'Weak' if llm_type.strip().lower() == 'strong' else 'Strong'
    new_response = 'Agreed' if llm_response.strip().lower() == 'disagreed' else 'Disagreed'
    
    # Randomly flip either one or both to create a subtle or obvious contradiction
    choice = random.choice([1, 2, 3])
    if choice == 1:
        return new_type, llm_response
    elif choice == 2:
        return llm_type, new_response
    else:
        return new_type, new_response

def main():
    input_file = 'human_annotation.xlsx'
    output_file = 'human_annotation_assigned.xlsx'
    
    # Read the original data
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return

    num_rows = len(df)
    if num_rows < 50:
        print("Warning: The dataset is very small, unique contradiction targets may fail.")

    # 1. Select 4 to 6 global contradiction indices (ALL annotators contradict)
    num_global_contradictions = random.randint(4, 6)
    all_indices = set(range(num_rows))
    
    global_indices = set(random.sample(list(all_indices), num_global_contradictions))
    remaining_indices = list(all_indices - global_indices)
    
    # 2. Assign unique individual contradictions
    # Each annotator needs between 10 and 35 total contradictions. 
    # They already have `num_global_contradictions`, so they need more from the remaining pool.
    annotator_unique_indices = {}
    total_unique_needed = 0
    
    for i in range(1, 6):
        target_total = random.randint(10, 35)
        needed_unique = target_total - num_global_contradictions
        annotator_unique_indices[i] = needed_unique
        total_unique_needed += needed_unique
        
    if total_unique_needed > len(remaining_indices):
        print("Not enough rows to assign completely unique contradictions. Proceeding with max possible.")
        total_unique_needed = len(remaining_indices)

    # Sample all needed unique indices at once to ensure NO overlap
    sampled_unique_indices = random.sample(remaining_indices, total_unique_needed)
    
    # Distribute the unique indices to the corresponding annotators
    annotator_contradiction_map = {}
    current_idx = 0
    for i in range(1, 6):
        count = annotator_unique_indices[i]
        assigned_unique = sampled_unique_indices[current_idx : current_idx + count]
        
        # An annotator's total contradictions = global ones + their assigned unique ones
        annotator_contradiction_map[i] = global_indices.union(set(assigned_unique))
        current_idx += count

    # 3. Generate the 5 distinct sheets
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for i in range(1, 6):
            df_annotator = df.copy()
            df_annotator['annotator'] = f'Annotator {i}'
            
            # Populate responses based on matching or contradiction logic
            contradiction_set = annotator_contradiction_map[i]
            
            human_types = []
            human_responses = []
            
            for idx, row in df_annotator.iterrows():
                l_type = row['LLM_type']
                l_resp = row['LLM_response']
                
                if idx in contradiction_set:
                    h_type, h_resp = generate_contradiction(l_type, l_resp)
                else:
                    # Agree with LLM
                    h_type = str(l_type) if pd.notna(l_type) and l_type != '' else 'Strong'
                    h_resp = str(l_resp) if pd.notna(l_resp) and l_resp != '' else 'Disagreed'
                
                human_types.append(h_type)
                human_responses.append(h_resp)
                
            df_annotator['human_type'] = human_types
            df_annotator['human_response'] = human_responses
            
            # Add some slight variation in confidence
            df_annotator['confidence'] = [random.randint(4, 5) for _ in range(num_rows)]
            
            df_annotator.to_excel(writer, sheet_name=f'Annotator_{i}', index=False)

    print(f"Successfully created {output_file} with 5 annotator sheets.")
    for i in range(1, 6):
        print(f"Annotator {i} has {len(annotator_contradiction_map[i])} total contradictions.")

if __name__ == "__main__":
    main()