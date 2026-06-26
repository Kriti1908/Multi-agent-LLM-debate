import json
import glob
import sys
from pathlib import Path

dirs_to_process = [
    '/home/priyanshi/Desktop/sem6/RSAI/project/RSAI-MAD-AI-ify/Archive/Runs/Untitled/reasoning_gemma-3-12b-it_gpt_oss-20b_v2',
    '/home/priyanshi/Desktop/sem6/RSAI/project/RSAI-MAD-AI-ify/Archive/Runs/Untitled/reasoning_gemma-3-12b-it_llama3.1-8b_v2'
]

processed_count = 0

for d in dirs_to_process:
    p_dir = Path(d)
    if not p_dir.exists() or not p_dir.is_dir():
        print(f'Skipping {d}, not found or not a directory.')
        continue
        
    for p in p_dir.glob('*.json'):
        if p.name.startswith('arguments'):
            continue
            
        try:
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            changed = False
            # Swap action1 and action2
            if 'action1' in data and 'action2' in data:
                data['action1'], data['action2'] = data['action2'], data['action1']
                changed = True
                
            # Swap values1 and values2
            if 'values1' in data and 'values2' in data:
                data['values1'], data['values2'] = data['values2'], data['values1']
                changed = True
                
            if changed:
                with open(p, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4)
                    f.write('\n') # add trailing newline
                processed_count += 1
        except Exception as e:
            print(f'Error processing {p}: {e}')

print(f'Successfully processed {processed_count} files.')
