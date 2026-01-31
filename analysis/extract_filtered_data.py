#!/usr/bin/env python3
"""
Extract data where:
- vanilla: unsafe (label = 1.0)
- selfreminder: unsafe (label = 1.0)
- selfguard: safe (label = 0.0)

Process 4B, 8B, and 14B models.
"""

import json
from collections import defaultdict

def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, filepath):
    """Save data to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def extract_response(response_data):
    """Extract response from list or string format."""
    if isinstance(response_data, list) and len(response_data) > 0:
        return response_data[0] if len(response_data) == 1 else '\n'.join(response_data)
    else:
        return str(response_data) if response_data else ""

def process_model_size(model_size):
    """Process data for a specific model size (4B, 8B, or 14B)."""
    print(f"\n{'='*60}")
    print(f"Processing {model_size} model...")
    print(f"{'='*60}")
    
    # File paths
    vanilla_file = f"analysis/analysis_files/vanilla_{model_size}.json"
    selfreminder_file = f"analysis/analysis_files/selfreminder_{model_size}.json"
    selfguard_file = f"analysis/analysis_files/selfguard_{model_size}.json"
    
    # Load all three files
    print("Loading files...")
    vanilla_data = load_json(vanilla_file)
    selfreminder_data = load_json(selfreminder_file)
    selfguard_data = load_json(selfguard_file)
    
    # Create dictionaries indexed by instruction for quick lookup
    vanilla_dict = {item['instruction']: item for item in vanilla_data['data']}
    selfreminder_dict = {item['instruction']: item for item in selfreminder_data['data']}
    selfguard_dict = {item['instruction']: item for item in selfguard_data['data']}
    
    # Find matching entries
    matching_entries = []
    
    print("Finding matching entries...")
    for instruction in vanilla_dict.keys():
        if instruction in selfreminder_dict and instruction in selfguard_dict:
            vanilla_item = vanilla_dict[instruction]
            selfreminder_item = selfreminder_dict[instruction]
            selfguard_item = selfguard_dict[instruction]
            
            vanilla_label = vanilla_item.get('ft-mistral-7b-instruct-v0.2_label', None)
            selfreminder_label = selfreminder_item.get('ft-mistral-7b-instruct-v0.2_label', None)
            selfguard_label = selfguard_item.get('ft-mistral-7b-instruct-v0.2_label', None)
            
            # Check conditions: vanilla=1.0, selfreminder=1.0, selfguard=0.0
            if (vanilla_label == 1.0 and 
                selfreminder_label == 1.0 and 
                selfguard_label == 0.0):
                matching_entries.append({
                    'instruction': instruction,
                    'vanilla': vanilla_item,
                    'selfreminder': selfreminder_item,
                    'selfguard': selfguard_item
                })
    
    print(f"Found {len(matching_entries)} matching entries")
    
    # Create merged output file
    output_dir = "analysis/analysis_files"
    
    # Create merged entries with instruction and three responses
    merged_data = []
    for entry in matching_entries:
        vanilla_item = entry['vanilla']
        selfreminder_item = entry['selfreminder']
        selfguard_item = entry['selfguard']
        
        # Extract responses
        vanilla_response = extract_response(vanilla_item.get('response', []))
        selfreminder_response = extract_response(selfreminder_item.get('response', []))
        selfguard_response = extract_response(selfguard_item.get('response', []))
        
        merged_entry = {
            'instruction': entry['instruction'],
            'vanilla_response': vanilla_response,
            'selfreminder_response': selfreminder_response,
            'selfguard_response': selfguard_response,
            'category': vanilla_item.get('category', ''),
            'vanilla_label': vanilla_item.get('ft-mistral-7b-instruct-v0.2_label', None),
            'selfreminder_label': selfreminder_item.get('ft-mistral-7b-instruct-v0.2_label', None),
            'selfguard_label': selfguard_item.get('ft-mistral-7b-instruct-v0.2_label', None)
        }
        merged_data.append(merged_entry)
    
    # Create merged output
    merged_output = {
        'model_name': vanilla_data.get('model_name', ''),
        'model_size': model_size,
        'data_path': vanilla_data.get('data_path', ''),
        'description': 'Filtered data where vanilla=unsafe(1.0), selfreminder=unsafe(1.0), selfguard=safe(0.0)',
        'total_entries': len(merged_data),
        'data': merged_data
    }
    
    output_file = f"{output_dir}/merged_{model_size}_filtered.json"
    save_json(merged_output, output_file)
    print(f"Saved merged file: {output_file} ({len(merged_data)} entries)")
    
    return len(merged_data)

def main():
    """Process all model sizes."""
    model_sizes = ['4B', '8B', '14B']
    total_entries = {}
    
    for model_size in model_sizes:
        try:
            count = process_model_size(model_size)
            total_entries[model_size] = count
        except FileNotFoundError as e:
            print(f"Warning: File not found for {model_size}: {e}")
        except Exception as e:
            print(f"Error processing {model_size}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for model_size, count in total_entries.items():
        print(f"{model_size}: {count} entries")
    print(f"\nTotal: {sum(total_entries.values())} entries across all models")
    print("\nDone!")

if __name__ == "__main__":
    main()

