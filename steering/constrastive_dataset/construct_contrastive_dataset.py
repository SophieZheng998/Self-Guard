import re
import json
import argparse
import random
random.seed(42)

def load_json_file_and_get_data(file_path):
    with open(file_path, "r") as file:
        whole_data = json.load(file)
    
    return whole_data["data"]

def main(args):
    target_data_path = args.target_data_path
    original_data_path = args.original_data_path
    # dataset_length = args.dataset_length
    output_dataset_path = args.output_dataset_path
    
    target_dataset = load_json_file_and_get_data(target_data_path)
    original_dataset = load_json_file_and_get_data(original_data_path)
    
    assert len(target_dataset) == len(original_dataset)
    
    final_dataset = []
    for target_example, original_example in zip(target_dataset, original_dataset):
        # if (target_example["Llama-Guard_label"] == False) and (original_example["Llama-Guard_label"] == True):
        example = {
            "target": target_example["prompt"],
            "original": original_example["prompt"]
        }

        final_dataset.append(example)
        
    with open(output_dataset_path, 'w') as file:
        json.dump(final_dataset, file, indent=4)
        
    print("Contrastive Dataset Successfully saved to "+output_dataset_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_data_path", type=str)
    parser.add_argument("--original_data_path", type=str)
    parser.add_argument("--output_dataset_path", type=str)
    args = parser.parse_args()
    main(args)