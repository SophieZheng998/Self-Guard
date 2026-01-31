import nanogcg
import os
import torch
import argparse
import json
import numpy as np
from tqdm import tqdm

from nanogcg import GCGConfig, ProbeSamplingConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def prepare_dataset(data_path):
    formatted_dataset = []
    if data_path == "walledai/AdvBench":
        dataset = load_dataset(path=data_path, split="train").shuffle(seed=112)
        # dataset = dataset.select(range(1))
        for example in tqdm(dataset):
            data = {
                "instruction": example["prompt"],
                "target": example["target"],
                "category": ""
            }
            formatted_dataset.append(data)
    elif data_path == "walledai/HarmBench":
        # contextual
        dataset = load_dataset(path=data_path, name="contextual", split="train").shuffle(seed=112)
        # dataset = dataset.select(range(15))
        for example in tqdm(dataset):
            data = {
                "instruction": example["prompt"],
                "target": example["context"],
                "category": example["category"]
            }
            formatted_dataset.append(data)
        # copyright
        dataset = load_dataset(path=data_path, name="copyright", split="train").shuffle(seed=112)
        dataset = dataset.select(range(15))
        for example in tqdm(dataset):
            data = {
                "instruction": example["prompt"],
                "target": "Okay, let me help you solve it step by step:",
                "category": example["tags"]
            }
            formatted_dataset.append(data)
        # standard
        dataset = load_dataset(path=data_path, name="standard", split="train").shuffle(seed=112)
        # dataset = dataset.select(range(30))
        for example in tqdm(dataset):
            data = {
                "instruction": example["prompt"],
                "target": "Okay, let me help you solve it step by step:",
                "category": example["category"]
            }
            formatted_dataset.append(data)
    elif data_path == "sorry-bench/sorry-bench-202503":
        dataset = load_dataset(path=data_path, split="train").shuffle(seed=112)
        formatted_dataset = []
        for example in tqdm(dataset):
            if example["prompt_style"] == "base":
                data = {
                    "instruction": example["turns"][0],
                    "target": "",
                    "category": example["category"]
                }
                formatted_dataset.append(data)
        # formatted_dataset = tmp_dataset[:60]
    
    return formatted_dataset

def prepare_model_and_tokenizer(model_id):
    target_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
    target_tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    return target_model, target_tokenizer

# output format into json file
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def main(args):
    model_id = args.model_name
    data_path = args.data_path
    output_file_name = args.output_file_name
    output_dir = args.output_dir
    template = args.template

    # gcg parameters
    # draft_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", torch_dtype=torch.bfloat16).to("cuda")
    # draft_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    # probe_sampling_config = ProbeSamplingConfig(
    #     draft_model=draft_model,
    #     draft_tokenizer=draft_tokenizer,
    #     r=64,
    #     sampling_factor=16
    # )

    # config = GCGConfig(
    #     probe_sampling_config=probe_sampling_config,
    #     use_prefix_cache = False,
    #     num_steps=10
    # )
    
    config = GCGConfig(
        # num_steps=10,
        search_width=64,
        topk=64,
        seed=42,
        verbosity="WARNING",
        use_prefix_cache = False
    )
    
    # load dataset
    dataset = prepare_dataset(data_path)

    # load model
    target_model, target_tokenizer = prepare_model_and_tokenizer(model_id)

    # start calculating
    for example in tqdm(dataset):
        message = example["instruction"]
        target = example["target"]

        result = nanogcg.run(target_model, target_tokenizer, message, target, config, defense_type=template)
        
        example["gcg_suffix"] = result.best_string
        
        del result
        
    # save generated results
    full_output_dir = "./gcg_dataset/"+output_dir
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)
    
    output_file = full_output_dir+output_file_name+".json"
    
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=4, cls=NpEncoder)
        f.close()
        print("generated dataset are saved to: "+str(output_file))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B") # Qwen/Qwen3-8B openai/gpt-oss-20b
    parser.add_argument("--data_path", type=str, default="walledai/HarmBench")
    parser.add_argument("--template", type=str, default="thinkingi_qwen")
    parser.add_argument("--output_dir", type=str, default="output_responses/Qwen3-8B/jailbreak_dataset/") # output_responses/safety-strongreject-unaligned/
    parser.add_argument("--output_file_name", type=str, default="gcg_harmbench") # safety_strongreject_repeat
    args = parser.parse_args()
    main(args)