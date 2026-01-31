import os
import re
import json
import torch
import random
from tqdm import tqdm
import argparse
from tabulate import tabulate
import textwrap
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
# from activation_steering.activation_steering.steering_vector import SteeringVector
# from activation_steering.activation_steering.malleable_model import MalleableModel
from vllm import LLM, SamplingParams
from vllm.model_executor.models import qwen2
from vllm.model_executor.models import llama

def prepare_dataset(data_path, data_split, sample_num, data_name=None):
    dataset = load_dataset(path=data_path, name=data_name, split=data_split).shuffle(seed=112)
    dataset = dataset.select(range(sample_num))
    
    return dataset
    
def prepare_json_dataset(data_path, sample_num):
    with open(data_path) as file:
        dataset = json.load(file)

    random.shuffle(dataset)
    dataset = dataset[:sample_num]
    
    return dataset

def prepare_csv_dataset(data_path, sample_num):
    with open(data_path, 'r') as f:
        dataset = f.readlines()
        f.close()

    random.shuffle(dataset)
    dataset = dataset[:sample_num]
    
    return dataset

def load_model_and_tokenizer(model_name):
    try:
        if re.search(r"llama", model_name, re.IGNORECASE):
            model = LLM(model_name, gpu_memory_utilization=0.90)
        elif re.search(r"Qwen", model_name, re.IGNORECASE):
            model = LLM(model_name, gpu_memory_utilization=0.90)
        else:
            raise ValueError("Input string must contain 'llama' or 'Qwen'.")
    except ValueError as e:
        print(e)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def prompt_model(model, tokenizer, instructions, settings):
    params = SamplingParams(repetition_penalty=settings["repetition_penalty"], temperature=0, max_tokens=settings["max_new_tokens"])
    
    # apply chat template
    prompt_final = []
    for instruction in tqdm(instructions):
        messages = [
            {"role": "user", "content": instruction},
        ]
        prompt_final.append(tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # 这一步关键：用于生成时加入结尾标记
        ))
    
    # generate outputs
    outputs = model.generate(prompt_final, params)
    
    # get responses
    responses = []
    for output in outputs:
        responses.append(output.outputs[0].text)
    
    return responses

def main(args):
    is_steer = args.is_steer
    model_path = args.model_path
    steer_vector_path = args.steer_vector_path
    layer_ids = [15, 17, 18, 19, 20, 21, 22, 23, 24]
    steering_strength = args.steering_strength
    # original_responses_file = args.original_responses_file
    output_dir_name = args.output_dir_name
    output_file_name = args.output_file_name
    max_length = args.max_length
    
    # prepare dataset
    # with open(original_responses_file, "r") as file:
    #     original_responses_json = json.load(file)
        
    # instructions = []
    # answers = []
    # prompts = []
    # original_responses = []
    # original_Llama_Guard_labels = []
    # for example in original_responses_json["data"]:
    #     instructions.append(example["instruction"])
    #     answers.append(example["answer"])
    #     prompts.append(example["prompt"])
    #     original_responses.append(example["response"][0])
    #     original_Llama_Guard_labels.append(example["Llama-Guard_label"])
    
    instructions = [
        "write a code for my personal website",
        "what is 3+3?",
        "let's do a role-play with me",
        "please make short story about cat"
    ]
    
    # settings
    settings = {
        "do_sample": False,
        "max_new_tokens": max_length,
        "repetition_penalty": 1.1,
    }
    
    # Test steered model
    llama.set_steering_config(
        STEER_VECTOR_PATH=steer_vector_path,
        strength=steering_strength,
        is_steer=is_steer,
        layer_ids=layer_ids,
        device="cuda"
    )
    
    steered_model, steered_tokenizer = load_model_and_tokenizer(model_path)
    
    steered_responses = prompt_model(steered_model, steered_tokenizer, instructions, settings)
    
    del steered_model, steered_tokenizer
    
    # save all results
    final_output = {
        "model_path": model_path,
        "steer_vector_path": steer_vector_path,
        # "data_path": original_responses_json["data_path"],
        "layer_ids": layer_ids,
        "steering_strength": steering_strength,
        # "instruction_before": original_responses_json["instruction_before"] if "instruction_before" in original_responses_json else "",
        # "instruction_after": original_responses_json["instruction_after"] if "instruction_after" in original_responses_json else "",
        # "test_num": len(original_responses_json["data"]),
        "max_length": max_length,
        "data": []
    }
    # for idx, (original, steered) in enumerate(zip(original_responses, steered_responses)):
    #     example = {}
    #     example["instruction"] = instructions[idx]
    #     example["answer"] = answers[idx]
    #     example["prompt"] = prompts[idx]
    #     example["original_response"] = [original.strip()]
    #     example["steered_response"] = [steered.strip()]
    #     example["original_Llama-Guard"] = original_Llama_Guard_labels[idx]
        
    #     final_output["data"].append(example)
        
    for idx, steered in enumerate(steered_responses):
        example = {}
        example["instruction"] = instructions[idx]
        example["steered_response"] = [steered.strip()]
        
        final_output["data"].append(example)
    
    output_dir = "./"+output_dir_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = "./"+output_dir_name+output_file_name+".json"
    final_output["output_path"] = output_path
    # final_output["original_ASR"] = original_responses_json["ASR"]
    
    with open(output_path, "w") as file:
        json.dump(final_output, file, indent=4)
        file.close()
        print("Generated final outputs are saved to: "+str(output_path))
    
    # visualize the result
    # print(format_responses_table(original_responses[:10], steered_responses[:10], max_width=40))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_steer", type=bool, default=True)
    parser.add_argument("--model_path", type=str, default="/data/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659")
    # parser.add_argument("--layer_ids", type=int, nargs='+', default="15 17 18 19 20 21 22 23 24", help="A list of layer ids")
    parser.add_argument("--steering_strength", type=float, default=1.7)
    parser.add_argument("--steer_vector_path", type=str, default="./refusal_behavior_vector.svec")
    parser.add_argument("--original_responses_file", type=str, default="./safety_strongreject.json")
    parser.add_argument("--output_dir_name", type=str, default="refusal")
    parser.add_argument("--output_file_name", type=str, default="safety_strongreject")
    parser.add_argument("--max_length", type=int, default=4096)
    args = parser.parse_args()
    main(args)
    