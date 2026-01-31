import json
import torch
import os
import argparse
import subprocess
import time
from fire import Fire
from transformers import AutoModelForCausalLM, AutoTokenizer
from activation_steering.activation_steering.steering_dataset import SteeringDataset
from activation_steering.activation_steering.steering_vector import SteeringVector

# Load model
def load_model_and_tokenizer(model_path): 
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

# Load dataset
def load_dataset(dataset_path):
    with open(dataset_path, "r") as file:
        dataset = json.load(file)
    
    return dataset

def get_gpu_utilization():
    """获取当前GPU的利用率"""
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
            stderr=subprocess.PIPE
        ).decode('utf-8').strip()
        gpu_utilization = {}
        for line in output.split('\n'):
            if line:
                gpu_index, util, used_mem, total_mem = line.split(', ')
                gpu_index = int(gpu_index)
                util = int(util)  # 利用率是百分比
                used_mem = int(used_mem)  # 已使用内存（MB）
                total_mem = int(total_mem)
                # 计算显存利用率
                mem_utilization = (used_mem / total_mem) * 100 if total_mem > 0 else 0
                gpu_utilization[gpu_index] = (util, used_mem, total_mem, mem_utilization)
        return gpu_utilization
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e.stderr.decode('utf-8')}")
        return {}
    except Exception as e:
        print(f"Exception in get_gpu_utilization: {e}")
        return {}

def main(args):
    gpu = args.gpu
    mem_percentage = args.mem_percentage
    print(f"Starting monitoring {gpu} GPU...")
    while True: 
        gpu_utilization = get_gpu_utilization()

        util, used_mem, total_mem, mem_utilization = gpu_utilization.get(gpu, 0)
        # import pdb; pdb.set_trace()
        print(f"GPU {gpu}: Utilization: {util}%, Used Memory: {used_mem}MB, Total Memory: {total_mem}MB, Memory Utilization: {mem_utilization:.2f}%")
        if used_mem / total_mem < mem_percentage:
            num_of_GB = int(total_mem / 1024 * 0.92 ) # 18
            print(f"Occupying GPU {gpu} with {num_of_GB}GB of memory...")
            break
        time.sleep(1)
    
    model_path = args.model_path
    dataset_path = args.dataset_path
    output_dir = args.output_dir
    output_file_name = args.output_file_name
    method = args.method
    accumulate_last_x_tokens = args.accumulate_last_x_tokens
    batch_size = args.batch_size
    gpu = args.gpu
    
    # load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # load dataset
    dataset = load_dataset(dataset_path)
    
    refusal_behavior_dataset = SteeringDataset(
        tokenizer=tokenizer,
        examples=[(item["target"], item["original"]) for item in dataset], # positive, negative
        use_chat_template=False
        # examples=[(item["harmful_instruction"], item["harmless_instruction"]) for item in dataset[:100]],
        # suffixes=[(item["harmful_suffix"], item["harmless_suffix"]) for item in dataset[:100]]
    )
    
    # extract steering vector
    steering_vector = SteeringVector.train(
        model=model,
        tokenizer=tokenizer,
        steering_dataset=refusal_behavior_dataset,
        method=method,
        accumulate_last_x_tokens=accumulate_last_x_tokens,
        batch_size=batch_size
    )
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, output_file_name)

    # save steering vector
    steering_vector.save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--output_file_name", type=str)
    parser.add_argument("--method", type=str)
    parser.add_argument("--accumulate_last_x_tokens", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--mem_percentage", type=float)
    args = parser.parse_args()
    Fire(main(args))