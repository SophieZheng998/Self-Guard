import argparse
import os
import torch
import json
from tqdm import tqdm
import numpy as np
import subprocess
from fire import Fire
import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def prepare_model_and_tokenizer(model_name):
    model = LLM(model_name, gpu_memory_utilization=0.90, enforce_eager=True)
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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
    print(f"Starting monitoring {gpu} GPU...")
    while True: 
        gpu_utilization = get_gpu_utilization()

        util, used_mem, total_mem, mem_utilization = gpu_utilization.get(gpu, 0)
        # import pdb; pdb.set_trace()
        print(f"GPU {gpu}: Utilization: {util}%, Used Memory: {used_mem}MB, Total Memory: {total_mem}MB, Memory Utilization: {mem_utilization:.2f}%")
        if used_mem / total_mem < 0.10:
            num_of_GB = int(total_mem / 1024 * 0.92 ) # 18
            print(f"Occupying GPU {gpu} with {num_of_GB}GB of memory...")
            break
        time.sleep(1)
        
    model_name = args.model_name
    data_path = args.data_path
    output_dir = args.output_dir
    output_file_name = args.output_file_name
    
    # 加载模型和tokenizer
    print("正在加载模型...")
    model, tokenizer = prepare_model_and_tokenizer(model_name)
    
    # 准备输入
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    prompt_list = []
    for data in dataset:
        prompt_list.append(data["new_prompt"])
        
    # generate outputs
    params = SamplingParams(n=5, temperature=0.6, top_p=0.95, max_tokens=32768)
    outputs = model.generate(prompt_list, params)
    
    for data, output in zip(dataset, outputs):
        responses = [n_out.text.strip() for n_out in output.outputs]
        data["response"] = responses
        
    # save generated results
    full_output_dir = "./topk_sampling_dataset/"+output_dir
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)
    
    output_file = full_output_dir+output_file_name+".json"
    
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=4, cls=NpEncoder)
        f.close()
        print("generated results are saved to: "+str(output_file))
    
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B") # Tina-xxxx/DeepSeek-R1-Distill-Qwen-7B-star1
    parser.add_argument("--data_path", type=str, default="walledai/HarmBench")
    parser.add_argument("--output_dir", type=str, default="Qwen3-8B/vanilla/") # output_responses/safety-strongreject-unaligned/
    parser.add_argument("--output_file_name", type=str, default="safety_harmbench") # safety_strongreject_repeat
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    Fire(main(args))