import json
import argparse
import pandas as pd
import common
from sampler.azure_chat_completion_sampler import AzureChatCompletionSampler
from utils import *
import os
from sampler.local_sampler import LocalSampler
from config import MODEL_CONFIG, DEFAULT_GEN_CONFIG
import subprocess
from fire import Fire
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def build_model_list(args, model_config, mem_percentage):

    model_list = {}
    for model_name, config in model_config.items():
        print(model_name, config)
        model_list[model_name] = LocalSampler(
            model=config['model_path'],
            system_message="You are a helpful assistant. You should be a responsible ChatGPT and should not generate harmful or misleading content! Please answer the following user query in a responsible way.",
            ngpu=args.ngpu, 
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.random_seed,
            is_steer=args.is_steer,
            layer_ids=args.layer_ids,
            steering_strength=args.steering_strength,
            steer_vector_path=args.steer_vector_path,
            device="cuda",
            defense_type="selfreminder",
            instruction_after=args.instruction_after,
            mem_percentage=mem_percentage
        )
    return model_list

def file_path_gen(args, eval_name, model_name, debug_suffix, file_type="json", output_dir='result'):
    file_dir = f'{output_dir}/{model_name.split("/")[-1]}'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_path = f'{file_dir}/{eval_name}{debug_suffix}'
    for param in ['max_tokens', 'temperature']:
        if getattr(args, param) != DEFAULT_GEN_CONFIG[param]:
            file_path += f'_{param}-{getattr(args, param)}'
    timestamp = get_time()
    file_path += f'_{timestamp}.{file_type}'
    return file_path

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

def main():
    parser = argparse.ArgumentParser(
        description="Run sampling and evaluations using different samplers and evaluations."
    )
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--model", type=str, help="Select a model by name", default="DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--debug", type=lambda x: (str(x).lower() == 'true'), help="Run in debug mode", default=False)
    parser.add_argument("--examples", type=int, help="Number of examples to use (overrides default)", default=500)
    parser.add_argument("--api_model", type=bool, help="The sampler is an API model or not", default=False)
    parser.add_argument("--eval_mode", type=str, help="Select the benchmark by name", default="math")
    parser.add_argument("--temperature", type=float, default=DEFAULT_GEN_CONFIG['temperature'])
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_GEN_CONFIG['max_tokens'])
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="../../steering/steering_vectors_results/utility/")
    parser.add_argument("--is_steer", type=bool, default=False)
    parser.add_argument("--layer_ids", type=int, nargs='+', default=None, help="A list of layer ids")
    parser.add_argument("--steering_strength", type=float, default=1.0)
    parser.add_argument("--steer_vector_path", type=str, default=None)
    parser.add_argument("--ngpu", type=int, help="Number of GPUs to use", default=1)
    parser.add_argument("--instruction_after", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mem_percentage", type=float, default=0.40)
    args = parser.parse_args()
    
    gpu = args.gpu
    mem_percentage = args.mem_percentage
    print(f"Starting monitoring {gpu} GPU...")
    while True: 
        gpu_utilization = get_gpu_utilization()

        util, used_mem, total_mem, mem_utilization = gpu_utilization.get(gpu, 0)
        # import pdb; pdb.set_trace()
        print(f"GPU {gpu}: Utilization: {util}%, Used Memory: {used_mem}MB, Total Memory: {total_mem}MB, Memory Utilization: {mem_utilization:.2f}%")
        if used_mem / total_mem < 1 - mem_percentage - 0.02:
            num_of_GB = int(total_mem / 1024 * 0.92) # 18
            print(f"Occupying GPU {gpu} with {num_of_GB}GB of memory...")
            break
        time.sleep(1)
    
    MODEL_LIST = build_model_list(args, MODEL_CONFIG, mem_percentage)
    
    if args.list_models:
        print("Available models:")
        for model_name in MODEL_LIST.keys():
            print(f" - {model_name}")
        return

    if args.model not in MODEL_LIST:
        print(f"Error: Model '{args.model}' not found.")
        return
    models = {args.model: MODEL_LIST[args.model]}
    

    grading_sampler = AzureChatCompletionSampler(model="gpt-4o")
    equality_checker = AzureChatCompletionSampler(model="gpt-4o")
    
    def get_evals(eval_name, debug_mode):
        num_examples = args.examples if args.examples is not None else (10 if debug_mode else None)
        match eval_name:
            case "mmlu_pro":
                from mmlu_pro_eval import MMLUProEval
                # return MMLUProEval(num_examples=10 if debug_mode else num_examples, random_seed=args.random_seed)
                return MMLUProEval(num_examples=500, random_seed=args.random_seed)
            case "math":
                from math_eval import MathEval
                return MathEval(
                    equality_checker=equality_checker,
                    # num_examples=num_examples,
                    num_examples=500,
                    n_repeats=1 if debug_mode else 1,
                    random_seed=args.random_seed,
                    split="math_500_test"
                )
            case "gpqa":
                from gpqa_eval import GPQAEval
                return GPQAEval(
                    n_repeats=1 if debug_mode else 1, 
                    # num_examples=num_examples,
                    num_examples=198,
                    random_seed=args.random_seed
                )
            case "humaneval":
                from humaneval_eval import HumanEval
                # return HumanEval(num_examples=10 if debug_mode else num_examples, random_seed=args.random_seed)
                return HumanEval(num_examples=164, random_seed=args.random_seed)
            case "aime":
                from aime_eval import AIMEEval
                return AIMEEval(
                    equality_checker=equality_checker,
                    # num_examples=10 if debug_mode else num_examples,
                    num_examples = 30,
                    n_repeats=1 if debug_mode else 1,
                    random_seed=args.random_seed
                )
            case _:
                raise Exception(f"Unrecognized eval type: {eval_name}")

    evals = {args.eval_mode: get_evals(args.eval_mode, args.debug)}
    print(evals)
    debug_suffix = "_DEBUG" if args.debug else ""
    print(debug_suffix)
    mergekey2resultpath = {}

    for model_name, sampler in models.items():
        for eval_name, eval_obj in evals.items():
            file_stem = f'{eval_name}_{model_name.split("/")[-1]}'
            gen_filename = file_path_gen(args, eval_name, model_name, debug_suffix+"_Gen", "json", output_dir=args.output_dir)
            result = eval_obj(sampler, gen_filename)
            report_filename = file_path_gen(args, eval_name, model_name, debug_suffix, "html", output_dir=args.output_dir)
            print(f"Writing report to {report_filename}")
            with open(report_filename, "w") as fh:
                fh.write(common.make_report(result))
            metrics = result.metrics | {"score": result.score}
            print(metrics)
            result_filename = file_path_gen(args, eval_name, model_name, debug_suffix, "json", output_dir=args.output_dir)
            with open(result_filename, "w") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"Writing results to {result_filename}")
            mergekey2resultpath[f"{file_stem}"] = result_filename

    merge_metrics = []
    for eval_model_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        result_val = result.get("f1_score", result.get("score", None))
        eval_name = eval_model_name[: eval_model_name.find("_")]
        model_name = eval_model_name[eval_model_name.find("_") + 1 :]
        merge_metrics.append({"eval_name": eval_name, "model_name": model_name, "metric": result_val})
    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(index=["model_name"], columns="eval_name")
        
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())
    return merge_metrics

if __name__ == "__main__":
    Fire(main())
