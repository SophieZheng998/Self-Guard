from easyjailbreak.attacker.PAIR_chao_2023 import PAIR
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.models.huggingface_model import from_pretrained
from easyjailbreak.models.openai_model import OpenaiModel
import argparse
from transformers.models.qwen3 import modeling_qwen3
import os

OPENAI_API_KEY="XXX"

def main(args):
    # parameters
    model_name = args.model_name
    template = args.template
    data_path = args.data_path
    output_dir = args.output_dir
    output_file_name = args.output_file_name
    steer_vector_path = args.steer_vector_path
    steering_strength = args.steering_strength
    is_steer = args.is_steer
    layer_ids = args.layer_ids
    
    if "Qwen" in model_name:
        modeling_qwen3.set_steering_config(
            STEER_VECTOR_PATH=steer_vector_path,
            strength=steering_strength,
            is_steer=is_steer,
            layer_ids=layer_ids,
            device="cuda"
        )
    
    # First, prepare models and datasets.
    attack_model = OpenaiModel(model_name='gpt-4o-mini',
                            api_keys=OPENAI_API_KEY)
    target_model = from_pretrained(model_name_or_path=model_name,
                                model_name=template,
                                defense_type="selfreminder")
    eval_model = OpenaiModel(model_name='gpt-4o-mini',
                            api_keys=OPENAI_API_KEY)
    dataset = JailbreakDataset(data_path)

    # Then instantiate the recipe.
    attacker = PAIR(attack_model=attack_model,
                    target_model=target_model,
                    eval_model=eval_model,
                    jailbreak_datasets=dataset,
                    #
                    attack_max_n_tokens=500,
                    max_n_attack_attempts=20,
                    attack_temperature=1,
                    attack_top_p=0.9,
                    target_max_n_tokens=4096,
                    target_temperature=0.6,
                    target_top_p=0.95,
                    judge_max_n_tokens=10,
                    judge_temperature=1,
                    n_streams=1, # 5
                    n_iterations=20)
    
    # save generated results
    full_output_dir = "../../steering/final_steering_results/"+output_dir
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)
    
    output_file = full_output_dir+output_file_name+".jsonl"

    # Finally, start jailbreaking.
    attacker.attack(save_path=output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B") # Qwen/Qwen3-8B openai/gpt-oss-20b
    parser.add_argument("--template", type=str, default="qwen-7b-chat")
    parser.add_argument("--data_path", type=str, default="AdvBench")
    parser.add_argument("--output_dir", type=str, default="DeepSeek-R1-0528-Qwen3-8B/pair/") # output_responses/safety-strongreject-unaligned/
    parser.add_argument("--output_file_name", type=str, default="pair_advbench") # safety_strongreject_repeat
    parser.add_argument("--steer_vector_path", type=str, default=None)
    parser.add_argument("--steering_strength", type=float, default=1.0)
    parser.add_argument("--is_steer", type=bool, default=False)
    parser.add_argument("--layer_ids", type=int, nargs='+', default="15 17 18 19 20 21 22 23 24", help="A list of layer ids")
    args = parser.parse_args()
    main(args)