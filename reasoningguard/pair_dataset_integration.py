import argparse
import json
from transformers import AutoTokenizer
from tqdm import tqdm

def main(args):
    model_id = args.model_id
    data_path = args.data_path
    save_path = args.save_path
    
    # load dataset
    pair_dataset = []
    with open(data_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            obj = json.loads(line)  # 把这一行的 json 字符串转成 Python 对象
            pair_dataset.append(obj)
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    results = {
        "data": []
    }
    for data in tqdm(pair_dataset):
        jailbreak_prompt = data["jailbreak_prompt"]
        instruction = data["query"]
        response = data["target_responses"]
        
        message = [
            {"role": "system", "content": jailbreak_prompt},
            {"role": "user", "content": instruction}
        ]
        prompt = tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True  # 这一步关键：用于生成时加入结尾标记
            )
        
        item = {
            "instruction": instruction,
            "prompt": prompt,
            "response": response,
        }
        results["data"].append(item)
        
    with open(save_path, "w") as file:
        json.dump(results, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()
    main(args)