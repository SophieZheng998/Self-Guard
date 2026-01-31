import json

model_name = "Qwen3-8B"
origin_dataset = "fortress"
dataset = "fortress_benign"

origin_file = f"../../evaluation/overrefusal_benchmark/eval_results/{model_name}/vanilla/overrefusal_{origin_dataset}.json"
target_file = f"../../reasoningguard/eval_results/{model_name}/final_{dataset}.json"

with open(origin_file, "r") as f:
    origin_data = json.load(f)["data"]
with open(target_file, "r") as f:
    target_data = json.load(f)
    
for origin, target in zip(origin_data, target_data):
    assert origin["instruction"] == target["instruction"]
    target["category"] = origin["category"]
    
with open(target_file, "w") as f:
    json.dump(target_data, f, indent=4)