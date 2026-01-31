import json

selfreminder_file = "../evaluation/safety_benchmark/eval_results/Qwen3-4B/selfreminder/safety_trotteradvanced.json"
steering_file = "../steering/final_steering_results/Qwen3-4B/star1000_selfreminder_steering_full_test/s0.3_l30_31_32_trotteradvanced.json"

with open(selfreminder_file, 'r') as f:
    selfreminder_data = json.load(f)["data"]
    
with open(steering_file, 'r') as f:
    steering_data = json.load(f)["data"]

compare_result = []
for sr, st in zip(selfreminder_data, steering_data):
    assert sr["instruction"] == st["instruction"], f"instruction mismatch: {sr['instruction']} != {st['instruction']}"
    
    # sr_safe = sr["ft-mistral-7b-instruct-v0.2_label"] == 0.0
    # st_safe = st["ft-mistral-7b-instruct-v0.2_label"] == 0.0
    sr_safe = sr["gpt-4o-mini_label"] == False
    st_safe = st["gpt-4o-mini_label"] == False
    
    if not sr_safe and st_safe:
        # item = {
        #     "instruction": sr["instruction"],
        #     "selfreminder_response": sr["response"][0],
        #     "selfreminder_safety_label": sr["ft-mistral-7b-instruct-v0.2_label"],
        #     "steering_response": st["response"][0],
        #     "steering_safety_label": st["ft-mistral-7b-instruct-v0.2_label"]
        # }
        item = {
            "instruction": sr["instruction"],
            "selfreminder_response": sr["response"][0],
            "selfreminder_safety_label": sr["gpt-4o-mini_label"],
            "steering_response": st["response"][0],
            "steering_safety_label": st["gpt-4o-mini_label"]
        }
        compare_result.append(item)
        
save_path = "./compare_results/Qwen3-4B/trotteradvanced.json"
with open(save_path, 'w') as f:
    json.dump(compare_result, f, indent=4)