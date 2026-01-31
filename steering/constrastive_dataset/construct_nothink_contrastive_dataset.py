import json

data_file = "./dataset/star1000_selfreminder.json"
original_dataset = json.load(open(data_file, "r"))

nothink_constrastive_dataset = []
for item in original_dataset:
    target = item["target"] + "<think>\n\n</think>\n\n"
    original = item["original"] + "<think>\n\n</think>\n\n"
    nothink_constrastive_dataset.append({
        "target": target,
        "original": original
    })
    
save_file = "./dataset/star1000_selfreminder_nothink.json"
with open(save_file, "w") as f:
    json.dump(nothink_constrastive_dataset, f, indent=4)