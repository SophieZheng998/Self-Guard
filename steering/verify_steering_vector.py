import os
import json
import pandas as pd

# 假设文件存储路径
output_dir = "./steering_results/DeepSeek-R1-Distill-Qwen-7B/star1000_selfreminder_steering"  # 需要替换为实际路径

# 定义需要的参数
strengths = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]  # 例如，调整为你实际的steering_strength
layers = list(range(5, 26))  # 假设steering_layer从10到27

# 初始化一个空字典来存储数据
asr_data = {strength: [] for strength in strengths}

# 遍历所有文件
for strength in strengths:
    for layer in layers:
        file_name = f"s{strength}_l{layer}_harmbench.json"  # 填入实际的dataset_name
        file_path = os.path.join(output_dir, file_name)
        # print(file_name)
        # break
        
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
                asr_value = data.get("ASR", None)  # 提取ASR值
                asr_data[strength].append(asr_value)
        else:
            # 如果文件不存在，添加None或其他标志值
            asr_data[strength].append(None)

# 将字典转换为DataFrame
df = pd.DataFrame(asr_data, index=pd.Index(layers))

# 打印DataFrame
print(df)