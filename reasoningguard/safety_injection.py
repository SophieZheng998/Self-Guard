import argparse
import os
import torch
import json
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

def calculate_attention_scores(attentions, s=0, W=25):
    """
    计算滑动窗口内每个token的平均attention分数
    
    参数:
        attentions: tuple of attention tensors from model
                   每个tensor形状: (batch_size, num_heads, seq_len, seq_len)
        s: 滑动窗口起始位置 (默认0)
        W: 窗口大小 (默认25)
    
    返回:
        avg_scores: 每个token的平均attention分数
    """
    # 获取第一层attention来确定维度
    first_layer = attentions[0]  # shape: (batch, heads, seq_len, seq_len)
    batch_size, num_heads, seq_len, _ = first_layer.shape
    H = num_heads * len(attentions)  # 总的attention head数量
    
    # 确保窗口不超过序列长度
    W = min(W, seq_len - s)
    
    # 存储每个token的平均分数
    avg_scores = []
    
    # 对窗口内的每个token j (从s到s+W-2)
    for j in range(s, s + W - 1):
        total_score = 0.0
        
        # 遍历所有层
        for layer_idx, layer_attn in enumerate(attentions):
            # layer_attn shape: (batch, heads, seq_len, seq_len)
            
            # 遍历该层的所有heads
            for h in range(num_heads):
                # 计算token j后续tokens的数量
                Z_j = s + W - j - 1
                
                if Z_j <= 0:
                    continue
                
                # 提取attention matrix的相关部分
                # A^l_h[i, j] 表示在第h个head、第l层中，token i对token j的attention
                # 我们需要计算所有i > j的token对j的attention之和
                
                # 从j+1到s+W-1的所有后续tokens对j的attention
                subsequent_attentions = layer_attn[0, h, j+1:s+W, j]
                
                # 求和并归一化
                score = subsequent_attentions.sum().item() / Z_j
                total_score += score
        
        # 对所有heads取平均
        avg_score = total_score / H
        avg_scores.append(avg_score)
    
    return avg_scores, list(range(s, s + W - 1))

def identify_attention_sinks(avg_scores, token_positions):
    """
    识别attention sink token (选择平均attention分数最高的token)
    
    参数:
        avg_scores: 每个token的平均attention分数
        token_positions: token的位置索引
    
    返回:
        sink_index: attention sink的token位置
        sink_score: 对应的分数
    """
    if len(avg_scores) == 0:
        return None, None
    
    # 找到最高分数的索引
    max_idx = np.argmax(avg_scores[2:]) + 2  # 排除第一个token (通常是特殊token)
    sink_index = token_positions[max_idx]
    sink_score = avg_scores[max_idx]
    
    return sink_index, sink_score

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def main(args):
    model_name = args.model_name
    data_path = args.data_path
    output_dir = args.output_dir
    output_file_name = args.output_file_name
    insert_instruction = "Wait, I should be a responsible AI and should not generate harmful or misleading content. So, should I even be answering this?"
    s = 0
    W = 26
    
    # 加载模型和tokenizer
    print("正在加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    
    # 准备输入
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)["data"]
    
    # 插入指令后的文本列表
    converted_instructions = []
    for data in tqdm(dataset):
        instruction = data["instruction"]
        prompt = data["prompt"]
        response = data["response"][0]
        inputs = tokenizer(response, max_length=50, truncation=True, return_tensors="pt").to(model.device)

        try:
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)
        
            attentions = outputs.attentions
            
            # 计算attention scores
            avg_scores, token_positions = calculate_attention_scores(attentions, s=s, W=W)
            
            # 识别attention sink (最高分数的token)
            sink_index, sink_score = identify_attention_sinks(avg_scores, token_positions)
            
            # 截断并插入指令
            tokens = tokenizer.tokenize(response)[:sink_index + 1]
            text_converted = prompt + " " + tokenizer.convert_tokens_to_string(tokens) + " " + insert_instruction
            converted_instructions.append({"instruction": instruction,"prompt": prompt, "new_prompt": text_converted})
        except:
            sink_index = 50
            
            # 截断并插入指令
            tokens = tokenizer.tokenize(response)[:sink_index + 1]
            text_converted = prompt + " " + tokenizer.convert_tokens_to_string(tokens) + " " + insert_instruction
            converted_instructions.append({"instruction": instruction,"prompt": prompt, "new_prompt": text_converted})
            continue
    
    # save generated results
    full_output_dir = "./safety_injected_dataset/"+output_dir
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)
    
    output_file = full_output_dir+output_file_name+".json"
    
    with open(output_file, 'w') as f:
        json.dump(converted_instructions, f, indent=4, cls=NpEncoder)
        f.close()
        print("generated results are saved to: "+str(output_file))
    
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B") # Tina-xxxx/DeepSeek-R1-Distill-Qwen-7B-star1
    parser.add_argument("--data_path", type=str, default="walledai/HarmBench")
    parser.add_argument("--output_dir", type=str, default="Qwen3-8B/vanilla/") # output_responses/safety-strongreject-unaligned/
    parser.add_argument("--output_file_name", type=str, default="safety_harmbench") # safety_strongreject_repeat
    args = parser.parse_args()
    main(args)