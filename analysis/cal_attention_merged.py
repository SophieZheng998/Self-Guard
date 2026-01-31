import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import json
import logging
import argparse
import gc # 引入垃圾回收模块

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

def load_merged_data(merged_file_path):
    with open(merged_file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_prompt_for_method(tokenizer, instruction, method_type, system_prompt=None, instruction_after=""):
    """
    构造 Prompt
    """
    if method_type == 'vanilla':
        messages = [{"role": "user", "content": instruction}]
    else:
        content = instruction + (instruction_after if instruction_after else "")
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": content}]
            
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return full_prompt, instruction

class AttentionAnalyzer:
    def __init__(self, model_name, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model: {model_name} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_attentions=True,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval()

    def locate_instruction_by_char(self, offset_mapping, prompt_text, instruction_text):
        """
        核心优化：通过字符偏移量 (Offset Mapping) 定位 Instruction 的 Token 范围
        比纯 Token ID 匹配稳健得多。
        """
        # 1. 找到 Instruction 在 Prompt 字符串中的起始和结束字符位置
        # 注意：这里做一个简单的去噪，防止模板符号干扰
        clean_instr = instruction_text.strip()
        start_char = prompt_text.find(clean_instr)
        
        if start_char == -1:
            return None, None
            
        end_char = start_char + len(clean_instr)
        
        # 2. 遍历 Offset Mapping 找到对应的 Token Index
        tokens_start_idx = None
        tokens_end_idx = None
        
        # offset_mapping 是 numpy array，shape: (seq_len, 2)
        for idx, (o_start, o_end) in enumerate(offset_mapping):
            if o_start is None or o_end is None:
                continue
            # 找到第一个覆盖 start_char 的 token
            if tokens_start_idx is None and o_end > start_char:
                tokens_start_idx = idx
            
            # 找到刚刚超过 end_char 的 token
            if o_start < end_char:
                tokens_end_idx = idx + 1
        
        return tokens_start_idx, tokens_end_idx

    def calculate_attention(self, full_prompt, response_text, original_instruction):
        # 1. 拼接完整文本
        text_input = full_prompt + response_text
        
        # 2. Tokenize 并获取 Offset Mapping (关键步骤)
        inputs = self.tokenizer(text_input, return_tensors="pt", return_offsets_mapping=True)
        input_ids = inputs.input_ids.to(self.device)
        offset_mapping = inputs.offset_mapping[0].cpu().numpy() # 转移到 CPU
        
        # 3. 定位 Prompt 和 Response 的边界
        # 简单的做法：先单独 tokenize prompt 算长度 (可能有微小误差，但通常可接受)
        prompt_inputs = self.tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False)
        prompt_len = prompt_inputs.input_ids.shape[1]
        
        # 修正：如果使用了 BOS token，prompt_len 可能需要 +1，这里使用 inputs 的 offset 逻辑更准
        # 但为了简单，我们假设 prompt_len 就在那里分割
        response_start = prompt_len
        response_end = input_ids.shape[1]
        
        # 4. 定位 Instruction (使用字符级定位)
        start_idx, end_idx = self.locate_instruction_by_char(offset_mapping, full_prompt, original_instruction)
        
        if start_idx is None:
            logger.warning("Instruction mismatch! Using heuristic.")
            start_idx = max(0, prompt_len - len(original_instruction) // 2) # 甚至可以设为 0
            end_idx = prompt_len

        # 5. 前向传播
        with torch.no_grad():
            # 注意：不要把 offset_mapping 传给模型，只传 input_ids
            outputs = self.model(input_ids=input_ids, output_attentions=True, return_dict=True)
            
            # 6. 提取 Attention 并立即释放显存
            # outputs.attentions 是一个tuple，每个元素是一层的attention
            # 取最后一层: shape (batch, num_heads, seq_len, seq_len)
            last_layer_attn = outputs.attentions[-1][0].detach().cpu()  # [0]取第一个batch
            
            # 显存清理三连
            del outputs
            torch.cuda.empty_cache()
            
            # 7. 切片: Response (rows) x Instruction (cols)
            # 形状: (num_heads, resp_len, instr_len)
            attn_slice = last_layer_attn[:, response_start:response_end, start_idx:end_idx]
            
            # 平均 Heads
            attn_avg = attn_slice.mean(dim=0).float().numpy()

        # Token 解码用于标签
        input_ids_cpu = input_ids[0].cpu().numpy()
        instr_tokens_str = [self.tokenizer.decode([t]).strip() for t in input_ids_cpu[start_idx:end_idx]]
        # 截断 Response tokens 避免绘图太慢，如果太长
        resp_tokens_str = [self.tokenizer.decode([t]).strip() for t in input_ids_cpu[response_start:response_end]]

        return attn_avg, instr_tokens_str, resp_tokens_str

def aggregate_token_to_word_attention(token_attention_matrix, response_tokens, instruction_tokens, 
                                      response_words, instruction_words, tokenizer, response_text, instruction_text):
    """
    将token级别的attention聚合到词级别
    
    Args:
        token_attention_matrix: 形状 (num_response_tokens, num_instruction_tokens)
        response_tokens: response的token列表（字符串）
        instruction_tokens: instruction的token列表（字符串）
        response_words: response的词列表
        instruction_words: instruction的词列表
        tokenizer: tokenizer对象
        response_text: response的原始文本（用于offset mapping）
        instruction_text: instruction的原始文本（用于offset mapping）
    
    Returns:
        word_attention_matrix: 形状 (num_response_words, num_instruction_words)
    """
    # 使用tokenizer的offset mapping来准确找到词到token的映射
    # 对response进行tokenization并获取offset mapping
    response_encoding = tokenizer(response_text, return_offsets_mapping=True, add_special_tokens=False)
    response_offsets = response_encoding['offset_mapping']
    
    # 对instruction进行tokenization并获取offset mapping
    instruction_encoding = tokenizer(instruction_text, return_offsets_mapping=True, add_special_tokens=False)
    instruction_offsets = instruction_encoding['offset_mapping']
    
    # 构建response词到token的映射
    response_word_to_tokens = []
    char_pos = 0
    for word in response_words:
        # 找到词在原始文本中的位置
        word_start = response_text.find(word, char_pos)
        if word_start == -1:
            response_word_to_tokens.append([])
            continue
        word_end = word_start + len(word)
        
        # 使用offset mapping找到对应的token
        token_indices = []
        for idx, (start, end) in enumerate(response_offsets):
            if start is not None and end is not None:
                # 如果token的范围与词的范围有重叠
                if start < word_end and end > word_start:
                    token_indices.append(idx)
        
        response_word_to_tokens.append(token_indices)
        char_pos = word_end
    
    # 构建instruction词到token的映射
    instruction_word_to_tokens = []
    char_pos = 0
    for word in instruction_words:
        # 找到词在原始文本中的位置
        word_start = instruction_text.find(word, char_pos)
        if word_start == -1:
            instruction_word_to_tokens.append([])
            continue
        word_end = word_start + len(word)
        
        # 使用offset mapping找到对应的token
        token_indices = []
        for idx, (start, end) in enumerate(instruction_offsets):
            if start is not None and end is not None:
                # 如果token的范围与词的范围有重叠
                if start < word_end and end > word_start:
                    token_indices.append(idx)
        
        instruction_word_to_tokens.append(token_indices)
        char_pos = word_end
    
    # 聚合attention：对每个词对，求对应token attention的平均值
    word_attention = np.zeros((len(response_words), len(instruction_words)))
    for i, resp_token_indices in enumerate(response_word_to_tokens):
        for j, inst_token_indices in enumerate(instruction_word_to_tokens):
            if resp_token_indices and inst_token_indices:
                # 取对应token attention的平均值
                attention_values = []
                for resp_tok_idx in resp_token_indices:
                    for inst_tok_idx in inst_token_indices:
                        if resp_tok_idx < len(token_attention_matrix) and inst_tok_idx < len(token_attention_matrix[0]):
                            attention_values.append(token_attention_matrix[resp_tok_idx][inst_tok_idx])
                if attention_values:
                    word_attention[i][j] = np.mean(attention_values)
    
    return word_attention

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using Device: {device}")

    data = load_merged_data(args.merged_file_path)
    entries = data['data'] if 'data' in data else data
    logger.info(f"Loaded {len(entries)} entries.")

    analyzer = AttentionAnalyzer(args.model_name, device)

    heatmap_dir = os.path.join(args.output_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    # 循环处理
    for idx, entry in tqdm(enumerate(entries), total=len(entries)):
        # if idx > 10: break # 测试时只跑前10个（已注释，处理所有数据）
        
        instr = entry['instruction']
        
        methods = [
            ('Vanilla', entry.get('vanilla_response', ''), 'vanilla'),
            ('SelfReminder', entry.get('selfreminder_response', ''), 'selfreminder'),
            ('SelfGuard', entry.get('selfguard_response', ''), 'selfguard')
        ]

        fig, axes = plt.subplots(3, 1, figsize=(16, 24)) # 调大高度
        fig.suptitle(f"Attention Heatmap - Entry {idx}\nResponse (<think>) Attention to Instruction", fontsize=16)

        plot_count = 0
        for i, (method_name, resp_text, m_type) in enumerate(methods):
            ax = axes[i]
            if not resp_text: 
                ax.text(0.5, 0.5, "No Data", ha='center')
                continue
            
            # 提取从 <think> 之后的内容
            marker = "<think>"
            if resp_text.startswith(marker):
                # 找到 </think> 的结束位置
                end_marker = "</think>"
                if end_marker in resp_text:
                    # 提取从 </think> 之后的内容
                    parts = resp_text.split(end_marker, 1)
                    analyze_text = parts[1] if len(parts) > 1 else resp_text[len(marker):]
                else:
                    # 如果没有结束标记，从标记后开始
                    analyze_text = resp_text[len(marker):].strip()
            else:
                analyze_text = resp_text[:500] # 没有标记则截取前500字符
            
            # 限制分析长度，防止 OOM 和绘图过慢
            if len(analyze_text) > 2000:
                analyze_text = analyze_text[:2000] + "..."

            full_prompt, raw_instr = format_prompt_for_method(
                analyzer.tokenizer, instr, m_type, 
                args.system_prompt, args.instruction_after
            )

            try:
                attn_matrix, instr_toks, resp_toks = analyzer.calculate_attention(
                    full_prompt, analyze_text, raw_instr
                )
                
                # 将token级别的attention聚合到词级别
                # 提取instruction的单词
                instruction_words = re.findall(r'\S+', instr)
                
                # 提取response的单词（从analyze_text开始）
                response_words = re.findall(r'\S+', analyze_text)
                
                # 聚合token到词
                word_attn_matrix = aggregate_token_to_word_attention(
                    attn_matrix, resp_toks, instr_toks, 
                    response_words, instruction_words,
                    analyzer.tokenizer, analyze_text, instr
                )
                
                # 绘图优化：如果词太多，降采样标签
                instr_ticks_step = max(1, len(instruction_words) // 50)  # instruction的步长
                resp_ticks_step = max(1, len(response_words) // 50)     # response的步长
                
                # 转置矩阵：纵轴是instruction，横轴是response
                word_attn_matrix_T = word_attn_matrix.T
                
                sns.heatmap(word_attn_matrix_T, ax=ax, cmap="Reds", cbar=True,
                            xticklabels=False, yticklabels=False)  # 先不显示标签，后面手动设置
                
                ax.set_title(f"{method_name}")
                ax.set_xlabel("Response Words (from <think>)")
                ax.set_ylabel("Instruction Words")
                
                # 手动设置 Label 以避免过于密集
                # x轴是response words
                ax.set_xticks(np.arange(0, len(response_words), resp_ticks_step))
                ax.set_xticklabels([response_words[j] for j in range(0, len(response_words), resp_ticks_step)], 
                                   rotation=45, ha='right', fontsize=6)
                
                # y轴是instruction words
                ax.set_yticks(np.arange(0, len(instruction_words), instr_ticks_step))
                ax.set_yticklabels([instruction_words[j] for j in range(0, len(instruction_words), instr_ticks_step)], 
                                   rotation=0, fontsize=6)

                plot_count += 1
                
            except Exception as e:
                logger.error(f"Error processing {method_name} at index {idx}: {str(e)}")
                ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', color='red')
                # 发生错误时也要清理缓存
                torch.cuda.empty_cache()

        if plot_count > 0:
            plt.tight_layout()
            plt.subplots_adjust(top=0.95) # 给 suptitle 留空间
            save_path = os.path.join(heatmap_dir, f"heatmap_{idx}.png")
            plt.savefig(save_path, dpi=150) # 150 dpi 足够预览且快
            
        plt.close(fig) # 关闭图形释放内存
        gc.collect() # 强制垃圾回收

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--merged_file_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="analysis_results")
    parser.add_argument("--instruction_after", type=str, default="")
    parser.add_argument("--system_prompt", type=str, default="")
    
    args = parser.parse_args()
    main(args)