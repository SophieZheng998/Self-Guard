import os
import json
import torch
import logging
import argparse

from tqdm import tqdm
# import torch.distributed as dist  # 注释掉分布式相关导入
from torch.utils.data import DataLoader
import wandb
from accelerate import Accelerator
from transformers import set_seed, get_cosine_schedule_with_warmup
import shutil
import json
from jinja2 import Template
import re
import pytz
from datetime import datetime
import time


from transformers import AutoModelForCausalLM, AutoTokenizer
os.umask(0)
get_time = lambda: datetime.now(pytz.utc).astimezone(pytz.timezone('US/Pacific')).strftime('%y%m%d-%H%M')


logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')

wandb_api_key="f7ec8deb2afe297ecebe4285180c63f2f86f2d92"
wandb.login(key=wandb_api_key)


class Train_dataset(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        
        # if the data is jsonl file, load the data
        if config.data_path.endswith('.jsonl'):
            with open(config.data_path) as f:
                lines = f.readlines()
                self.data = [json.loads(line) for line in lines]
        elif config.data_path.endswith('.json'):
            with open(config.data_path) as f:
                self.data = json.load(f)
        
        newdata = []
        for da in self.data:
            newdata.append(da)
        print('filter out',len(self.data),len(newdata))
        self.data = newdata

        self.max_seq_len = self.config.max_seq_len
        self.debug = 1
        
        default_template = tokenizer.chat_template
        remove_text = """{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}"""
        new_template = default_template.replace(remove_text.strip(), "")
        
        self.template = Template(new_template)

    def __getitem__(self, index):
        return self.data[index]

    def get_response(self,da):
        base_flag = self.config.base_flag
        think_flag = self.config.think_flag
        
        response = da["response"].replace("\\/", "/").strip()

        match = re.search(r"<think>\s*(.*?)\s*</think>\s*(.*)", response, re.DOTALL)
        if match:
            thinking_trajectory = match.group(1).strip()
            attempt = match.group(2).strip()
        else:
            print(f"Warning: `<think>` parsing failed for response: {response}")
            thinking_trajectory = ""
            attempt = response.strip()
        
        if think_flag:
            return f"<think>\n{thinking_trajectory}\n</think>\n\n{attempt}"
        else:
            if base_flag:
                return attempt
            else:
                return f"<think>\n\n</think>\n\n{attempt}"
        
    def get_prompt(self,da):
        q = da['question']
        a = self.get_response(da)
        assert q is not None and a is not None, f'q:{q} a:{a}'
        if self.config.base_model == 'Llama':
            input =  self.template.render(messages=[{"role": "user", "content": q},{"role": "assistant", "content": a}],bos_token=self.tokenizer.bos_token,add_generation_prompt=False)
            query = self.template.render(messages=[{"role": "user", "content": q}],bos_token=self.tokenizer.bos_token,add_generation_prompt=True)
        elif self.config.base_model == 'Qwen':
            input =  self.template.render(messages=[{"role": "user", "content": q},{"role": "assistant", "content": a}],add_generation_prompt=False)
            query = self.template.render(messages=[{"role": "user", "content": q}],add_generation_prompt=True)
            # print("Input: ", input)
            # print("Query: ", query)
        input_ids = self.tokenizer.encode(input,add_special_tokens= False)
        query_ids = self.tokenizer.encode(query,add_special_tokens= False)
        labels = [-100]*len(query_ids) + input_ids[len(query_ids):]
        assert len(labels) == len(input_ids)
        return {"input_ids": input_ids[-self.max_seq_len:], "labels": labels[-self.max_seq_len:]}        

    def collate_fn(self, batch):
        data = [ self.get_prompt(da) for da in batch]
        input_ids = [item["input_ids"] for item in data]
        labels = [item["labels"] for item in data]
        max_len = max(len(x) for x in input_ids)
        max_len = min(max_len,self.max_seq_len)
        input_ids = [ item[:max_len] + [self.tokenizer.eos_token_id]*(max_len-len(item)) for item in input_ids]
        labels = [ item[:max_len] + [-100]*(max_len-len(item)) for item in labels]
        if self.debug < 3:
            print('input_ids',self.tokenizer.decode(input_ids[-1]))
            print('labels',self.tokenizer.decode([0 if x == -100 else x for x in labels[-1]]))
            self.debug += 1

        return {
                # 确保输入数据类型正确
                "input_ids": torch.LongTensor(input_ids),  # input_ids必须是long类型
                "labels": torch.LongTensor(labels),        # labels必须是long类型
            }
    
    def __len__(self):
        return len(self.data)

class SFTMetric:
    def __init__(self, device):
        self.n_step = 0
        # 使用bfloat16来减少显存占用
        self.right = torch.tensor(0.0, dtype=torch.bfloat16, device=device)
        self.total = torch.tensor(0.0, dtype=torch.bfloat16, device=device)
        self.total_loss = torch.tensor(0.0, dtype=torch.bfloat16, device=device)
        # self.world_size = dist.get_world_size()  # 注释掉分布式世界大小获取
        self.world_size = 1  # 单GPU设置为1

    def __call__(self, logits, labels, loss):
        return self.update(logits, labels, loss)

    def update(self, logits, labels, loss):
        self.n_step += 1
        with torch.no_grad():
            shift_preds = logits[..., :-1, :].argmax(dim=-1)
            shift_labels = labels[..., 1:]
            self.right += (shift_preds == shift_labels).masked_fill(shift_labels.eq(-100), 0).sum().item()
            self.total += (shift_labels != -100).sum().item()
            self.total_loss += loss.item()

    def get_metric(self, reset=True):
        # 注释掉分布式reduction操作
        # dist.all_reduce(self.right, op=torch.distributed.ReduceOp.SUM)
        # dist.all_reduce(self.total, op=torch.distributed.ReduceOp.SUM)
        # dist.all_reduce(self.total_loss, op=torch.distributed.ReduceOp.SUM)

        acc = (self.right / self.total).item()
        loss = self.total_loss.item() / (self.world_size * self.n_step)

        if reset:
            self.n_step = 0
            self.right.fill_(0)
            self.total.fill_(0)
            self.total_loss.fill_(0)
        return acc, loss


def get_gpu_memory_info():
    """获取GPU内存信息"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        return memory_allocated, memory_reserved, memory_total
    return 0, 0, 0

def train(args):
    
    # 设置默认的tensor dtype为bfloat16来节省显存
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 添加更多内存优化设置
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cudnn.benchmark = False  # 禁用benchmark以节省内存
    torch.backends.cudnn.deterministic = True
    
    # 检查GPU内存并给出建议
    if torch.cuda.is_available():
        memory_allocated, memory_reserved, memory_total = get_gpu_memory_info()
        print(f"GPU Memory Info - Total: {memory_total:.2f}GB, Reserved: {memory_reserved:.2f}GB, Allocated: {memory_allocated:.2f}GB")
        
        # 根据GPU内存大小给出建议
        if memory_total < 16:
            print("Warning: GPU memory < 16GB, consider reducing max_seq_len to 2048 or lower")
        elif memory_total < 24:
            print("Warning: GPU memory < 24GB, consider reducing max_seq_len to 4096 or lower")
    
    # 修改Accelerator配置，移除或简化分布式相关配置
    accelerator = Accelerator(
        mixed_precision='bf16', 
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # 移除不支持的cpu_offload参数
    ) 
    timestamp = get_time()
    if accelerator.is_main_process:
        model_name = args.model_path.split("/")[-1]
        wandb.init(project = args.experiment_name, config=args, dir=args.log_dir, name=f"{model_name}_think_flag{args.think_flag}_{timestamp}")
    
    accelerator.print(f'args:\n{args}')
    
    # 修改DeepSpeed配置部分，单GPU可能不需要这些配置
    # accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_bsz_per_gpu
    # accelerator.state.deepspeed_plugin.deepspeed_config['train_batch_size'] = args.train_bsz_per_gpu*dist.get_world_size()*accelerator.gradient_accumulation_steps
    
    # 单GPU版本的batch size配置
    if hasattr(accelerator.state, 'deepspeed_plugin') and accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_bsz_per_gpu
        accelerator.state.deepspeed_plugin.deepspeed_config['train_batch_size'] = args.train_bsz_per_gpu * accelerator.gradient_accumulation_steps

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # 添加更多内存优化选项
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # 确保使用bf16
        device_map="auto",  # 重新启用自动设备映射以优化内存使用
        low_cpu_mem_usage=True,  # 减少CPU内存使用
        # 添加更多优化选项
        use_cache=False,  # 禁用KV cache节省显存
    )

    # open gradient checkpointing
    model.gradient_checkpointing_enable()

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas= (0.9, 0.95))

    train_dataset = Train_dataset(args, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_bsz_per_gpu, shuffle=True, drop_last=True, collate_fn=train_dataset.collate_fn)

    # 修改训练步数计算，移除分布式相关的world_size
    # num_training_steps = int(len(train_dataloader) * (args.n_epochs)) // accelerator.gradient_accumulation_steps // dist.get_world_size()
    num_training_steps = int(len(train_dataloader) * (args.n_epochs)) // accelerator.gradient_accumulation_steps
    
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_rates * num_training_steps), num_training_steps=num_training_steps)
    accelerator.print(f'gradient_accumulation_steps:{accelerator.gradient_accumulation_steps} data_path:{args.data_path} lr:{args.learning_rate} num_training_steps:{num_training_steps}')
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    start_epoch = 0
    start_step = 0
    global_step = 0

    metric = SFTMetric(device=torch.cuda.current_device())

    def save_checkpoint(epoch, step, global_step):
        save_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")
        if accelerator.is_main_process:
            checkpoint_files = os.listdir(args.output_dir)
            checkpoint_files = [file for file in checkpoint_files if file.startswith("checkpoint-")]
            num_checkpoints = len(checkpoint_files)
            if args.max_ckpts>0:
                if num_checkpoints >= args.max_ckpts:
                    checkpoint_files.sort(key=lambda x: os.path.getctime(os.path.join(args.output_dir, x)))
                    oldest_checkpoint = checkpoint_files[0]
                    shutil.rmtree(os.path.join(args.output_dir, oldest_checkpoint))        
            os.makedirs(save_dir, exist_ok=True)
            output_dir = os.path.join(save_dir, 'tfmr')
            
            # 修改DeepSpeed相关的保存逻辑
            if hasattr(accelerator.state, 'deepspeed_plugin') and accelerator.state.deepspeed_plugin is not None and accelerator.state.deepspeed_plugin.zero_stage!=3:
                model.save_pretrained(output_dir,state_dict=accelerator.get_state_dict(model))
            else:
                # 单GPU情况下的保存逻辑
                unwrap_model = accelerator.unwrap_model(model)
                unwrap_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

        # 修改DeepSpeed ZeRO stage 3相关的保存逻辑
        if hasattr(accelerator.state, 'deepspeed_plugin') and accelerator.state.deepspeed_plugin is not None and accelerator.state.deepspeed_plugin.zero_stage==3:
            unwrap_model = accelerator.unwrap_model(model)
            unwrap_model.save_pretrained(os.path.join(save_dir, f'tfmr'),is_main_process=accelerator.is_main_process,save_function=accelerator.save,state_dict=accelerator.get_state_dict(model))
            
        accelerator.wait_for_everyone()
        accelerator.save({"epoch": epoch, "step": step, "global_step": global_step}, os.path.join(save_dir, "training_state.pt"))
        accelerator.print(f'checkpoint checkpoint-{epoch}-{global_step} is saved...')

    # 修改DeepSpeed配置打印
    if hasattr(accelerator.state, 'deepspeed_plugin') and accelerator.state.deepspeed_plugin is not None:
        accelerator.print(accelerator.deepspeed_config)
    
    # model.eval()
    # save_checkpoint(0,0,0)
    
    model.train()
    for epoch in range(start_epoch, args.n_epochs):
        time.sleep(10)
        train_dataloader_iterator = tqdm(enumerate(train_dataloader), total=len(train_dataloader)) if accelerator.is_main_process else enumerate(train_dataloader)
        for batch_cnt, batch in train_dataloader_iterator:
            if epoch==start_epoch and batch_cnt<start_step:
                continue

            if batch_cnt == 1 and epoch == 0:
                torch.cuda.empty_cache()
            
            # 更频繁地清理缓存以节省内存
            if global_step > 0 and global_step % 10 == 0:
                torch.cuda.empty_cache()
            
            # 在每次前向传播前检查内存使用情况
            if global_step % 100 == 0:
                memory_allocated, memory_reserved, memory_total = get_gpu_memory_info()
                if accelerator.is_main_process:
                    print(f"Step {global_step}: Memory allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB, Total: {memory_total:.2f}GB")
                    # 如果内存使用率过高，给出警告
                    if memory_allocated / memory_total > 0.9:
                        print("Warning: GPU memory usage > 90%, consider reducing batch size or sequence length")

            input_ids=batch['input_ids']
            labels=batch['labels']

            # 确保模型在正确的dtype下进行前向传播
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = model(input_ids=input_ids, labels=labels, return_dict=True, use_cache=False)
                loss = output.loss

            metric(output.logits, labels, loss)
            acc, train_loss = metric.get_metric()
            accelerator.backward(loss)
            if (global_step+1) % accelerator.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # 在优化器步骤后清理缓存
                torch.cuda.empty_cache()

            global_step += 1

            if accelerator.is_main_process:
                train_dataloader_iterator.set_postfix(epoch=epoch, current_step=batch_cnt, total_step=len(train_dataloader), skip=accelerator.optimizer_step_was_skipped, loss=round(train_loss, 3), acc=round(acc, 3), length=len(input_ids[0]), lr=lr_scheduler.get_last_lr()[0])

            if global_step % 3 == 0 and accelerator.is_main_process:
                wandb.log({
                    'skip': int(accelerator.optimizer_step_was_skipped),
                    'loss': train_loss,
                    'acc': acc,
                    'lr': lr_scheduler.get_last_lr()[0]
                }, step=global_step)

        accelerator.wait_for_everyone()
        save_checkpoint(epoch, batch_cnt, global_step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args of sft')
    # Experiment Args
    parser.add_argument('--experiment_name', type=str, required=True)

    # Model Args
    parser.add_argument('--model_path', required=True, type=str)

    # Data Args
    parser.add_argument('--data_path', required=True, type=str)

    # Training Args
    parser.add_argument('--output_dir', default='./ckpts', type=str)
    parser.add_argument('--max_ckpts', default=1, type=int)
    parser.add_argument('--log_dir', default='./train_logs', type=str)
    parser.add_argument('--max_seq_len', default=4096, type=int)  # 降低默认序列长度
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', default=32, type=int)  # 增加梯度累积步数
    parser.add_argument('--train_bsz_per_gpu', default=1, type=int) # train_bsz_per_gpu * num_gpu should be 8
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--warmup_rates', default=0.05, type=float)
    parser.add_argument('--n_epochs', default=5, type=int)
    parser.add_argument('--base_model', required=True, type=str, choices= ['Qwen', 'Llama'])
    parser.add_argument('--think_flag', required=True, type=int)
    parser.add_argument('--base_flag', required=True, type=int)

    # Other Args
    parser.add_argument('--seed', default=2002, type=int)

    args = parser.parse_args()
    model_name = args.model_path.split("/")[-1]
    args.log_dir = os.path.join(args.log_dir,args.experiment_name)
    args.log_dir = os.path.join(args.log_dir,model_name)
    args.log_dir = os.path.join(args.log_dir, f"think_flag{args.think_flag}")
    
    
    args.output_dir = os.path.join(args.output_dir,args.experiment_name)
    args.output_dir = os.path.join(args.output_dir,model_name)
    args.output_dir = os.path.join(args.output_dir, f"think_flag{args.think_flag}")
    

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    train(args)