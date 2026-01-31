"""
MMLU Pro: A Multi-discipline Multi-task Language Understanding Benchmark
Reference: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
"""

import random
import re
import os
import pathlib
import json
from typing import Optional, List, Any

import pandas as pd
from datasets import Dataset, load_dataset

import common
from common import HTML_JINJA
from classes import Eval, EvalResult, SamplerBase, SingleEvalResult

# Constants for answer options
ANSWER_OPTIONS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

# Categories in MMLU Pro
CATEGORIES = [
    'computer science', 'math', 'chemistry', 'engineering', 'law', 'biology',
    'health', 'physics', 'business', 'philosophy', 'economics', 'other',
    'psychology', 'history'
]

QUERY_TEMPLATE = """
Answer the following multiple choice question. Think step by step before answering.

IMPORTANT: Your final answer MUST be in the format "ANSWER: X" where X is one of the option letters {options}.
For example, if you think option B is correct, the last line of your response should be exactly "ANSWER: B".

Question: {question}

Options:
{formatted_options}
""".strip()

def parse_options(options_data: Any) -> List[str]:
    """Parse options from the dataset format to a list of strings."""
    if isinstance(options_data, list):
        return options_data
    
    if isinstance(options_data, str):
        try:
            parsed = json.loads(options_data)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            try:
                fixed_json = options_data.replace("'", '"')
                parsed = json.loads(fixed_json)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        
        if options_data.strip().startswith('[') and options_data.strip().endswith(']'):
            content = options_data.strip()[1:-1].strip()
            
            matches = re.findall(r'(?:"([^"]*)")|(?:\'([^\']*)\')', content)
            if matches:
                return [match[0] if match[0] else match[1] for match in matches]
            
            parts = []
            for part in content.split(','):
                part = part.strip()
                if (part.startswith('"') and part.endswith('"')) or (part.startswith("'") and part.endswith("'")):
                    part = part[1:-1]
                if part:
                    parts.append(part)
            if parts:
                return parts
    
    return []

def format_options(options: Any) -> str:
    """Format options in a consistent way for the prompt."""
    option_list = parse_options(options)
    
    cleaned_options = []
    for opt in option_list:
        if isinstance(opt, str):
            cleaned = opt.strip()
            cleaned = cleaned.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
            cleaned = cleaned.replace('&#39;', "'").replace('&quot;', '"')
            cleaned_options.append(cleaned)
        else:
            cleaned_options.append(str(opt))
    
    formatted = []
    for i, opt_text in enumerate(cleaned_options):
        if i < len(ANSWER_OPTIONS):
            option_letter = ANSWER_OPTIONS[i]
            if not opt_text.strip():
                opt_text = f"Option {option_letter}"
            formatted.append(f'({option_letter}): {opt_text}')
    
    return '\n'.join(formatted)

def extract_answer(response_text: str) -> Optional[str]:
    """Extract the answer from the model's response using multiple patterns."""
    patterns = [
        r"ANSWER:\s*([A-J])",                      # ANSWER: X
        r"(?:the\s+)?answer\s+is\s+\(?([A-J])\)?", # The answer is X
        r"answer:\s*([A-J])",                      # answer: X
        r"\*\*ANSWER:\*\*\s*([A-J])",              # **ANSWER:** X
        r"答案[是为:：]\s*([A-J])",                 
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    lines = response_text.strip().split('\n')
    for line in reversed(lines):  
        for opt in ANSWER_OPTIONS:
            if re.search(r'[^A-Z]' + opt + r'[^A-Z]|^' + opt + r'[^A-Z]|[^A-Z]' + opt + r'$|^' + opt + r'$', line):
                return opt
            
    return None

def load_mmlu_pro_dataset(data_dir: pathlib.Path) -> Dataset:
    """Load the MMLU Pro dataset."""
    os.makedirs(data_dir, exist_ok=True)
    
    cache_file = data_dir / "mmlu_pro.csv"
    if os.path.exists(cache_file):
        return Dataset.from_pandas(pd.read_csv(cache_file))
    
    dataset = load_dataset('TIGER-Lab/MMLU-Pro', split='test')
    df = dataset.to_pandas()
    df.to_csv(cache_file, index=False)
    return dataset

class MMLUProEval(Eval):
    def __init__(
        self,
        num_examples: int | None = None,
        random_seed: int = 0,
        data_dir: str | pathlib.Path = "./datasets/MMLUPro",
    ):
        self.rng = random.Random(random_seed)
        data_dir = pathlib.Path(data_dir)
        dataset = load_mmlu_pro_dataset(data_dir)
        examples = dataset.to_list()
        if num_examples:
            examples = self.rng.sample(examples, num_examples)
        self.examples = examples
        self.categories = {cat: 0 for cat in CATEGORIES}

    def __call__(self, sampler: SamplerBase, gen_file_path: str) -> EvalResult:
        prompt_messages_list = []
        for row in self.examples:
            option_list = parse_options(row['options'])
            formatted_options = format_options(row['options'])
            num_options = min(len(option_list), len(ANSWER_OPTIONS)) or 1
            prompt = QUERY_TEMPLATE.format(
                options=ANSWER_OPTIONS[:num_options],
                question=row['question'],
                formatted_options=formatted_options
            )
            prompt_messages = [sampler._pack_message(content=prompt, role="user")]
            prompt_messages_list.append(prompt_messages)
        
        response_texts = sampler(prompt_messages_list, gen_file_path)
        
        results = []
        for i, response_text in enumerate(response_texts):
            row = self.examples[i]
            prompt_messages = prompt_messages_list[i]
            extracted_answer = extract_answer(response_text)
            correct_answer = row['answer'].strip().upper() if 'answer' in row and isinstance(row['answer'], str) else None
            score = float(extracted_answer == correct_answer) if extracted_answer and correct_answer else 0.0
            category = row.get('category', 'unknown')
            
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=correct_answer,
                extracted_answer=extracted_answer,
            )
            
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            result = SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics={
                    'category_score': score,
                    f'category_{category}': score,
                    'answer_extracted': float(extracted_answer is not None)
                }
            )
            results.append(result)
        
        return common.aggregate_results(
            results,
            name2stats={
                'score': ('mean', 'std'),
                **{f'category_{cat}': ('mean',) for cat in CATEGORIES}
            }
        )

