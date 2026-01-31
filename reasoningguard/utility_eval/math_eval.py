"""
Measuring Mathematical Problem Solving With the MATH Dataset
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2103.03874

This module also supports loading data from local JSONL files for MATH dataset.
"""

import random
from typing import Literal
import re
import os
import json
import pathlib
from tqdm import tqdm

import common
from classes import Eval, EvalResult, SamplerBase, SingleEvalResult


DEFAULT_MATH_TEST_PATH = "../../evaluation/utility_benchmark/datasets/MATH/test.jsonl"

QUERY_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{Question}

Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.
""".strip()

def math_postproc_instruction(answer):
    return f"""Extract ONLY the final answer from the following solution text. Do not include any explanations or additional text. Return exactly the substring that follows the last occurrence of "Answer:" in the solution.

Solution text:
{answer}

Final Answer:"""


def math_judge_instruction(pred, target):
    return """Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 5
    Expression 2: x=5

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes
(give benefit of the doubt to units)

---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: %s
    Expression 2: %s""" % (pred, target)

def extract_actual_answer(response_text: str) -> str:
    response_text = response_text.strip()
    match = re.search(r'\.\s*([^\.]+)$', response_text)
    if match:
        return match.group(1).strip()
    return response_text

def load_math_dataset(path: str, split: str = "math_test") -> list:
    """Load dataset from local JSONL file
    
    Args:
        path: Path to the JSONL file
        split: Dataset split name, used for logging
        
    Returns:
        List of dictionaries containing questions and answers
    """
    try:
        if os.path.exists(path):
            print(f"Loading {split} dataset from {path}")
            examples = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    if "problem" in item and "answer" in item:
                        examples.append({
                            "Question": item["problem"],
                            "Answer": item["answer"],
                            "Solution": item.get("solution", ""),
                            "Subject": item.get("subject", ""),
                            "Level": item.get("level", ""),
                            "ID": item.get("id", item.get("unique_id", ""))
                        })
                    elif "Question" in item and "Answer" in item:
                        examples.append(item)
            
            print(f"Successfully loaded {len(examples)} examples from {split} dataset")
            return examples
        else:
            print(f"Warning: File {path} does not exist")
            return []
    except Exception as e:
        print(f"Error loading dataset from {path}: {e}")
        return []

class MathEval(Eval):
    def __init__(
        self,
        equality_checker: SamplerBase,
        num_examples: int | None = None,
        n_repeats: int = 16,
        split: Literal["math_test", "math_500_test"] = "math_test",
        random_seed: int | None = None,
        data_dir: str | pathlib.Path = "../../evaluation/utility_benchmark/datasets"  # Add data directory parameter
    ):
        # Set data file paths
        if isinstance(data_dir, str):
            data_dir = pathlib.Path(data_dir)
        
        
        # Try to load MATH dataset from local file
        local_path = data_dir / "MATH" / "test.jsonl"
        examples = load_math_dataset(local_path, split)
            
            
        
        # Initialize random number generator
        self.rng = random.Random(random_seed)
        
        # If num_examples is specified, sample
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported for num_examples = None"
            examples = self.rng.sample(examples, num_examples)
        
        # Apply repetition
        self.examples = examples * n_repeats
        self.equality_checker = equality_checker

    def __call__(self, sampler: SamplerBase, gen_file_path: str) -> EvalResult:
        # Prepare all prompts in a batch
        prompt_messages_list = []
        for row in tqdm(self.examples):
            template = QUERY_TEMPLATE
            prompt_messages = [
                sampler._pack_message(content=template.format(**row), role="user")
            ]
            prompt_messages_list.append(prompt_messages)
        
        # Get responses from the model in a single batch
        # response_texts = sampler(prompt_messages_list, gen_file_path)
        # xx
        with open(gen_file_path, "r", encoding="utf-8") as f:
            total_responses = json.load(f)
        response_texts = []
        for item in total_responses:
            response_texts.append(item["response"][item["optimal_idx"]])
        assert len(response_texts) == len(prompt_messages_list), "Mismatch between number of responses and prompts"
        # xx
        
        # Prepare extraction prompts in a batch
        extract_messages_list = []
        for response_text in tqdm(response_texts):
            # Extract actual answer, removing any thinking process
            actual_response = extract_actual_answer(response_text)
            extract_prompt = math_postproc_instruction(actual_response)
            extract_messages = self.equality_checker._pack_message(content=extract_prompt, role="user")
            extract_messages_list.append(extract_messages)
        
        # Get extracted answers in batch
        extracted_answers_responses = []  
        for extract_messages in tqdm(extract_messages_list):
            response = self.equality_checker([extract_messages])
            if response == "" or (isinstance(response, str) and response.strip() == ""):
                extracted_answers_responses.append("?")
            else:
                extracted_answers_responses.append(response)

        # Prepare judge prompts for valid answers
        judge_messages_list = []
        judge_indices = []  # Track which indices have valid answers
        
        for i, (extracted_answer, row) in enumerate(tqdm(zip(extracted_answers_responses, self.examples))):
            judge_prompt = math_judge_instruction(extracted_answer, row["Answer"])
            judge_messages = self.equality_checker._pack_message(content=judge_prompt, role="user")
            judge_messages_list.append(judge_messages)
            judge_indices.append(i)
        
        judge_responses = []
        for judge_message in tqdm(judge_messages_list):
            response = self.equality_checker([judge_message])
            judge_responses.append(response)
        
        # Process results
        results = []
        judge_idx = 0
        
        for i in range(len(response_texts)):
            response_text = response_texts[i]
            row = self.examples[i]
            prompt_messages = prompt_messages_list[i]
            
            # Get extracted answer
            extracted_answer = None
            if extracted_answers_responses[i] != "?":
                extracted_answer = extracted_answers_responses[i]
            
            # Get score
            score = 0.0
            if i in judge_indices:
                score = float(judge_responses[judge_idx].strip() == "Yes")
                judge_idx += 1
            
            # Generate HTML report
            html = common.jinja_env.from_string(common.HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
            )
            
            # Create conversation record
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            result = SingleEvalResult(
                html=html, 
                score=score, 
                convo=convo,
                metrics={
                    "answer_extracted": extracted_answer is not None,
                    "answer_normalized": bool(extracted_answer),
                }
            )
            results.append(result)
        
        return common.aggregate_results(results)
