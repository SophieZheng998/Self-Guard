import re
import random
import os
import json
import pathlib
from typing import List, Dict, Any

import common
from common import HTML_JINJA, check_equality
from classes import Eval, EvalResult, SamplerBase, SingleEvalResult

# Default dataset path
DEFAULT_AIME_PATH = "datasets/AIME/test.jsonl"

QUERY_TEMPLATE = """
Solve the following AIME math problem step by step. The last line of your response should be of the form \\boxed{{X}} where X is the answer to the problem.

{Problem}

IMPORTANT: Your final answer MUST be in the format \\boxed{{X}} where X is your numerical answer. This is critical for automated evaluation.
""".strip()

# Updated pattern to better capture complex expressions
BOXED_PATTERN = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"

ANSWER_IS_PATTERN = r"(?i)(?:the\s+)?answer\s+(?:is|=)\s+(\d+)"

LAST_LINE_NUMBER_PATTERN = r"(?m)^(\d+)$"

def load_aime_dataset(path: str, random_seed: int = 0, num_examples: int = None) -> List[Dict[str, Any]]:
    """Load AIME dataset from local JSONL file
    
    Args:
        path: Path to the JSONL file
        random_seed: Random seed used for sampling
        num_examples: Number of examples to sample, if specified
        
    Returns:
        List of dictionaries containing problems and answers
    """
    try:
        if os.path.exists(path):
            print(f"Loading AIME dataset from {path}")
            examples = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    if all(k in item for k in ['Problem', 'Answer']):
                        if isinstance(item['Answer'], str) and item['Answer'].isdigit():
                            item['Answer'] = item['Answer'].lstrip('0')
                            if not item['Answer']:
                                item['Answer'] = '0'
                        examples.append(item)
                    elif "problem" in item and "answer" in item:
                        answer = item["answer"]
                        if isinstance(answer, str) and answer.isdigit():
                            answer = answer.lstrip('0')
                            if not answer:
                                answer = '0'
                                
                        examples.append({
                            "ID": item.get("id", item.get("unique_id", "")),
                            "Problem": item["problem"],
                            "Answer": answer,
                            "Solution": item.get("solution", "")
                        })
            
            print(f"Successfully loaded {len(examples)} examples from AIME dataset")
            
            # If number of examples is specified, randomly sample
            if num_examples and num_examples < len(examples):
                rng = random.Random(random_seed)
                examples = rng.sample(examples, num_examples)
                
            return examples
        else:
            print(f"Warning: File {path} does not exist")
            return []
    except Exception as e:
        print(f"Error loading dataset from {path}: {e}")
        return []

class AIMEEval(Eval):
    def __init__(
        self,
        equality_checker: SamplerBase,
        num_examples: int | None = None,
        n_repeats: int = 1,
        random_seed: int = 0,
        data_dir: str | pathlib.Path = "./datasets"  # Add data directory parameter
    ):
        # Set data file path
        if isinstance(data_dir, str):
            data_dir = pathlib.Path(data_dir)
            
        # Load AIME dataset from local file
        local_path = data_dir / "AIME" / "test.jsonl"
        examples = load_aime_dataset(local_path, random_seed, num_examples)
        
        # If local file doesn't exist or fails to load, use synthetic examples as fallback
        if not examples:
            print("Warning: Failed to load AIME dataset from local file. Using synthetic examples as fallback.")
            # Create some synthetic examples as fallback
            examples = [
                {
                    'ID': 'synthetic-1',
                    'Problem': 'Find the value of x in the equation 2x + 3 = 7.',
                    'Answer': '2'
                },
                {
                    'ID': 'synthetic-2',
                    'Problem': 'Find the sum of all positive integers less than 10 that are relatively prime to 10.',
                    'Answer': '20'
                },
                {
                    'ID': 'synthetic-3',
                    'Problem': 'In how many ways can 5 distinct books be arranged on a shelf?',
                    'Answer': '120'
                }
            ]
        
        # Apply repetition
        self.examples = examples * n_repeats
        self.equality_checker = equality_checker
        
    def extract_answer(self, response_text):
        # First try to find the last \boxed{} expression
        boxed_matches = re.findall(BOXED_PATTERN, response_text)
        
        if boxed_matches:
            # Get the content of the last \boxed{} expression
            raw_answer = boxed_matches[-1].strip()
            
            # Check if the answer is a simple number
            if raw_answer.isdigit():
                return raw_answer
            
            # For AIME problems, the answer is usually a number
            # If it's the last content in the boxed content, try to extract just the number
            number_match = re.search(r'(\d+)$', raw_answer)
            if number_match:
                return number_match.group(1)
            
            # If we can't extract a simple number, return the full content
            # This handles cases like "4\sqrt{7} - 3"
            print(f"Extracted complex answer from last \\boxed: {raw_answer}")
            return raw_answer
        
        # Fallback: Check for patterns like "The answer is 42" or "Answer: 42"
        answer_is_match = re.search(ANSWER_IS_PATTERN, response_text)
        if answer_is_match:
            answer = answer_is_match.group(1).strip()
            print(f"Extracted answer from 'answer is' pattern: {answer}")
            return answer
        
        # Last resort: Check if the last line is just a number
        last_line_match = re.search(LAST_LINE_NUMBER_PATTERN, response_text)
        if last_line_match:
            answer = last_line_match.group(1).strip()
            print(f"Extracted answer from last line: {answer}")
            return answer
        
        return None

    def __call__(self, sampler: SamplerBase, gen_file_path: str) -> EvalResult:
        prompt_messages_list = []
        for row in self.examples:
            prompt = QUERY_TEMPLATE.format(Problem=row['Problem'])
            prompt_messages = [sampler._pack_message(content=prompt, role="user")]
            prompt_messages_list.append(prompt_messages)
        
        # response_texts = sampler(prompt_messages_list, gen_file_path)
        # xx
        with open(gen_file_path, "r", encoding="utf-8") as f:
            total_responses = json.load(f)
        response_texts = []
        for item in total_responses:
            response_texts.append(item["response"][item["optimal_idx"]])
        assert len(response_texts) == len(prompt_messages_list), "Mismatch between number of responses and prompts"
        # xx
        results = []
        
        for i, response_text in enumerate(response_texts):
            row = self.examples[i]
            prompt_messages = prompt_messages_list[i]
            
            extracted_answer = self.extract_answer(response_text)
            
            # Ensure target_answer is properly processed - strip leading zeros for consistent comparison
            target_answer = str(row['Answer']).strip()
            # print(target_answer,row['Answer'])
            # Check for exact match or use equation checker
            if extracted_answer == target_answer:
                score = 1.0
            elif extracted_answer and target_answer:
                # For more complex expressions, try checking mathematical equality
                check_result = check_equality(
                    self.equality_checker, extracted_answer, target_answer
                )
                score = float(check_result)
            else:
                score = 0.0
                
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=target_answer,
                extracted_answer=extracted_answer,
            )
            
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            result = SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics={
                    "extracted": float(extracted_answer is not None),
                    "chars": len(response_text),
                }
            )
            results.append(result)
            
        return common.aggregate_results(results)


# Example usage:
if __name__ == "__main__":
    # Simple test to verify dataset loading
    test_path = os.path.join("datasets", "AIME", "test.jsonl")
    examples = load_aime_dataset(test_path, num_examples=1)
    if examples:
        print("Dataset sample:")
        item = examples[0]
        print(f"ID: {item.get('ID')}")
        print(f"Problem: {item.get('Problem')[:100]}...")
        print(f"Answer: {item.get('Answer')}")
    
    # Test answer extraction
    test_responses = [
        r"The answer is \boxed{113}",
        r"Therefore, \boxed{4\sqrt{7} - 3}",
        r"Final answer: \boxed{4\sqrt{7} - 3}",
        r"We get \boxed{x = 42}"
    ]
    
    # Only create evaluator if we can load a model
    try:
        evaluator = AIMEEval(equality_checker=None, num_examples=1)
        for resp in test_responses:
            print(f"\nTesting: {resp}")
            print(f"Extracted: {evaluator.extract_answer(resp)}")
    except:
        print("Skipping evaluator test as equality_checker is required")

