'''
Save datasets with keys "problem" and "answer"

TODO: similarly to how mlp_test.py loads numina, and gsm8k data from huggingface, load this data
and save it in the form of a {dataset_name}/test.jsonl file where keys include "problem" and "answer"
'''

import argparse
import json
import os
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Dataset Files for Token Deprivation")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["gsm8k", "numina"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "test"],
        help="Split to generate"
    )
    
    return parser.parse_args()

def load_gsm8k_data(split):
    """Load GSM8K dataset from HuggingFace"""
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    processed_data = []
    
    for i, item in enumerate(dataset):
        question = item["question"]
        answer_field = item["answer"]
        # Extract final answer after "#### "
        if "#### " in answer_field:
            ground_truth = answer_field.split("#### ")[-1].strip()
        else:
            ground_truth = answer_field.strip()
            
        processed_data.append({
            "id": f"{split}_{i}",
            "problem": question,
            "answer": ground_truth
        })
    
    return processed_data

def load_numina_data(split):
    """Load Numina dataset"""
    ds = load_dataset("AI-MO/NuminaMath-CoT", split=split)#, cache_dir="/n/netscratch/dwork_lab/Lab/katrina/datasets")
    
    test_data = []
    for i, sample in enumerate(ds):
        question = sample["problem"]
        answer_field = sample["solution"]
        if "boxed{" not in answer_field:
            continue
        ground_truth = answer_field.split("boxed{")[1].strip()
        if "}$" in ground_truth:
            ground_truth = ground_truth.split("}$")[0]
        elif ground_truth[-1] == "}":
            ground_truth = ground_truth[:-1]
        if ground_truth.endswith("}]"):
            ground_truth = ground_truth[:-2]
        # if there are any non-numeric characters in the answer then continue
        #if not ground_truth.replace(".", "").replace("-", "").isdigit():
        #    continue
        # if there are any alphabetical characters in the answer then continue
        #if any(c.isalpha() for c in ground_truth):
        #    continue
        #if len(ground_truth) > 6:
        #    # throw out answers with more than 6 characters for ease of matching
        #    continue
        test_data.append({"id": f"test_{i}", "problem": question, "answer": ground_truth})
    print(f"Loaded {len(test_data)} questions from Numina dataset for split {split}")

def save_jsonl(data, output_file):
    """Save data in JSONL format"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')

def main():
    args = parse_args()
    
    # Set output directory
    output_dir = f"Dynasor/benchmark/TokenDeprivation/data/{args.dataset}"
    
    # Load and process data based on dataset
    if args.dataset == "gsm8k":
        data = load_gsm8k_data(args.split)
    elif args.dataset == "numina":
        data = load_numina_data(args.split)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported for generation")
    
    # Save to JSONL file
    output_file = os.path.join(output_dir, f"{args.split}.jsonl")
    save_jsonl(data, output_file)
    print(f"Saved {len(data)} examples to {output_file}")

if __name__ == "__main__":
    main()