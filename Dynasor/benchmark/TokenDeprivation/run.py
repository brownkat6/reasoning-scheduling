import argparse
from tqdm import tqdm
try:
    from utils import save_json, load_dataset, set_seed
    from clients import vllmClientModel, apply_chat_template
except:
    from Dynasor.benchmark.TokenDeprivation.utils import save_json, load_dataset, set_seed
    from Dynasor.benchmark.TokenDeprivation.clients import vllmClientModel, apply_chat_template
from dynasor.core.evaluator import (
    extract_answer,
    strip_string,
    math_equal,
    extract_first_boxed_answer,
)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Token Deprivation Experiment")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["amc23", "aime24", "GPQADiamond", "math500", "gsm8k"],
        help="Dataset to use (amc23 or aime24 or math500 or gsm8k)",
    )
    parser.add_argument(
        "--output", type=str, default="", help="Path to output results file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Name or path of the model to use",
    )
    parser.add_argument(
        "--probe",
        type=str,
        default="**Final Answer**\n\n\\[ \\boxed{",
        help="probe the LLM to output the answer in the format of boxed{...}",
    )
    parser.add_argument(
        "--probe-tokens", type=int, default=10, help="Number of tokens in probe"
    )

    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000/v1",
        help="URL of the model to use",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="token-abc123",
        help="API key of the model to use",
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Start index of the question"
    )
    parser.add_argument(
        "--end", type=int, default=10000, help="End index of the question"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum number of tokens per request",
    )
    parser.add_argument(
        "--step", type=int, default=128, help="Step size for token budget"
    )
    parser.add_argument(
        "--num-trials", type=int, default=10, help="Number of trials per question"
    )

    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Sampling temperature"
    )
    parser.add_argument("--top-p", type=float, default=0.95, help="Top p for sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def load_model_and_tokenizer(model_name, cache_dir=None):
    """Load model and tokenizer from HuggingFace."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        padding_side="left",
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer


def generate_batch_local_model(model, tokenizer, prompts, max_new_tokens, top_p, temperature, is_actives=None, batch_size=4):
    """Helper function to generate batch responses for local (non-vllm) models.
    
    Args:
        model: The HuggingFace model instance
        tokenizer: The tokenizer to use
        prompts: List of prompts to process
        max_new_tokens: Maximum number of new tokens to generate
        top_p: Top-p sampling parameter
        temperature: Temperature for sampling
        is_actives: List of booleans indicating which prompts to process. If None, process all.
        batch_size: Number of prompts to process in each batch
    
    Returns:
        List of response objects mimicking vllm's response format
    """
    device = next(model.parameters()).device
    responses = []
    
    if is_actives is None:
        is_actives = [True] * len(prompts)
    
    # Filter active prompts
    active_prompts = [(i, prompt) for i, (prompt, is_active) in enumerate(zip(prompts, is_actives)) if is_active]
    # Process in batches
    batch_responses = []
    for i in range(0, len(active_prompts), batch_size):
        batch_indices, batch_prompts = zip(*active_prompts[i:i + batch_size])
        
        # Tokenize entire batch at once
        inputs = tokenizer(
            list(batch_prompts), 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(device)
        
        torch.cuda.empty_cache()
        with torch.inference_mode():
            print(f"Generating batch of {len(batch_prompts)} prompts with shape {inputs.input_ids.shape} with {max_new_tokens} tokens")
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=1
            )
            
            for j, output in enumerate(outputs):
                input_length = inputs.input_ids[j].ne(tokenizer.pad_token_id).sum()
                generated_text = tokenizer.decode(output[input_length:], skip_special_tokens=True)
                
                response = type('Response', (), {
                    'choices': [type('Choice', (), {
                        'text': generated_text,
                        'finish_reason': 'length' if len(output) >= max_new_tokens else 'stop',
                        'logprobs': None
                    })]
                })
                batch_responses.append(response)
    
    # Reconstruct full response list with None for inactive prompts
    full_responses = [None] * len(prompts)
    for (idx, _), response in zip(active_prompts, batch_responses):
        full_responses[idx] = response
    return full_responses


def execute_question_reuse(
    model,
    prompt,
    target,
    max_tokens=[2048],
    probe=None,
    probe_tokens=10,
    num_trials=10,
    problem_id=None,
    output_dir=None,
    top_p=0.95,
    temperature=0.6,
    tokenizer=None,
    predicted_score=None,
):
    results = []
    current_prompts = [apply_chat_template(prompt, model.config._name_or_path) for _ in range(num_trials)]
    is_vllm = isinstance(model, vllmClientModel)
    assert(is_vllm==False)
    
    if not is_vllm:
        device = next(model.parameters()).device
        if tokenizer is None:
            tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else model.config.tokenizer
    print(len(max_tokens),max_tokens,"max tokens to execute")
    round_results_arr = []
    for i in tqdm(range(len(max_tokens)), desc="Executing questions"):
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        # Track which trials are finished
        if i == 0:
            is_finished = [False] * num_trials
            if is_vllm:
                responses = model.generate_batch(
                    current_prompts,
                    max_tokens=max_tokens[i],
                    is_actives=[True] * num_trials,
                    top_p=top_p,
                    temperature=temperature,
                )
            else:
                responses = generate_batch_local_model(
                    model,
                    tokenizer,
                    current_prompts,
                    max_new_tokens=max_tokens[i],
                    top_p=top_p,
                    temperature=temperature
                )
        else:
            # Calculate remaining tokens needed
            remaining_tokens = max_tokens[i] - max_tokens[i - 1]
            # Stitch previous response to prompt
            current_prompts = [
                current_prompt + completion[0]
                for current_prompt, completion in zip(current_prompts, completions)
            ]
            # Only generate for unfinished trials
            if is_vllm:
                responses = model.generate_batch(
                    current_prompts,
                    max_tokens=remaining_tokens,
                    is_actives=[not finished for finished in is_finished],
                    top_p=top_p,
                    temperature=temperature,
                )
            else:
                responses = generate_batch_local_model(
                    model,
                    tokenizer,
                    current_prompts,
                    max_new_tokens=remaining_tokens,
                    top_p=top_p,
                    temperature=temperature,
                    is_actives=[not finished for finished in is_finished]
                )

        # Process responses and create completions
        completions = []
        for trial in range(num_trials):
            if is_finished[trial]:
                completions.append(("", None))  # Empty completion for finished trials
            else:
                response = responses[trial]
                if response is None:
                    completions.append(("", None))
                else:
                    text = response.choices[0].text
                    finish_reason = response.choices[0].finish_reason
                    logprobs = response.choices[0].logprobs
                    completions.append((text, finish_reason))
                    # Update finished status if LLM completed naturally
                    if finish_reason != "length":
                        is_finished[trial] = True

        # Save results for this round
        round_results = {
            "round": i,
            "problem_id": problem_id,
            "max_tokens": max_tokens[i],
            "prompts": current_prompts,
            "new_tokens": [completion[0] for completion in completions],
            "finish_reasons": [completion[1] for completion in completions],
            "is_finished": is_finished,
            "target": target,
            "predicted_score": float(predicted_score) if predicted_score is not None else None
        }

        # Generate and save probed responses
        probe_prompts = [
            current_prompt + completion[0] + probe
            for current_prompt, completion in zip(current_prompts, completions)
        ]
        # Only generate probe responses for unfinished trials
        if is_vllm:
            probe_responses = model.generate_batch_probe(
                probe_prompts,
                max_tokens=probe_tokens,
                is_actives=[not finished for finished in is_finished],
            )
        else:
            probe_responses = generate_batch_local_model(
                model,
                tokenizer,
                probe_prompts,
                max_new_tokens=probe_tokens,
                top_p=top_p,
                temperature=temperature,
                is_actives=[not finished for finished in is_finished]
            )

        round_results["probe_prompts"] = probe_prompts
        round_results["probe_responses"] = [
            response.choices[0].text if response else "" for response in probe_responses
        ]

        is_corrects = []
        is_corrects_original = []
        for trial in range(num_trials):
            if is_finished[trial]:
                finished_result = extract_answer(
                    current_prompts[trial] + completions[trial][0], "aime24"
                )
                is_corrects.append(math_equal(finished_result, target))
            else:
                probe_result = extract_first_boxed_answer(
                        probe_prompts[trial] + probe_responses[trial].choices[0].text,
                        "aime24",
                )
                is_corrects.append(math_equal(probe_result, target))

            is_corrects_original.append(
                math_equal(
                    extract_answer(
                        current_prompts[trial] + completions[trial][0], "aime24"
                    ),
                    target,
                )
            )

        # Calculate and print actual proportion
        actual_proportion = sum(is_corrects) / len(is_corrects)
        if predicted_score is not None:
            print(f"Question {problem_id} - Token budget {max_tokens[i]}:")
            print(f"  Predicted proportion correct: {predicted_score:.2f}")
            print(f"  Actual proportion correct:    {actual_proportion:.2f}")

        round_results["is_corrects"] = is_corrects
        round_results["is_corrects_original"] = is_corrects_original
        round_results_arr.append(round_results)

        # Save results for this round to a file
        if output_dir:
            round_filename = (
                f"{output_dir}/question_{problem_id}_tokens_{max_tokens[i]}.json"
            )
            save_json(round_results, round_filename)
    return actual_proportion, round_results_arr

def main():
    args = parse_args()
    set_seed(args.seed)
    data = load_dataset(args.dataset)
    cache_dir = "/n/holylabs/LABS/dwork_lab/Everyone/cache/transformers"

    # Create output directory
    if args.output:
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
    else:
        # Create output directory with model name, dataset, parameters and date
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        model_name = args.model.replace("/", "-")
        output_dir = f"results/{model_name}_{args.dataset}_step{args.step}_max{args.max_tokens}_trials{args.num_trials}_{args.start}_{args.end}"
        os.makedirs(output_dir, exist_ok=True)

    # Replace load_model with load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model, cache_dir)

    for problem_id, item in enumerate(data):
        if problem_id < args.start:
            continue
        if problem_id >= args.end:
            break
        # execute question
        prompt = item["problem"].strip()
        target = strip_string(item["answer"])

        print(f"Executing question {problem_id} with target [{target}]")
        print(f"Prompt: {prompt}")
        print("-" * 100)
        token_budgets = list(range(args.step, args.max_tokens + args.step, args.step))
        batch_results, stats = execute_question_reuse(
            model,
            prompt,
            target,
            max_tokens=token_budgets,
            probe=args.probe,
            probe_tokens=args.probe_tokens,
            num_trials=args.num_trials,
            problem_id=problem_id,
            output_dir=output_dir,
            top_p=args.top_p,
            temperature=args.temperature,
            tokenizer=tokenizer,  # Pass tokenizer to execute_question_reuse
        )

    print("Saved results to", output_dir)


if __name__ == "__main__":
    main()
