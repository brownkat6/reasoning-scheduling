import os
import glob
import ast
import pandas as pd
import torch
import argparse
import wandb

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, EvalPrediction
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def load_early_stop_csvs(csv_dir: str, split: str):
    pattern = os.path.join(csv_dir, f"gsm8k_Y_{split}_*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched {pattern}")
    dfs = []
    for p in paths:
        df = pd.read_csv(p, usecols=["question_text", "early_stop_correct_proportions"])
        df["labels"] = df["early_stop_correct_proportions"].apply(ast.literal_eval)
        dfs.append(df[["question_text", "labels"]])
    return pd.concat(dfs, ignore_index=True)

class EarlyStopDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length=512):
        self.prompts = df["question_text"].tolist()
        self.labels = df["labels"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.prompts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

class EarlyStopFinetuner(nn.Module):
    def __init__(self, model_name, num_positions, use_lora, lora_r, lora_alpha, output_dir):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
            attn_implementation="flash_attention_2",
            device_map="auto",
            cache_dir=(output_dir + '/cache'),
        )
        hidden_size = self.lm.config.hidden_size
        if use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
            )
            self.lm = get_peft_model(self.lm, peft_config)
        self.regressor = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_positions),
            nn.Sigmoid(),
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]
        seq_lens = attention_mask.sum(dim=1) - 1
        last_h = last_hidden[torch.arange(len(seq_lens)), seq_lens]
        logits = self.regressor(last_h)
        loss = F.mse_loss(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)

def compute_metrics(eval_pred):
    preds, labels = (eval_pred.predictions, eval_pred.label_ids) \
        if isinstance(eval_pred, EvalPrediction) else eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    flat_preds = preds.reshape(-1)
    flat_labels = labels.reshape(-1)
    mse = mean_squared_error(flat_labels, flat_preds)
    pearson_r, _ = pearsonr(flat_labels, flat_preds)
    return {"mse": mse, "pearsonr": pearson_r}

def main():
    parser = argparse.ArgumentParser(description="Train EarlyStop Finetuner with LoRA")
    parser.add_argument("--csv_dir", type=str, default='/n/home04/amuppidi/reasoning-scheduling/data/gsm8k_results')
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_positions", type=int, default=16)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--wandb_project", type=str, default="early-stop-finetuner")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online")
    args = parser.parse_args()

    # Initialize W&B
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args), mode=args.wandb_mode)

    # Load data
    train_df = load_early_stop_csvs(args.csv_dir, "train")
    eval_df = load_early_stop_csvs(args.csv_dir, "test")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    train_ds = EarlyStopDataset(train_df, tokenizer)
    eval_ds = EarlyStopDataset(eval_df, tokenizer)

    # Model
    model = EarlyStopFinetuner(
        model_name=args.model_name,
        num_positions=args.num_positions,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        output_dir=args.output_dir,
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        fp16=False,
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        remove_unused_columns=True,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()