import os
import glob
import pandas as pd
import torch
import argparse
import wandb

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# -----------------------------------------------------------------------------
# 1) DATA LOADING
# -----------------------------------------------------------------------------
def load_difficulty_csvs(csv_dir: str, split: str):
    pattern = os.path.join(
        csv_dir, f"gsm8k_Y_{split}_*_with_difficulty.csv"
    )
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched {pattern}")
    dfs = []
    for p in paths:
        df = pd.read_csv(p, usecols=["question_text", "difficulty"])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


class DifficultyDataset(torch.utils.data.Dataset):
    label2id = {"easy": 0, "medium": 1, "hard": 2}

    def __init__(self, df: pd.DataFrame, tokenizer, max_length=512):
        self.texts = df["question_text"].tolist()
        # map "easy"/"medium"/"hard" → 0/1/2
        self.labels = [self.label2id[d] for d in df["difficulty"]]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# -----------------------------------------------------------------------------
# 2) MODEL DEFINITION
# -----------------------------------------------------------------------------
class DifficultyFinetuner(nn.Module):
    def __init__(self, model_name, use_lora, lora_r, lora_alpha, num_labels, output_dir):
        super().__init__()
        # load base LM
        self.lm = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
            device_map="auto",
            cache_dir=os.path.join(output_dir, "cache"),
        )

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

        hidden_size = self.lm.config.hidden_size
        # classification head: hidden_size → num_labels
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_labels),
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # grab last token representation
        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden)
        seq_lens = attention_mask.sum(dim=1) - 1   # index of last real token
        last_repr = hidden_states[torch.arange(len(seq_lens)), seq_lens]  # (batch, hidden)
        logits = self.classifier(last_repr)  # (batch, num_labels)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


def compute_metrics(p: EvalPrediction):
    preds = p.predictions.argmax(axis=-1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# -----------------------------------------------------------------------------
# 3) TRAIN / EVAL SETUP
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Finetune for Difficulty Classification")
    parser.add_argument("--csv_dir", type=str,
                        default="/n/home04/amuppidi/reasoning-scheduling/data/gsm8k_results_with_difficulty")
    parser.add_argument("--model_name", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--wandb_project", type=str, default="difficulty-classifier")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online")
    args = parser.parse_args()

    # init W&B
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
        mode=args.wandb_mode
    )

    # load dataframes
    train_df = load_difficulty_csvs(args.csv_dir, "train")
    eval_df  = load_difficulty_csvs(args.csv_dir, "test")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    train_ds = DifficultyDataset(train_df, tokenizer)
    eval_ds  = DifficultyDataset(eval_df, tokenizer)

    # build model
    model = DifficultyFinetuner(
        model_name=args.model_name,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        num_labels=3,
        output_dir=args.output_dir
    )

    # training arguments
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
        metric_for_best_model="accuracy",
        greater_is_better=True,
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
    wandb.finish()
    print(f"✓ Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
