import argparse
import os
import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from backend.core import config


def _parse_line(line):
    if "<|assistant|>" not in line:
        return None
    try:
        user_part, assistant_part = line.split("<|assistant|>", 1)
        user_part = user_part.replace("<|user|>", "").strip()
        assistant_part = assistant_part.replace("<|end|>", "").strip()
        if not user_part or not assistant_part:
            return None
        return user_part, assistant_part
    except Exception:
        return None


class QADataset(Dataset):
    def __init__(self, path, tokenizer, max_source_len=256, max_target_len=128, limit=None):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parsed = _parse_line(line)
                if parsed is None:
                    continue
                self.samples.append(parsed)
                if limit and len(self.samples) >= limit:
                    break
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user_text, assistant_text = self.samples[idx]
        model_inputs = self.tokenizer(
            user_text,
            max_length=self.max_source_len,
            truncation=True,
        )
        labels = self.tokenizer(
            text_target=assistant_text,
            max_length=self.max_target_len,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a seq2seq HF model on cleaned Q/A data.")
    parser.add_argument("--model-name", default=config.HF_GENERATION_MODEL)
    parser.add_argument("--data-path", default=config.PROCESSED_DATA_PATH)
    parser.add_argument("--output-dir", default=os.path.join(config.BASE_DIR, "models", "hf_finetuned"))
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--limit-samples", type=int, default=1000)
    parser.add_argument("--max-source-len", type=int, default=192)
    parser.add_argument("--max-target-len", type=int, default=96)
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data path not found: {args.data_path}")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=token)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, token=token)
    model.config.use_cache = False

    dataset = QADataset(
        args.data_path,
        tokenizer=tokenizer,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
        limit=args.limit_samples,
    )
    if len(dataset) == 0:
        raise RuntimeError("No training samples found.")

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        logging_steps=20,
        save_steps=100,
        save_total_limit=2,
        report_to=[],
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned model to: {args.output_dir}")


if __name__ == "__main__":
    main()
