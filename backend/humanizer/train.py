import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader
import torch
from backend.core import config


def load_pairs(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    pairs = []
    robot = None
    for line in lines:
        if line.lower().startswith("robot:"):
            robot = line
        elif line.lower().startswith("human:") and robot:
            pairs.append(f"{robot}\n{line}")
            robot = None
    return pairs


def main():
    data_path = f"{config.DATA_DIR}/humanizer_dataset.txt"
    pairs = load_pairs(data_path)
    if not pairs:
        print("No training pairs found.")
        return

    tokenizer = GPT2Tokenizer.from_pretrained(config.HUMANIZER_BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(config.HUMANIZER_BASE_MODEL)

    def preprocess(text):
        enc = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        enc["labels"] = enc["input_ids"].clone()
        return enc

    batches = [preprocess(p) for p in pairs]
    loader = DataLoader(batches, batch_size=1, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    max_steps = 5
    step = 0
    for _ in range(1):
        for batch in loader:
            batch = {k: v.squeeze(0).to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            print(f"step {step} loss {loss.item():.4f}", flush=True)
            if step >= max_steps:
                break
        if step >= max_steps:
            break

    model.save_pretrained(config.HUMANIZER_MODEL_PATH)
    tokenizer.save_pretrained(config.HUMANIZER_MODEL_PATH)
    print(f"Saved humanizer model to {config.HUMANIZER_MODEL_PATH}")


if __name__ == "__main__":
    main()
