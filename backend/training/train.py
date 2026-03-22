import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from backend.training.dataset import StreamingTextDataset
from backend.core.model import GPTMini
from backend.core.tokenizer_bpe import vocab_size
from backend.core import config

# =========================
# DEVICE SETUP
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# EVALUATION FUNCTION
# =========================
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)

            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                yb.view(-1),
                ignore_index=-100  # IMPORTANT (mask user tokens)
            )

            total_loss += loss.item()
            count += 1

            if count >= 100:  # limit validation steps
                break

    model.train()
    return total_loss / count if count > 0 else 0


# =========================
# TRAIN FUNCTION
# =========================
def train_model(file_path, epochs=None, steps_per_epoch=None):

    epochs = epochs if epochs is not None else config.EPOCHS
    steps_per_epoch = (
        steps_per_epoch if steps_per_epoch is not None else config.MAX_STEPS_PER_EPOCH
    )

    print(f"Using device: {device}")

    # =========================
    # MODEL
    # =========================
    model = GPTMini(
        vocab_size,
        embed_size=config.EMBED_SIZE,
        heads=config.HEADS,
        layers=config.LAYERS,
        block_size=config.BLOCK_SIZE,
    ).to(device)

    model.train()

    # =========================
    # OPTIMIZER + SCHEDULER
    # =========================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=steps_per_epoch
    )

    # =========================
    # DATASET
    # =========================
    dataset = StreamingTextDataset(
        file_path,
        block_size=config.BLOCK_SIZE,
        shuffle_buffer=config.SHUFFLE_BUFFER,
    )

    num_workers = 0 if os.name == "nt" or device == "cpu" else 2
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    # Validation (optional same dataset or separate file)
    val_loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE
    )

    # =========================
    # TRAIN LOOP
    # =========================
    global_step = 0

    for epoch in range(epochs):
        print(f"\n===== Epoch {epoch} =====")

        for step, (xb, yb) in enumerate(loader):
            xb = xb.to(device)
            yb = yb.to(device)

            # Forward
            logits = model(xb)

            # Loss (MASKING ENABLED)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                yb.view(-1),
                ignore_index=-100
            )

            # Backprop
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (IMPORTANT)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            global_step += 1

            # =========================
            # LOGGING
            # =========================
            if step % 100 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")

            # =========================
            # VALIDATION
            # =========================
            if step % 500 == 0 and step > 0:
                val_loss = evaluate(model, val_loader)
                print(f"Validation Loss: {val_loss:.4f}")

            # =========================
            # CHECKPOINTING
            # =========================
            if step % 1000 == 0 and step > 0:
                checkpoint_path = f"checkpoint_step_{global_step}.pt"
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": global_step
                }, checkpoint_path)

                print(f"Checkpoint saved: {checkpoint_path}")

            if step + 1 >= steps_per_epoch:
                break

    # =========================
    # SAVE FINAL MODEL
    # =========================
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)

    torch.save(model.state_dict(), config.MODEL_PATH)

    print("\nTraining complete. Model saved.")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    train_model(config.PROCESSED_DATA_PATH)
