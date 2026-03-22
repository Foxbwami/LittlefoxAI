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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPTMini(
        vocab_size,
        embed_size=config.EMBED_SIZE,
        heads=config.HEADS,
        layers=config.LAYERS,
        block_size=config.BLOCK_SIZE,
    ).to(device)

    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    model.train()

    dataset = StreamingTextDataset(
        config.PERSONALITY_PATH,
        block_size=config.BLOCK_SIZE,
        shuffle_buffer=config.SHUFFLE_BUFFER,
    )
    loader = DataLoader(dataset, batch_size=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    for epoch in range(3):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                yb.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Fine-tune epoch {epoch} done")

    torch.save(model.state_dict(), config.MODEL_PATH)
    print("Fine-tuning complete. Model saved.")


if __name__ == "__main__":
    main()
