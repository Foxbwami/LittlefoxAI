import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from backend.core import config
from backend.core.model import GPTMini
from backend.core.tokenizer_bpe import encode, decode, vocab_size
from backend.core.postprocess import format_response


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(path=None):
    model_path = path or config.MODEL_PATH
    model = GPTMini(
        vocab_size,
        embed_size=config.EMBED_SIZE,
        heads=config.HEADS,
        layers=config.LAYERS,
        block_size=config.BLOCK_SIZE,
    ).to(DEVICE)
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        return model, model_path
    raise FileNotFoundError(f"Model not found: {model_path}")


def generate_text(
    prompt,
    max_new_tokens=80,
    temperature=0.9,
    top_k=40,
    model_path=None,
):
    model, loaded_path = _load_model(model_path)
    tokens = encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    decoded = decode(out[0].tolist())
    cleaned = format_response(decoded)
    return cleaned, loaded_path


def _diagnostics(model_path):
    print(f"[gen] device={DEVICE}")
    print(f"[gen] model_path={model_path}")
    print(f"[gen] vocab_size={vocab_size}")
    print(f"[gen] tokenizer_path={config.TOKENIZER_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Generate text with the local GPTMini model")
    parser.add_argument("prompt", nargs="*", help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()

    if args.prompt:
        prompt = " ".join(args.prompt)
    else:
        prompt = input("Prompt: ").strip()

    if not prompt:
        print("No prompt provided.")
        return

    try:
        text, loaded_path = generate_text(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            model_path=args.model_path,
        )
        if args.diagnostics:
            _diagnostics(loaded_path)
        print("\n" + text)
    except Exception as exc:
        msg = str(exc)
        if "size mismatch" in msg and "state_dict" in msg:
            msg += "\nHint: tokenizer vocab changed. Re-run training to regenerate model weights."
        print(f"Generation failed: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
