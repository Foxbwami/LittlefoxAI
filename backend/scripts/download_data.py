import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.core import config


def stream_to_file(dataset, text_field, out_path, max_chars, prefix=None):
    written = 0
    with open(out_path, "a", encoding="utf-8") as f:
        if prefix:
            f.write(prefix + "\n")
        for sample in dataset:
            text = sample.get(text_field, "")
            if not text:
                continue
            f.write(text.replace("\n", " ") + "\n")
            written += len(text)
            if written >= max_chars:
                break
    return written


def main():
    os.makedirs(os.path.dirname(config.RAW_DATA_PATH), exist_ok=True)
    if os.path.exists(config.RAW_DATA_PATH):
        os.remove(config.RAW_DATA_PATH)

    try:
        from datasets import load_dataset
    except Exception as e:
        raise SystemExit(
            "Missing 'datasets' library. Install with: python -m pip install datasets"
        ) from e

    total = 0
    print("Streaming WikiText-103...")
    wikitext = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    total += stream_to_file(
        wikitext,
        "text",
        config.RAW_DATA_PATH,
        max_chars=config.RAW_MAX_CHARS // 2,
        prefix="",
    )

    print("Streaming OpenWebText...")
    openweb = load_dataset("openwebtext", split="train", streaming=True)
    total += stream_to_file(
        openweb,
        "text",
        config.RAW_DATA_PATH,
        max_chars=config.RAW_MAX_CHARS // 2,
        prefix="",
    )

    print(f"Wrote ~{total} characters to {config.RAW_DATA_PATH}")


if __name__ == "__main__":
    main()
