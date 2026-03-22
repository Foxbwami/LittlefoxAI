import torch
from backend.core.tokenizer_bpe import encode, decode, vocab_size

# =========================
# GLOBAL SETTINGS
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# ENCODE WRAPPER
# =========================
def encode_text(text):
    return encode(text)


# =========================
# DECODE WRAPPER
# =========================
def decode_tokens(tokens):
    return decode(tokens)


# =========================
# BUILD DATA TENSOR
# =========================
def build_data_tensor(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = encode(text)
    return torch.tensor(tokens, dtype=torch.long)


# =========================
# GET BATCH (FOR TESTING ONLY)
# =========================
def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))

    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

    return x.to(DEVICE), y.to(DEVICE)


# =========================
# GENERATE HELPER (SIMPLE)
# =========================
def generate(model, prompt, max_new_tokens=50, temperature=1.0, top_k=None):
    model.eval()

    tokens = encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        if hasattr(model, "generate"):
            out = model.generate(
                x,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            return decode(out[0].tolist())

        for _ in range(max_new_tokens):
            logits = model(x)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                top_vals, _ = torch.topk(logits, k=top_k, dim=-1)
                min_top = top_vals[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_top, torch.full_like(logits, -1e10), logits)

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            x = torch.cat([x, next_token], dim=1)

    return decode(x[0].tolist())
