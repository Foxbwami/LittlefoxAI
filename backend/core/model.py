import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_size, heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size),
        )
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        seq_len = x.size(0)
        attn_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device),
            diagonal=1,
        ).bool()
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.ln1(x + attn_out)
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x


class GPTMini(nn.Module):
    def __init__(self, vocab_size, embed_size=128, heads=4, layers=4, block_size=64):
        super().__init__()
        self.block_size = block_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(block_size, embed_size)
        self.blocks = nn.Sequential(
            *[Block(embed_size, heads) for _ in range(layers)]
        )
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = self.embed(x) + self.pos_embed(positions)
        x = x.permute(1, 0, 2)
        x = self.blocks(x)
        x = x.permute(1, 0, 2)
        return self.fc(x)

    def generate(self, x, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            x_cond = x[:, -self.block_size :]
            logits = self(x_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                k = min(int(top_k), logits.size(-1))
                if k <= 0:
                    k = None
                if k is not None:
                    top_vals, _ = torch.topk(logits, k=k, dim=-1)
                    min_top = top_vals[:, -1].unsqueeze(-1)
                    logits = torch.where(logits < min_top, torch.full_like(logits, -1e10), logits)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            x = torch.cat([x, next_token], dim=1)
        return x
