import random
import torch
from torch.utils.data import IterableDataset
from backend.core.tokenizer_bpe import encode

IGNORE_INDEX = -100


class StreamingInstructionDataset(IterableDataset):
    def __init__(self, file_path, block_size, shuffle_buffer=0):
        self.file_path = file_path
        self.block_size = block_size
        self.shuffle_buffer = shuffle_buffer

    # =========================
    # PARSE LINE INTO USER + ASSISTANT
    # =========================
    def parse_line(self, line):
        """
        Expected format:
        <|user|> ... <|assistant|> ... <|end|>
        """

        if "<|assistant|>" not in line:
            return None

        try:
            user_part, assistant_part = line.split("<|assistant|>", 1)

            # Remove <|user|> and <|end|> if present
            user_part = user_part.replace("<|user|>", "").strip()
            assistant_part = assistant_part.replace("<|end|>", "").strip()

            return user_part, assistant_part

        except:
            return None

    # =========================
    # BUILD TRAINING SAMPLE
    # =========================
    def build_sample(self, user_text, assistant_text):
        # Encode separately
        user_ids = encode("<|user|> " + user_text)
        assistant_ids = encode("<|assistant|> " + assistant_text + " <|end|>")

        # Combine input
        input_ids = user_ids + assistant_ids

        # Labels: mask user part
        labels = [IGNORE_INDEX] * len(user_ids) + assistant_ids

        # Truncate
        input_ids = input_ids[: self.block_size]
        labels = labels[: self.block_size]

        # Pad
        if len(input_ids) < self.block_size:
            pad_len = self.block_size - len(input_ids)
            input_ids += [0] * pad_len
            labels += [IGNORE_INDEX] * pad_len

        return torch.tensor(input_ids), torch.tensor(labels)

    # =========================
    # GENERATOR
    # =========================
    def _generator(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                parsed = self.parse_line(line)
                if parsed is None:
                    continue

                user_text, assistant_text = parsed

                try:
                    yield self.build_sample(user_text, assistant_text)
                except:
                    continue

    # =========================
    # ITERATOR WITH OPTIONAL SHUFFLE
    # =========================
    def __iter__(self):
        if self.shuffle_buffer <= 1:
            yield from self._generator()
            return

        buffer = []

        for item in self._generator():
            buffer.append(item)

            if len(buffer) >= self.shuffle_buffer:
                idx = random.randrange(len(buffer))
                yield buffer.pop(idx)

        while buffer:
            idx = random.randrange(len(buffer))
            yield buffer.pop(idx)


# Backward-compat alias used by training scripts.
class StreamingTextDataset(StreamingInstructionDataset):
    pass
