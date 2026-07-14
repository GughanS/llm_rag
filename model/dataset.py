"""
TinyStories dataset loading and tokenisation.

Loads the Microsoft TinyStories dataset via HuggingFace `datasets` and
tokenises with GPT-2's BPE tokenizer. Stories are packed into fixed-length
chunks (max_seq_len) for efficient, padding-free batching.

Why packing?
  Padding wastes compute — short stories padded to max_seq_len burn FLOPs on
  PAD tokens. Packing concatenates all tokenised text into one long stream and
  slices it into equal-length windows. Every token in every batch is a real
  training signal.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset


class TinyStoriesDataset(Dataset):
    """Pre-tokenised, packed dataset for causal language modelling.

    Each sample is a (input_ids, targets) pair of shape (max_seq_len,),
    where targets are input_ids shifted right by one position.

    Args:
        split: "train" or "validation"
        max_seq_len: sequence length for each training sample
        tokenizer_name: HuggingFace tokenizer to use
        max_stories: optional cap on number of stories to load (for debugging)
    """

    def __init__(
        self,
        split: str = "train",
        max_seq_len: int = 512,
        tokenizer_name: str = "gpt2",
        max_stories: int | None = None,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load dataset
        ds = load_dataset("roneneldan/TinyStories", split=split)
        if max_stories is not None:
            ds = ds.select(range(min(max_stories, len(ds))))

        # Tokenise all stories and concatenate into one long token stream
        all_tokens: list[int] = []
        eos_id = self.tokenizer.eos_token_id
        for example in ds:
            tokens = self.tokenizer.encode(example["text"])
            all_tokens.extend(tokens)
            all_tokens.append(eos_id)  # EOS between stories

        # Pack into fixed-length chunks
        # Each chunk is max_seq_len + 1 tokens (input + 1 shifted target)
        chunk_len = max_seq_len + 1
        n_chunks = len(all_tokens) // chunk_len
        # Truncate tail tokens that don't fill a complete chunk
        all_tokens = all_tokens[: n_chunks * chunk_len]
        self.data = torch.tensor(all_tokens, dtype=torch.long).view(n_chunks, chunk_len)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx]
        input_ids = chunk[:-1]   # (max_seq_len,)
        targets = chunk[1:]      # (max_seq_len,) — shifted right by 1
        return input_ids, targets

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size
