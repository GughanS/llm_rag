"""
Dataset loading and tokenisation for causal language modelling.

Supports multiple datasets:
  - TinyStories (default): Microsoft's TinyStories dataset
  - OpenWebText: stas/openwebtext-10k (reliable fallback)
  - Synthetic: Random tokens for pipeline validation

All datasets are tokenised with GPT-2's BPE tokenizer and packed into
fixed-length chunks for efficient, padding-free batching.

Why packing?
  Padding wastes compute — short texts padded to max_seq_len burn FLOPs on
  PAD tokens. Packing concatenates all tokenised text into one long stream and
  slices it into equal-length windows. Every token in every batch is a real
  training signal.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    """Random-token dataset for validating the training pipeline.

    Generates random token sequences so training can run without any
    network downloads. Loss should drop from ~ln(vocab_size) towards
    ~0 quickly, confirming the model can overfit.

    Args:
        n_samples: number of training samples to generate
        max_seq_len: sequence length for each sample
        vocab_size: vocabulary size (must match model config)
    """

    def __init__(
        self,
        n_samples: int = 2000,
        max_seq_len: int = 512,
        vocab_size: int = 50257,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self._vocab_size = vocab_size
        chunk_len = max_seq_len + 1
        self.data = torch.randint(0, vocab_size, (n_samples, chunk_len))

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx]
        return chunk[:-1], chunk[1:]

    @property
    def vocab_size(self) -> int:
        return self._vocab_size


class PackedTextDataset(Dataset):
    """Pre-tokenised, packed dataset for causal language modelling.

    Each sample is a (input_ids, targets) pair of shape (max_seq_len,),
    where targets are input_ids shifted right by one position.

    Supports multiple HuggingFace datasets via the `dataset_name` parameter:
      - "tinystories": roneneldan/TinyStories (text field: "text")
      - "openwebtext": stas/openwebtext-10k   (text field: "text")

    Args:
        dataset_name: which dataset to load
        split: "train" or "validation" (mapped to "test" for wikitext)
        max_seq_len: sequence length for each training sample
        tokenizer_name: HuggingFace tokenizer to use
        max_samples: optional cap on number of samples to load
    """

    # Registry of supported datasets
    _DATASETS = {
        "tinystories": {
            "path": "roneneldan/TinyStories",
            "text_field": "text",
            "split_map": {"train": "train", "validation": "validation"},
        },
        "openwebtext": {
            "path": "stas/openwebtext-10k",
            "text_field": "text",
            "split_map": {"train": "train", "validation": "train"}, # openwebtext-10k only has a train split
        },
    }

    def __init__(
        self,
        dataset_name: str = "tinystories",
        split: str = "train",
        max_seq_len: int = 512,
        tokenizer_name: str = "gpt2",
        max_samples: int | None = None,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        from transformers import AutoTokenizer
        from datasets import load_dataset

        if dataset_name not in self._DATASETS:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Available: {list(self._DATASETS.keys())}"
            )
        ds_config = self._DATASETS[dataset_name]
        hf_split = ds_config["split_map"].get(split, split)
        text_field = ds_config["text_field"]

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load dataset — try streaming first, then fall back to full download
        load_kwargs = {"path": ds_config["path"], "split": hf_split}
        if "name" in ds_config:
            load_kwargs["name"] = ds_config["name"]

        try:
            ds = load_dataset(**load_kwargs, streaming=True)
            samples = []
            for i, example in enumerate(ds):
                if max_samples is not None and i >= max_samples:
                    break
                text = example[text_field].strip()
                if text:  # skip empty lines (common in wikitext)
                    samples.append(text)
            print(f"  Loaded {len(samples)} samples from {dataset_name} via streaming")
        except Exception as e:
            print(f"  Streaming failed ({e}), trying full download...")
            ds = load_dataset(**load_kwargs)
            if max_samples is not None:
                ds = ds.select(range(min(max_samples, len(ds))))
            samples = [ex[text_field].strip() for ex in ds if ex[text_field].strip()]

        # Tokenise all texts and concatenate into one long token stream
        all_tokens: list[int] = []
        eos_id = self.tokenizer.eos_token_id
        for text in samples:
            tokens = self.tokenizer.encode(text)
            all_tokens.extend(tokens)
            all_tokens.append(eos_id)

        # Pack into fixed-length chunks
        chunk_len = max_seq_len + 1
        n_chunks = len(all_tokens) // chunk_len
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


# Backwards compatibility alias
TinyStoriesDataset = PackedTextDataset
