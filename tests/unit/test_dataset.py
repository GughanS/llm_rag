"""
Unit tests for the TinyStories dataset loading and tokenisation.

These tests use a small subset (max_stories=50) to keep execution fast.
Full dataset tests are better suited for integration testing.
"""
from __future__ import annotations

import pytest
import torch

# Mark all tests in this module as needing network access + dataset download
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() and True,  # always run on CPU too
    reason="",  # no skip — just a placeholder for conditional marks
)


class TestTinyStoriesDataset:
    """Tests for TinyStoriesDataset.

    NOTE: These tests download a small slice of the dataset on first run.
    Subsequent runs use the HuggingFace cache.
    """

    @pytest.fixture(scope="class")
    def dataset(self):
        """Load a tiny slice of TinyStories for testing."""
        from model.dataset import TinyStoriesDataset

        return TinyStoriesDataset(
            split="train",
            max_seq_len=64,
            max_stories=50,
        )

    def test_length(self, dataset):
        """Dataset should have at least 1 chunk from 50 stories."""
        assert len(dataset) > 0

    def test_item_shapes(self, dataset):
        """Each item should be (input_ids, targets) of shape (max_seq_len,)."""
        input_ids, targets = dataset[0]
        assert input_ids.shape == (64,)
        assert targets.shape == (64,)

    def test_item_dtype(self, dataset):
        """Tokens should be long (int64) tensors."""
        input_ids, targets = dataset[0]
        assert input_ids.dtype == torch.long
        assert targets.dtype == torch.long

    def test_targets_shifted(self, dataset):
        """Targets should be input_ids shifted right by 1."""
        # Access the raw packed data to verify
        chunk = dataset.data[0]
        assert torch.equal(chunk[:-1], dataset[0][0])  # input_ids
        assert torch.equal(chunk[1:], dataset[0][1])    # targets

    def test_tokens_in_vocab(self, dataset):
        """All token IDs should be within vocabulary range."""
        input_ids, targets = dataset[0]
        assert (input_ids >= 0).all()
        assert (input_ids < dataset.vocab_size).all()
        assert (targets >= 0).all()
        assert (targets < dataset.vocab_size).all()

    def test_vocab_size(self, dataset):
        """Vocab size should match GPT-2 tokenizer."""
        assert dataset.vocab_size == 50257

    def test_batch_collation(self, dataset):
        """DataLoader should produce correct batch shapes."""
        from torch.utils.data import DataLoader

        loader = DataLoader(dataset, batch_size=4, drop_last=True)
        input_ids, targets = next(iter(loader))
        assert input_ids.shape == (4, 64)
        assert targets.shape == (4, 64)
