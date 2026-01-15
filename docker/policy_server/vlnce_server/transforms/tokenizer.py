"""
Instruction tokenization utilities for VLN-CE.
"""

import gzip
import json
import re
from typing import List, Optional

import numpy as np


def tokenize(sentence: str, keep: Optional[List[str]] = None) -> List[str]:
    """Tokenize a sentence into words.

    This matches the tokenization used in the VLN-CE dataset.

    Args:
        sentence: Input sentence to tokenize
        keep: List of special characters to keep (not split on)

    Returns:
        List of tokens (lowercase)
    """
    if keep is None:
        keep = ["'s", "'t", "'ll", "'m", "'ve", "'d", "'re"]

    # Pattern to split on non-alphanumeric, keeping specified strings
    pattern = r"([^\w\s]|\s)"
    for k in keep:
        pattern = pattern.replace(k, "")

    tokens = []
    for word in re.split(pattern, sentence.lower()):
        word = word.strip()
        if word:
            tokens.append(word)

    return tokens


class VocabDict:
    """Vocabulary dictionary for token to index conversion.

    Matches the VocabDict implementation in habitat.datasets.utils.
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self, word_list: List[str]):
        """Initialize vocabulary from word list.

        Args:
            word_list: List of vocabulary words in order.
                       Index 0 should be PAD, index 1 should be UNK.
        """
        self.word_list = word_list
        self.word2idx_dict = {word: idx for idx, word in enumerate(word_list)}
        self._unk_index = self.word2idx_dict.get(self.UNK_TOKEN, 1)

    def word2idx(self, word: str) -> int:
        """Convert word to index.

        Args:
            word: Word to convert

        Returns:
            Index in vocabulary, or UNK index if not found
        """
        return self.word2idx_dict.get(word, self._unk_index)

    def __len__(self) -> int:
        return len(self.word_list)

    def tokenize_and_index(self, sentence: str) -> List[int]:
        """Tokenize sentence and convert to indices.

        Args:
            sentence: Input sentence

        Returns:
            List of token indices
        """
        tokens = tokenize(sentence)
        return [self.word2idx(token) for token in tokens]


class InstructionTokenizer:
    """Tokenizes instruction text using VLN-CE vocabulary.

    Loads vocabulary from the R2R dataset and converts
    natural language instructions to padded token ID arrays.
    """

    def __init__(self, vocab_path: str):
        """Initialize tokenizer with vocabulary.

        Args:
            vocab_path: Path to dataset JSON file containing vocabulary.
                       e.g., data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz
        """
        self.vocab = self._load_vocab(vocab_path)

    def _load_vocab(self, vocab_path: str) -> VocabDict:
        """Load vocabulary from dataset file.

        Args:
            vocab_path: Path to gzipped JSON dataset file

        Returns:
            VocabDict instance
        """
        with gzip.open(vocab_path, "rt", encoding="utf-8") as f:
            data = json.load(f)

        word_list = data["instruction_vocab"]["word_list"]
        return VocabDict(word_list=word_list)

    def tokenize(self, text: str, max_length: int) -> np.ndarray:
        """Convert instruction text to padded token IDs.

        Args:
            text: Natural language instruction
            max_length: Maximum sequence length (will pad/truncate to this)

        Returns:
            Numpy array of token IDs, shape (max_length,), dtype int64
        """
        token_ids = self.vocab.tokenize_and_index(text)

        # Create padded array (0 is pad token)
        tokens = np.zeros(max_length, dtype=np.int64)

        # Fill with token IDs, truncating if necessary
        length = min(len(token_ids), max_length)
        tokens[:length] = token_ids[:length]

        return tokens

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
