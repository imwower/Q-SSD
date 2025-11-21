"""
Tiny Shakespeare character-level dataset loader for Q-SSD experiments.

Features:
- Auto-downloads the corpus if missing.
- Simple char tokenizer (stoi/itos).
- Train/val split (90/10 by default).
- Provides sliding window samples (x, y) with block_size context.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import requests
import torch
from torch.utils.data import DataLoader, Dataset

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def _download_if_needed(path: str) -> None:
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    resp = requests.get(TINY_SHAKESPEARE_URL, timeout=30)
    resp.raise_for_status()
    with open(path, "w", encoding="utf-8") as f:
        f.write(resp.text)


def _build_tokenizer(text: str) -> Tuple[Dict[str, int], List[str]]:
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = chars
    return stoi, itos


class TextDataset(Dataset):
    """
    Character-level dataset with sliding window samples.
    """

    def __init__(
        self,
        block_size: int,
        split: str = "train",
        data_path: str = "input.txt",
        val_ratio: float = 0.1,
        stoi: Dict[str, int] | None = None,
        itos: List[str] | None = None,
    ) -> None:
        super().__init__()
        _download_if_needed(data_path)
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()

        if stoi is None or itos is None:
            stoi, itos = _build_tokenizer(text)
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(self.stoi)

        # Split text
        split_idx = int(len(text) * (1.0 - val_ratio))
        split_idx = max(split_idx, block_size + 2)  # ensure minimal length
        if split == "train":
            text = text[:split_idx]
        else:
            text = text[split_idx:]

        encode = lambda s: torch.tensor([self.stoi[ch] for ch in s], dtype=torch.long)
        self.data = encode(text)
        if len(self.data) <= block_size:
            raise ValueError("Text length is too short for the given block_size.")
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def get_dataloaders(
    batch_size: int,
    block_size: int,
    *,
    data_path: str = "input.txt",
    val_ratio: float = 0.1,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, Dict[str, int], List[str]]:
    """
    Create train/val dataloaders and return shared tokenizer.
    """
    _download_if_needed(data_path)
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    stoi, itos = _build_tokenizer(text)

    train_ds = TextDataset(
        block_size=block_size,
        split="train",
        data_path=data_path,
        val_ratio=val_ratio,
        stoi=stoi,
        itos=itos,
    )
    val_ds = TextDataset(
        block_size=block_size,
        split="val",
        data_path=data_path,
        val_ratio=val_ratio,
        stoi=stoi,
        itos=itos,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, stoi, itos


__all__ = ["TextDataset", "get_dataloaders"]
