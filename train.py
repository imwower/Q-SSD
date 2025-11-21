"""
Minimal training loop for Q-SSD on Tiny Shakespeare (character-level).
Designed for quick iteration on Mac M2 (mps), with fallbacks to cuda/cpu.
"""
from __future__ import annotations

import math
import time
from contextlib import nullcontext
from typing import Tuple

import torch
from torch import nn

from config import QSSDConfig
from data import get_dataloaders
from models import QSSDModel


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def generate(model: QSSDModel, start: str, length: int, stoi, itos, device) -> str:
    model.eval()
    input_ids = torch.tensor([[stoi[ch] for ch in start]], device=device)
    for _ in range(length):
        logits = model(input_ids)  # (1, seq, vocab)
        next_logits = logits[:, -1, :]
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_id], dim=1)
    out = "".join(itos[idx] for idx in input_ids[0].tolist())
    return out


def train_loop(
    epochs: int = 5,
    batch_size: int = 32,
    block_size: int = 256,
    lr: float = 3e-4,
) -> None:
    device = get_device()
    print(f"Using device: {device}")

    pin = device.type == "cuda"  # pin_memory unsupported on mps; avoid warning
    train_loader, val_loader, stoi, itos = get_dataloaders(
        batch_size=batch_size,
        block_size=block_size,
        num_workers=0,
        val_ratio=0.1,
        data_path="input.txt",
        pin_memory=pin,
    )
    vocab_size = len(stoi)

    config = QSSDConfig(
        d_model=256,
        n_layer=4,
        vocab_size=vocab_size,
        ssm_cfg={"d_state": 16, "d_conv": 4, "expand": 2},
        rms_norm_eps=1e-5,
        bitnet_scale=True,
    )
    model = QSSDModel(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    use_amp = device.type == "cuda"  # CUDA amp only; mps/cpu fall back to FP32
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    autocast = torch.cuda.amp.autocast if use_amp else nullcontext

    def run_epoch(loader, train: bool) -> Tuple[float, int]:
        model.train(train)
        total_loss = 0.0
        total_steps = 0
        start_time = time.time()

        for step, (x, y) in enumerate(loader, 1):
            x = x.to(device)
            y = y.to(device)

            with autocast():
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), y.view(-1)
                )

            if train:
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item()
            total_steps += 1

            # Print a few times per epoch
            if train and step % max(1, math.ceil(len(loader) / 5)) == 0:
                elapsed = time.time() - start_time
                avg_loss = total_loss / total_steps
                print(
                    f"Iter {step}/{len(loader)} | "
                    f"{'Train' if train else 'Val'} Loss: {avg_loss:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )

        return total_loss / max(total_steps, 1), total_steps

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}")
        train_loss, _ = run_epoch(train_loader, train=True)
        val_loss, _ = run_epoch(val_loader, train=False)
        print(
            f"Epoch {epoch} done | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        # Generate sample text
        sample = generate(model, start="The ", length=100, stoi=stoi, itos=itos, device=device)
        print(f"Sample text:\n{sample}\n")


if __name__ == "__main__":
    train_loop(epochs=5, batch_size=32, block_size=256, lr=3e-4)
