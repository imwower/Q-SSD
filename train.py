"""
Advanced training loop for Q-SSD on Tiny Shakespeare with cosine warmup, grad accumulation,
checkpointing, and streaming generation.
"""
from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from config import QSSDConfig
from data import get_dataloaders
from models import QSSDModel


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cosine_with_warmup(
    warmup_steps: int,
    total_steps: int,
):
    def _schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return _schedule


def create_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    decay_params: List[Tensor] = []
    no_decay_params: List[Tensor] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and "embedding" not in name and "norm" not in name:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
    )


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_val: float,
    config: QSSDConfig,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "config": config.__dict__,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> Tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt.get("epoch", 0)
    best_val = ckpt.get("best_val", float("inf"))
    return start_epoch, best_val


@torch.no_grad()
def generate(
    model: QSSDModel,
    stoi: Dict[str, int],
    itos: List[str],
    device: torch.device,
    prompt: str = "The ",
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> str:
    model.eval()
    inference_params: Dict[str, List[Dict]] = {"layer_states": [dict() for _ in model.layers]}

    # Prime the state with the prompt.
    prompt_ids = torch.tensor([[stoi[ch] for ch in prompt]], device=device)
    for i in range(prompt_ids.size(1)):
        _ = model.step(prompt_ids[:, i : i + 1], inference_params)

    last_id = prompt_ids[:, -1:]
    generated: List[str] = list(prompt)

    for _ in range(max_new_tokens):
        logits = model.step(last_id, inference_params)
        logits = logits / temperature
        logits = torch.nan_to_num(logits, neginf=-50.0, posinf=50.0)
        probs = torch.softmax(logits, dim=-1)
        if torch.any(~torch.isfinite(probs)):
            probs = torch.full_like(probs, 1.0 / probs.size(-1))
        next_id = torch.multinomial(probs, num_samples=1)
        generated.append(itos[next_id.item()])
        last_id = next_id

    return "".join(generated)


def train_loop(
    epochs: int = 5,
    batch_size: int = 32,
    block_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    warmup_steps: int = 500,
    grad_accum_steps: int = 4,
    resume_from: Optional[str] = None,
    ckpt_dir: str = "checkpoints",
    max_train_steps_per_epoch: Optional[int] = 512,
    max_val_steps_per_epoch: Optional[int] = 128,
) -> None:
    device = get_device()
    print(f"Using device: {device}")

    pin = device.type == "cuda"
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

    optimizer = create_optimizer(model, lr=lr, weight_decay=weight_decay)
    total_train_steps = math.ceil(
        (max_train_steps_per_epoch or len(train_loader)) / grad_accum_steps
    ) * epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=cosine_with_warmup(warmup_steps, total_train_steps)
    )

    use_scaler = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    autocast = torch.amp.autocast if use_scaler else lambda *args, **kwargs: torch.autograd.profiler.record_function("no_autocast")

    start_epoch = 0
    best_val = float("inf")
    os.makedirs(ckpt_dir, exist_ok=True)
    if resume_from and os.path.exists(resume_from):
        start_epoch, best_val = load_checkpoint(resume_from, model, optimizer, scheduler)
        print(f"Resumed from {resume_from} at epoch {start_epoch}, best_val={best_val:.4f}")

    def run_epoch(loader, train: bool, max_steps: Optional[int]) -> float:
        model.train(train)
        total_loss = 0.0
        total_steps = 0
        start_time = time.time()

        optimizer.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(loader, 1):
            x = x.to(device)
            y = y.to(device)

            # Disable autocast on MPS/CPU to avoid BF16-induced NaNs; only CUDA uses autocast.
            with (autocast(device_type="cuda", dtype=torch.float16, enabled=use_scaler) if use_scaler else torch.autograd.profiler.record_function("fp32")):
                logits = model(x)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), y.view(-1)
                )
                loss = loss / grad_accum_steps
            if torch.isnan(loss) or torch.isinf(loss):
                print("Encountered NaN/Inf loss, skipping step.")
                optimizer.zero_grad(set_to_none=True)
                continue

            if train:
                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if step % grad_accum_steps == 0:
                    if use_scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if use_scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * grad_accum_steps
            total_steps += 1

            if step == 1 or step % max(1, len(loader) // 5) == 0 or step == len(loader):
                elapsed = time.time() - start_time
                avg_loss = total_loss / total_steps
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"{'Train' if train else 'Val'} "
                    f"Iter {step}/{len(loader)} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"LR: {lr_now:.2e} | "
                    f"Time: {elapsed:.1f}s"
                )

            if max_steps is not None and step >= max_steps:
                break

        # Flush leftover grads when early breaking or non-multiple batch count.
        if train and (total_steps % grad_accum_steps) != 0:
            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        return total_loss / max(total_steps, 1)

    for epoch in range(start_epoch + 1, epochs + 1):
        print(
            f"\nEpoch {epoch} | "
            f"train steps: {len(train_loader)}, "
            f"val steps: {len(val_loader)}, "
            f"batch_size: {batch_size}, "
            f"grad_accum: {grad_accum_steps}"
        )
        train_loss = run_epoch(
            train_loader, train=True, max_steps=max_train_steps_per_epoch
        )
        val_loss = run_epoch(
            val_loader, train=False, max_steps=max_val_steps_per_epoch
        )
        print(
            f"Epoch {epoch} done | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            save_checkpoint(
                os.path.join(ckpt_dir, "ckpt_best.pt"),
                model,
                optimizer,
                scheduler,
                epoch,
                best_val,
                config,
            )
        save_checkpoint(
            os.path.join(ckpt_dir, "ckpt_last.pt"),
            model,
            optimizer,
            scheduler,
            epoch,
            best_val,
            config,
        )

        sample = generate(
            model,
            stoi=stoi,
            itos=itos,
            device=device,
            prompt="The ",
            max_new_tokens=100,
            temperature=1.0,
        )
        print(f"Sample text:\n{sample}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Q-SSD on Tiny Shakespeare.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--max_train_steps_per_epoch", type=int, default=200)
    parser.add_argument("--max_val_steps_per_epoch", type=int, default=50)
    args = parser.parse_args()

    train_loop(
        epochs=args.epochs,
        batch_size=args.batch_size,
        block_size=args.block_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_accum_steps=args.grad_accum_steps,
        resume_from=args.resume_from,
        ckpt_dir=args.ckpt_dir,
        max_train_steps_per_epoch=args.max_train_steps_per_epoch,
        max_val_steps_per_epoch=args.max_val_steps_per_epoch,
    )
