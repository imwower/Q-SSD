"""
Smoke tests for Q-SSD components: text path, DVS path, and a tiny training step.
"""
from __future__ import annotations

import torch
from torch import nn

from config import QSSDConfig
from event2vec import Event2Vec
from models import QSSDModel
from quantization import BitLinear


def text_mode_smoke() -> None:
    print("=== Text mode ===")
    config = QSSDConfig()
    model = QSSDModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 128))

    logits = model(input_ids)
    print("Logits shape:", tuple(logits.shape))

    bitlinear_count = sum(1 for _ in model.modules() if isinstance(_, BitLinear))
    print("BitLinear layers:", bitlinear_count)
    print(model)


def dvs_mode_smoke() -> None:
    print("\n=== DVS mode ===")
    config = QSSDConfig()
    model = QSSDModel(config)
    encoder = Event2Vec(resolution=(32, 32), dim=config.d_model, in_channels=2)

    dvs = torch.randn(2, 64, 2, 32, 32)
    embeddings = encoder(dvs)  # (B, T, d_model)

    # Bypass token embedding and feed embeddings through the backbone + head.
    x = embeddings
    for layer in model.layers:
        x = layer(x)
    x = model.norm(x)
    logits = model.lm_head(x)

    print("DVS embeddings shape:", tuple(embeddings.shape))
    print("DVS logits shape:", tuple(logits.shape))


def tiny_training_step() -> None:
    print("\n=== Tiny training step ===")
    config = QSSDConfig()
    model = QSSDModel(config)
    model.train()

    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    targets = torch.randint(0, config.vocab_size, (2, 16))

    logits = model(input_ids)
    loss = nn.functional.cross_entropy(
        logits.view(-1, config.vocab_size), targets.view(-1)
    )
    loss.backward()

    # Check gradients propagate through BitLinear weights.
    grads_ok = all(
        p.grad is not None for p in model.parameters() if isinstance(p, torch.Tensor)
    )
    print("Loss:", loss.item())
    print("All parameter grads exist:", grads_ok)


def main() -> None:
    text_mode_smoke()
    dvs_mode_smoke()
    tiny_training_step()


if __name__ == "__main__":
    main()
