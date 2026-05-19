"""Label-masking / random-flip poisoning — defensive evaluation only.

Implements the contamination pipeline used in Table 4 of the manuscript:
a fraction ``rate`` of training labels are replaced by a uniformly drawn
incorrect class. Returns a NEW dataloader-style iterable; never modifies
the original dataset in place.
"""
from __future__ import annotations
import torch


def label_mask_poison(labels: torch.Tensor, num_classes: int,
                      rate: float, generator: torch.Generator | None = None
                      ) -> torch.Tensor:
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be in [0, 1]")
    n = labels.shape[0]
    k = int(round(rate * n))
    if k == 0:
        return labels.clone()
    idx = torch.randperm(n, generator=generator)[:k]
    flipped = labels.clone()
    rand_lbl = torch.randint(0, num_classes, (k,), generator=generator)
    # Avoid collisions with the original label
    collide = (rand_lbl == labels[idx])
    rand_lbl[collide] = (rand_lbl[collide] + 1) % num_classes
    flipped[idx] = rand_lbl
    return flipped
