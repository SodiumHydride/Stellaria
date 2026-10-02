"""Choosing validation clips: diverse lengths and pitch, never flagged."""

from __future__ import annotations

from ..schemas import Clip

MIN_VAL, MAX_VAL = 2, 20


def val_target(count: int, n_active: int) -> int:
    """``count`` clamped to 2..20 and to ≤ 5 % of the clips (but at least 2), leaving ≥ 1 for training."""
    target = max(MIN_VAL, min(MAX_VAL, int(count), int(n_active * 0.05)))
    return max(0, min(target, n_active - 1))


def pick_val(clips: list[Clip], count: int) -> set[str]:
    active = [c for c in clips if c.status == "active"]
    target = val_target(count, len(active))
    candidates = [c for c in active if not c.flags and 2.0 <= c.duration <= 20.0]
    if target == 0 or not candidates:
        return set()
    # sort by pitch then length and take evenly spaced picks → spread over the range
    candidates.sort(key=lambda c: ((c.metrics.f0_median or 0.0) if c.metrics else 0.0, c.duration, c.name))
    target = min(target, len(candidates))
    picks = {candidates[int((i + 0.5) * len(candidates) / target)].id for i in range(target)}
    return picks


def apply_val(clips: list[Clip], val_ids: set[str]) -> list[Clip]:
    """Set ``split`` for every active clip; returns the clips whose split changed."""
    changed = []
    for c in clips:
        if c.status != "active":
            continue
        split = "val" if c.id in val_ids else "train"
        if c.split != split:
            c.split = split
            changed.append(c)
    return changed
