"""Canonical Live2D / face-tracking parameter vector for numeric conditioning.

Order is fixed and must match between dataset build, training, and inference.
Missing keys are filled with 0.0.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

# 16 floats matching the captions used for character_2 (no brows).
PARAM_NAMES: tuple[str, ...] = (
    "ParamAngleX",
    "ParamAngleY",
    "ParamAngleZ",
    "ParamMouthOpenY",
    "ParamMouthForm",
    "ParamEyeLOpen",
    "ParamEyeROpen",
    "ParamEyeBallX",
    "ParamEyeBallY",
    "FaceAngleX",
    "FaceAngleY",
    "FaceAngleZ",
    "MouthOpen",
    "MouthSmile",
    "EyeOpenLeft",
    "EyeOpenRight",
)

NUM_PARAMS = len(PARAM_NAMES)

# Indices whose sign flips under horizontal image flip (mirror).
FLIP_SIGN_PARAM_INDICES: tuple[int, ...] = tuple(
    i
    for i, name in enumerate(PARAM_NAMES)
    if name.endswith("X") or name in ("ParamAngleZ", "FaceAngleZ")
)

_CAPTION_PAIR_RE = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z0-9_]*)\s*=\s*(?P<value>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
)


def flip_params_horizontal(values: Sequence[float]) -> list[float]:
    """Negate X/Z params to match a horizontally flipped image."""
    out = [float(v) for v in values]
    if len(out) != NUM_PARAMS:
        raise ValueError(f"Expected {NUM_PARAMS} values, got {len(out)}")
    for i in FLIP_SIGN_PARAM_INDICES:
        out[i] = -out[i]
    return out


def params_from_mapping(params: Mapping[str, Any] | None) -> list[float]:
    """Build a fixed-length float vector from a name->value mapping."""
    src = params or {}
    out: list[float] = []
    for name in PARAM_NAMES:
        raw = src.get(name, 0.0)
        try:
            out.append(float(raw))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def parse_caption_params(caption: str) -> list[float]:
    """Parse `character; Name=1.2; ...` captions into the fixed PARAM_NAMES vector."""
    found: dict[str, float] = {}
    for match in _CAPTION_PAIR_RE.finditer(caption or ""):
        name = match.group("name")
        if name in found:
            continue
        try:
            found[name] = float(match.group("value"))
        except ValueError:
            continue
    return params_from_mapping(found)


def normalize_params(
    values: Sequence[float],
    mins: Sequence[float],
    maxs: Sequence[float],
    eps: float = 1e-6,
) -> list[float]:
    """Map each dim to roughly [-1, 1] using per-param min/max."""
    if len(values) != NUM_PARAMS:
        raise ValueError(f"Expected {NUM_PARAMS} values, got {len(values)}")
    if len(mins) != NUM_PARAMS or len(maxs) != NUM_PARAMS:
        raise ValueError("mins/maxs must have length NUM_PARAMS")
    out: list[float] = []
    for v, lo, hi in zip(values, mins, maxs):
        span = float(hi) - float(lo)
        if span < eps:
            out.append(0.0)
        else:
            # [lo, hi] -> [-1, 1]
            out.append(2.0 * ((float(v) - float(lo)) / span) - 1.0)
    return out


def denormalize_params(
    values: Sequence[float],
    mins: Sequence[float],
    maxs: Sequence[float],
) -> list[float]:
    """Inverse of normalize_params."""
    out: list[float] = []
    for v, lo, hi in zip(values, mins, maxs):
        mid = 0.5 * (float(hi) + float(lo))
        half = 0.5 * (float(hi) - float(lo))
        out.append(mid + float(v) * half)
    return out


def format_caption(character: str | None, values: Sequence[float]) -> str:
    """Serialize a param vector back to the legacy caption string (for UI display)."""
    pieces = [str(character or "character_unknown").strip() or "character_unknown"]
    for name, value in zip(PARAM_NAMES, values):
        text = f"{float(value):.4f}".rstrip("0").rstrip(".")
        if text in {"", "-", "-0"}:
            text = "0"
        pieces.append(f"{name}={text}")
    return "; ".join(pieces)
