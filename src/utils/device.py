from __future__ import annotations


def resolve_torch_device(requested_device: str) -> str:
    """Resolve a torch device string with graceful CPU fallback."""

    try:
        import torch
    except ImportError:
        return "cpu"

    requested = (requested_device or "cpu").lower()
    if requested.startswith("cuda"):
        return requested if torch.cuda.is_available() else "cpu"
    return requested
