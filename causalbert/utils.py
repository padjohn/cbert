"""
Shared utilities: dtype detection, class weight computation, token cleaning.
"""

from collections import Counter
import torch
def get_compute_dtype(device: str | None = None) -> tuple[torch.dtype, str]:
    """Determine optimal dtype based on device capabilities.
    
    Returns:
        tuple: (torch_dtype, device_string)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported())
        else (torch.float16 if device == "cuda" else torch.float32)
    )
    return dtype, device

def compute_ce_weights_from_counts(counts: Counter, num_classes: int, smoothing: float = 1.0,
                                    max_ratio: float = 10.0) -> list[float]:
    """Inverse-frequency class weights for CrossEntropy loss."""
    totals = [counts.get(i, 0) + smoothing for i in range(num_classes)]
    inv = [1.0 / t for t in totals]
    weights = [w / sum(inv) for w in inv]
    lo, hi = 1.0 / max_ratio, max_ratio
    weights = [min(max(w, lo), hi) for w in weights]
    return weights

def clean_tok(tok: str) -> str:
    """Clean special tokens and fix umlaut encoding."""
    tok = tok.lstrip("Ã„  ").strip("  ").strip("Ġ ")
    try:
        tok = tok.encode('latin1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return tok