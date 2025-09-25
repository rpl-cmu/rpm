"""Hanning window functions for MMWCAS processing."""

import numpy as np
from jaxtyping import Float


def sym_hanning(n: int) -> Float[np.ndarray, "n"]:
    """Symmetric Hanning window.

    !!! note
        Returns an exactly symmetric N point window by evaluating
        the first half and then flipping the same samples over the other half.

    Args:
        n: Number of points in the window.
    """
    if n % 2 == 0:
        # Even length window
        w = calc_hanning(n // 2, n)
        w = np.concatenate((w, w[::-1]))
    else:
        # Odd length window
        w = calc_hanning((n + 1) // 2, n)
        w = np.concatenate((w, w[-2::-1]))
    return w


def calc_hanning(m, n):
    """Calculate first half of Hanning window."""
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, m + 1) / (n + 1)))
