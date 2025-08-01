import numpy as np


def sym_hanning(n):
    """
    Symmetric Hanning window.
    Returns an exactly symmetric N point window by evaluating
    the first half and then flipping the same samples over the other half.
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
    """
    Calculates Hanning window samples.
    Calculates and returns the first M points of an N point Hanning window.
    """
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, m + 1) / (n + 1)))
