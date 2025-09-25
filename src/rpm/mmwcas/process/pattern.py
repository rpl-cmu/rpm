"""Antenna pattern processing for mmwcas radar."""

import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Float
from scipy.io import loadmat

mmwcas_path = os.path.dirname(os.path.abspath(__file__))


class PatternELE:
    """Antenna pattern processing for elevation angle.

    Args:
        pattern_file: path to the .mat file containing the antenna pattern.
        clip_bound: maximum absolute elevation angle to consider.
        xla: array library to use (e.g., numpy, jax.numpy).
    """

    def __init__(
        self,
        pattern_file: str = f"{mmwcas_path}/pattern.mat",
        clip_bound: float = 25.0,
        xla=np,
    ):
        self.xla = xla
        self.clip_bound = clip_bound
        self.mat = loadmat(pattern_file)["pat"]
        self.mat = xla.asarray(self.mat, dtype=xla.float32)
        self.gain_dB = self.mat[:, 90:271]  # (elevation, azimuth)
        self.gain_dB = self.gain_dB - xla.max(self.gain_dB)
        self.gain_dB_ele = xla.clip(self.gain_dB[:, 90], -50, 10)

    def plot_gain_ele(self):
        """Plot gain vs elevation angle."""
        plt.plot(self.xla.arange(-90, 91, 1), self.gain_dB_ele)
        plt.show()

    def gain(self, angle: Float[Array, "n"]) -> Float[Array, "n"]:
        """Interpolate gain from angle.

        Args:
            angle: elevation angle in degree

        Returns:
            Interpolated gain in dB.
        """
        angle = self.xla.clip(angle, -self.clip_bound, self.clip_bound)
        u, d = (
            self.xla.ceil(angle).astype(self.xla.int32),
            self.xla.floor(angle).astype(self.xla.int32),
        )
        ug, dg = self.gain_dB_ele[u + 90], self.gain_dB_ele[d + 90]
        return dg + (angle - d) * (ug - dg)


class PatternAZI:
    """Antenna pattern processing for azimuth angle.

    Args:
        pattern_file: path to the .mat file containing the antenna pattern.
        clip_bound: maximum absolute azimuth angle to consider.
        xla: array library to use (e.g., numpy, jax.numpy).
    """

    def __init__(
        self, pattern_file=f"{mmwcas_path}/pattern.mat", clip_bound=80, xla=np
    ):
        self.xla = xla
        self.clip_bound = clip_bound
        self.mat = loadmat(pattern_file)["pat"]
        self.mat = xla.asarray(self.mat, dtype=xla.float32)
        self.gain_dB = self.mat[:, 90:271]  # (elevation, azimuth)
        self.gain_dB = self.gain_dB - xla.max(self.gain_dB)
        self.gain_dB_azi = xla.clip(self.gain_dB[90], -50, 10)

    def plot_gai_azi(self):
        """Plot gain vs azimuth angle."""
        plt.plot(self.xla.arange(-90, 91, 1), self.gain_dB_azi)
        plt.show()

    def gain(self, angle: Float[Array, "n"]) -> Float[Array, "n"]:
        """Interpolate gain from azimuth angle.

        Args:
            angle: elevation angle in degree

        Returns:
            Interpolated gain in dB.
        """
        angle = self.xla.clip(angle, -self.clip_bound, self.clip_bound)
        u, d = (
            self.xla.ceil(angle).astype(self.xla.int32),
            self.xla.floor(angle).astype(self.xla.int32),
        )
        ug, dg = self.gain_dB_azi[u + 90], self.gain_dB_azi[d + 90]
        return dg + (angle - d) * (ug - dg)


if __name__ == "__main__":
    pattern = PatternAZI("pattern.mat")
    # pattern = PatternELE("pattern.mat")
    g = pattern.gain(jnp.arange(-90, 91, 1))
    plt.plot(np.arange(-90, 91, 1), g)
    plt.show()
