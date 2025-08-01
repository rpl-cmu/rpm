import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from jaxtyping import Float, Array

mmwcas_path = os.path.dirname(os.path.abspath(__file__))


class PatternELE:
    def __init__(
        self, pattern_file=f"{mmwcas_path}/pattern.mat", clip_bound=25, xla=np
    ):
        self.xla = xla
        self.clip_bound = clip_bound
        self.mat = loadmat(pattern_file)["pat"]
        self.mat = xla.asarray(self.mat, dtype=xla.float32)
        self.gain_dB = self.mat[:, 90:271]  # (elevation, azimuth)
        self.gain_dB = self.gain_dB - xla.max(self.gain_dB)
        self.gain_dB_ele = xla.clip(self.gain_dB[:, 90], -50, 10)

    def plotGainELE(self):
        plt.plot(self.xla.arange(-90, 91, 1), self.gain_dB_ele)
        plt.show()

    def gain(self, angle: Float[Array, ""]) -> Float[Array, ""]:
        """interpolate gain from elevation angle"""
        angle = self.xla.clip(angle, -self.clip_bound, self.clip_bound)
        u, d = self.xla.ceil(angle).astype(self.xla.int32), self.xla.floor(
            angle
        ).astype(self.xla.int32)
        ug, dg = self.gain_dB_ele[u + 90], self.gain_dB_ele[d + 90]
        return dg + (angle - d) * (ug - dg)


class PatternAZI:
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

    def plotGainAZI(self):
        plt.plot(self.xla.arange(-90, 91, 1), self.gain_dB_azi)
        plt.show()

    def gain(self, angle: Float[Array, ""]) -> Float[Array, ""]:
        """interpolate gain from azimuth angle. \n
        angle: azimuth angle in degree
        """
        angle = self.xla.clip(angle, -self.clip_bound, self.clip_bound)
        u, d = self.xla.ceil(angle).astype(self.xla.int32), self.xla.floor(
            angle
        ).astype(self.xla.int32)
        ug, dg = self.gain_dB_azi[u + 90], self.gain_dB_azi[d + 90]
        return dg + (angle - d) * (ug - dg)


if __name__ == "__main__":
    pattern = PatternAZI("pattern.mat")
    # pattern = PatternELE("pattern.mat")
    g = pattern.gain(np.arange(-90, 91, 1))
    plt.plot(np.arange(-90, 91, 1), g)
    plt.show()
