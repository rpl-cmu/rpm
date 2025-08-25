"""Defines the antenna layout for the radar."""

import numpy as np
from jaxtyping import Float32, Int

from .params import RadarParam


class AntennaLayout:
    """Defines the antenna layout for the radar.

    Args:
        param: RadarParam object containing radar parameters.
    """

    def __init__(self, param: RadarParam) -> None:
        wavelength = param.speedOfLight / param.cascade_antenna_designFreq

        center = np.array([16, 0, 0])
        rx_offset = np.array([-15, 34, 0])

        tx_pos = np.stack((param.dTx_azi, param.dTx_ele, np.zeros(12)), axis=-1)
        rx_pos = np.stack((param.dRX_azi, param.dRX_ele, np.zeros(16)), axis=-1)

        tx_pos = tx_pos - center
        rx_pos = rx_pos - center + rx_offset
        tx_pos = tx_pos * wavelength / 2
        rx_pos = rx_pos * wavelength / 2

        T = np.eye(4)
        T[:3, :3] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.tx_T = np.repeat(T[None, ...], 12, axis=0)
        self.rx_T = np.repeat(T[None, ...], 16, axis=0)
        self.tx_T[:, 1, 3], self.tx_T[:, 2, 3] = -tx_pos[:, 0], -tx_pos[:, 1]
        self.rx_T[:, 1, 3], self.rx_T[:, 2, 3] = -rx_pos[:, 0], -rx_pos[:, 1]

    def get_tx_transform(
        self, pose: Float32[np.ndarray, "4 4"], tx: Int[np.ndarray, "n"]
    ) -> Float32[np.ndarray, "... n 4 4"]:
        return pose[..., None, :, :] @ self.tx_T[tx][None, ...]

    def get_rx_transform(
        self, pose: Float32[np.ndarray, "... 4 4"], rx: Int[np.ndarray, "n"]
    ) -> Float32[np.ndarray, "... n 4 4"]:
        return pose[..., None, :, :] @ self.rx_T[rx][None, ...]
