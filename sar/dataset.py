import os
import pickle
import numpy as np
from os.path import join as pjoin

from mmwcas.dataset import CascadeADCDataset, AntennaLayout


class ChirpPoseDataset:
    def __init__(
        self,
        folder_path: str,
        calibrate: bool = True,
        amp_clib: bool = True,
    ):

        sensor = os.path.basename(folder_path)
        self.adc = CascadeADCDataset(
            folder_path, calibrate=calibrate, amp_clib=amp_clib
        )
        self.stamp_frame = self.adc.stamps
        path = pjoin(folder_path, f"../pose/pose_{sensor}.pkl")
        with open(path, "rb") as f:
            pose_info = pickle.load(f)
        self.stamp_chirp = pose_info["stamp"]
        self.chirp_poses = pose_info["pose"]

        assert len(self.chirp_poses) == len(self.stamp_frame) == len(self.adc)
        self.fps = round(1 / np.diff(self.stamp_frame).mean())

        # start with sensor moving
        start_idx = 0
        for i in range(1, len(self.chirp_poses)):
            start_idx, p = i, self.chirp_poses[i, 0, 0, :3, 3]
            d = np.linalg.norm(p - self.chirp_poses[0, 0, 0, :3, 3])
            if d > 0.1:
                break

        self.frame_idx = np.arange(start_idx, len(self.chirp_poses) - 1)

    def __len__(self):
        return len(self.frame_idx)

    def __iter__(self):
        self.idx = 0
        return self

    def __getitem__(self, i):
        if i >= 0 and i < self.__len__():
            chirps = self.adc[self.frame_idx[i]]
            chirps = np.moveaxis(chirps, (0, 1, 2, 3), (3, 0, 1, 2))
            poses = self.chirp_poses[self.frame_idx[i]]
            # poses[..., 2, 3] = 0  # z=0

            return chirps, poses

    def __next__(self):
        """next full frame"""
        if self.idx < len(self.frame_idx):
            chirps, poses = self.__getitem__(self.idx)

            self.idx += 1
            return chirps, poses
        else:
            raise StopIteration


class MIMODataset(ChirpPoseDataset):
    """Multi Input Multi Output Dataset

    Data was collected with 16 rx and 12 tx, but user can specify a subset of rx and tx here.
    """

    def __init__(
        self,
        folder_path: str,
        rx: list = None,
        tx: list = None,
    ):
        super().__init__(folder_path, calibrate=True, amp_clib=True)
        self.layout = AntennaLayout(self.adc.param)
        self.rx = np.arange(self.adc.param.numRx) if rx is None else rx
        self.tx = np.arange(self.adc.param.numTx) if tx is None else tx

        print(
            "synthetic antennas: ",
            len(self) * self.adc.param.numChirp * len(self.rx) * len(self.tx),
        )

    def __getitem__(self, i):
        if i >= 0 and i < self.__len__():
            chirps = self.adc[self.frame_idx[i]]
            chirps = chirps[:, :, self.rx, :][:, :, :, self.tx]
            chirps = np.moveaxis(chirps, (0, 1, 2, 3), (3, 0, 1, 2))
            # chirps: [chirp, rx, tx, sample]

            poses = self.chirp_poses[self.frame_idx[i]]
            # poses[..., 2, 3] = 0

            pose_tx = self.layout.get_tx_T(poses, self.tx)
            pose_rx = self.layout.get_rx_T(poses, self.rx)

            # select activated tx and repeated for all rx
            pose_tx = pose_tx[:, self.tx, self.tx, :, :]
            pose_tx = np.repeat(pose_tx[:, None], len(self.rx), axis=1)

            # select rx w.r.t. activated tx
            pose_rx = pose_rx[:, self.tx, ...]
            pose_rx = np.moveaxis(pose_rx, (1, 2), (2, 1))

            return chirps, pose_tx, pose_rx

    def __next__(self):
        """next full frame"""
        if self.idx < len(self.frame_idx):
            chirps, pose_tx, pose_rx = self.__getitem__(self.idx)

            self.idx += 1
            return chirps, pose_tx, pose_rx
        else:
            raise StopIteration
