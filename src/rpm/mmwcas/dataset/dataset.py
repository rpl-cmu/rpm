"""Dataset class for Cascade radar ADC and point cloud data."""

import os
import pickle
from glob import glob
from os.path import join as pjoin
from typing import List

import numpy as np
from jaxtyping import Complex64, Float32, Float64, Int16

from .calibrate import CascadeCalibration
from .params import RadarParam


def extract_stamp(adc_folder: str) -> tuple[Float64[np.ndarray, "n"], List[int]]:
    """Extract frame timestamps from index files.

    Args:
        adc_folder: Path to ADC data folder.

    Returns:
        timeStampFrame: Array of frame timestamps.
        fileFrames: List of number of frames per file.
    """
    flist = glob(pjoin(adc_folder, "master_*_idx.bin"))
    flist.sort()
    timeStampFrame, fileFrames = [], []

    for i_file in range(len(flist)):
        idxFile = open(flist[i_file], "r")
        indexInfo = np.fromfile(idxFile, dtype=np.uint64)
        newStamp = indexInfo[7::6]
        timeStampFrame = np.concatenate((timeStampFrame, newStamp))
        fileFrames.append(newStamp.size)
        idxFile.close()
    timeStampFrame = (timeStampFrame - timeStampFrame[0]) / 1e6
    return timeStampFrame, fileFrames


class CascadeADCDataset:
    """Cascade radar ADC dataset.

    Args:
        folder_path: Path to dataset folder.
        calibrate: Whether to calibrate the ADC data.
        amp_clib: Whether to apply amplitude calibration.
    """

    def __init__(self, folder_path: str, calibrate=True, amp_clib=False):
        sensor = os.path.basename(folder_path)
        self.param = RadarParam(pjoin(folder_path, f"../config/{sensor}.mmwave.json"))
        self.do_calibrate = calibrate
        self.calibrate = CascadeCalibration(
            pjoin(folder_path, f"../config/{sensor}_calib.mat"),
            self.param,
            amp_clib=amp_clib,
        )
        with open(pjoin(folder_path, "../stamp/radar_start.txt"), "r") as file:
            for line in file.readlines():
                if sensor in line:
                    start_time = float(line.split(" ")[-1])

        self.stamps, self.fileFrames = extract_stamp(folder_path)
        self.stamps += start_time  # pyright: ignore[reportPossiblyUnboundVariable]
        self.numFrames = len(self.stamps)
        self.sizePerFramePerDevice = (
            self.param.numTx
            * self.param.numChirp
            * self.param.numRxPerDevice
            * self.param.numADCSample
        ) * 2  # 2 int16 for IQ

        chips = ["master", "slave1", "slave2", "slave3"]
        flist = [sorted(glob(pjoin(folder_path, f"{c}_*_data.bin"))) for c in chips]
        self.adc_memmap = [self.read_single_chip(f) for f in flist]
        self.frame_fidx = np.concatenate(
            [np.ones(self.fileFrames[i]) * i for i in range(len(self.fileFrames))]
        ).astype(int)
        self.frame_offset = np.cumsum([0] + self.fileFrames)

    @staticmethod
    def read_single_chip(flist: List[str]):
        """Read ADC data files for a single chip.

        Args:
            flist: List of file paths for the chip.

        Returns:
            List of memmap arrays for each file.
        """
        return [np.memmap(f, dtype=np.int16, mode="r") for f in flist]

    def to_iq(
        self, adc_data: Int16[np.ndarray, "n"]
    ) -> Complex64[np.ndarray, "Sample Chirp Rx Tx"]:
        """Convert raw ADC data to IQ format.

        Args:
            adc_data: Raw ADC data.

        Returns:
            iq: IQ formatted data.
        """
        iq = adc_data[0::2] + 1j * adc_data[1::2]
        iq = iq.reshape(
            self.param.numRxPerDevice,
            self.param.numADCSample,
            self.param.numTx,
            self.param.numChirp,
            order="F",
        )
        iq = np.transpose(iq, (1, 3, 0, 2))
        iq = iq.astype(np.complex64)
        return iq

    def __len__(self) -> int:
        """Number of frames."""
        return self.numFrames

    def __getitem__(
        self, frame: int | np.int64
    ) -> Complex64[np.ndarray, "Sample Chirp Rx Tx"]:
        """Get single frame.

        Args:
            frame: Frame index.

        Returns:
            adc: ADC data for the specified frame.
        """
        if frame < 0:
            frame += len(self)
        if frame < 0 or frame >= len(self):
            raise IndexError

        fidx = self.frame_fidx[frame]
        offset = (frame - self.frame_offset[fidx]) * self.sizePerFramePerDevice
        seg = [
            chip[fidx][offset : offset + self.sizePerFramePerDevice]
            for chip in self.adc_memmap
        ]
        iq = [self.to_iq(data) for data in seg]
        adc = np.concatenate(iq, axis=2)

        if self.do_calibrate:
            adc = self.calibrate(adc)

        adc = adc[:, :, self.param.RxOrder, :]

        return adc

    def __iter__(self) -> "CascadeADCDataset":
        """Iterator.

        Returns:
           self: The dataset iterator.
        """
        self.idx = 0
        return self

    def __next__(self):
        """Next frame.

        Returns:
            adc: ADC data for the next frame.
        """
        if self.idx < len(self):
            frame = self.idx
            self.idx += 1
            return self[frame]
        else:
            raise StopIteration

    def __str__(self) -> str:
        """String representation.

        Returns:
            info: Information about the dataset.
        """
        info = f"ADC dataset with {len(self)} frames"
        info += f"\nsignal profile:\n {self.param}"
        return info


class PointDataset:
    """Cascade radar point cloud dataset.

    Args:
        folder_path: Path to dataset folder.
    """

    def __init__(self, folder_path: str):
        sensor = os.path.basename(folder_path)
        self.param = RadarParam(pjoin(folder_path, f"../config/{sensor}.mmwave.json"))

        # time stamp
        with open(pjoin(folder_path, "../stamp/radar_start.txt"), "r") as file:
            for line in file.readlines():
                if sensor in line:
                    start_time = float(line.split(" ")[-1])
        self.stamps, self.fileFrames = extract_stamp(folder_path)
        self.stamps += start_time  # pyright: ignore[reportPossiblyUnboundVariable]

        # poses
        path = pjoin(folder_path, f"../pose/pose_{sensor}.pkl")
        with open(path, "rb") as f:
            pose_info = pickle.load(f)
        self.stamp_chirp = pose_info["stamp"]
        self.chirp_poses = pose_info["pose"]
        self.poses = self.chirp_poses[:, 0, 4, ...]  # sensor center chirp 0 tx 4

        # points
        self.point_fs = sorted(glob(pjoin(folder_path, "radar_points", "*.bin")))

    def __len__(self) -> int:
        """Number of frames.

        Returns:
            Number of frames in the dataset.
        """
        return len(self.poses)

    def __getitem__(
        self, idx: int
    ) -> tuple[Float32[np.ndarray, "n 5"], Float32[np.ndarray, "4 4"]]:
        """Get single frame.

        Args:
            idx: Frame index.

        Returns:
            Point cloud data for the specified frame.
            Pose SE3 for the specified frame.
        """
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError

        pcd = np.fromfile(self.point_fs[idx], dtype=np.float32)
        pcd = pcd.reshape(5, -1).T
        rx, ry = pcd[:, 0], pcd[:, 1]
        pcd[:, 0], pcd[:, 1] = ry, -rx
        pcd[:, 2] = -pcd[:, 2]

        return pcd, self.poses[idx]

    def __iter__(self) -> "PointDataset":
        """Iterator.

        Returns:
            self: The dataset iterator.
        """
        self.idx = 0
        return self

    def __next__(self):
        """Next frame.

        Returns:
            Point cloud data for the next frame.
            Pose SE3 for the next frame.
        """
        if self.idx < len(self):
            frame = self.idx
            self.idx += 1
            return self[frame]
        else:
            raise StopIteration

    def __str__(self) -> str:
        """String representation.

        Returns:
            info: Information about the dataset.
        """
        info = f"pc dataset with {len(self)} frames"
        info += f"\nsignal profile:\n {self.param}"
        return info
