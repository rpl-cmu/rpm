import os
import numpy as np
from os.path import join as pjoin
from mmwcas.dataset import MIMODataset


class DualRadarDataset:
    def __init__(
        self,
        folder_path: str,
        rx: list = [],
        tx: list = [],
    ):
        self.r0_dataset = MIMODataset(pjoin(folder_path, "radar0"), rx=rx, tx=tx)
        self.r1_dataset = MIMODataset(pjoin(folder_path, "radar1"), rx=rx, tx=tx)

        self.id_r0, self.id_r1 = 0, 0

    def __len__(self):
        return len(self.r0_dataset) + len(self.r1_dataset)

    def __iter__(self):
        self.idx = 0
        self.id_r0, self.id_r1 = 0, 0
        return self

    def __getitem__(self, i):
        if i >= 0 and i < self.__len__():

            ts_r0 = (
                self.r0_dataset.get_stamp(self.id_r0)
                if self.id_r0 < len(self.r0_dataset)
                else float("inf")
            )
            ts_r1 = (
                self.r1_dataset.get_stamp(self.id_r1)
                if self.id_r1 < len(self.r1_dataset)
                else float("inf")
            )

            if ts_r0 <= ts_r1:
                data = self.r0_dataset[self.id_r0]
                data = (data, "radar0")
                self.id_r0 += 1

            else:
                data = self.r1_dataset[self.id_r1]
                data = (data, "radar1")
                self.id_r1 += 1

            return data

    def __next__(self):
        """next full frame"""
        if self.idx < self.__len__():
            data = self.__getitem__(self.idx)
            self.idx += 1
            return data
        else:
            raise StopIteration
