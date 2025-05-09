import os
import numpy as np


class MergeDataset:
    def __init__(self, dataset0, dataset1):
        self.dataset0 = dataset0
        self.dataset1 = dataset1
        self.id_r0, self.id_r1 = 0, 0

    def __len__(self):
        return len(self.dataset0) + len(self.dataset1)

    def __iter__(self):
        self.idx = 0
        self.id_r0, self.id_r1 = 0, 0
        return self

    def __getitem__(self, i):
        if i >= 0 and i < self.__len__():
            ts_r0 = (
                self.dataset0.get_stamp(self.id_r0)
                if self.id_r0 < len(self.dataset0)
                else float("inf")
            )
            ts_r1 = (
                self.dataset1.get_stamp(self.id_r1)
                if self.id_r1 < len(self.dataset1)
                else float("inf")
            )

            if ts_r0 <= ts_r1:
                data = self.dataset0[self.id_r0]
                data += ("radar0",)
                self.id_r0 += 1
            else:
                data = self.dataset1[self.id_r1]
                data += ("radar1",)
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
