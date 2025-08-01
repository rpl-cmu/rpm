from ast import literal_eval
import numpy as np
import json
import re


class RadarParam:
    def __init__(self, config_file) -> None:
        with open(config_file) as f:
            config = json.load(f)
            device = config["mmWaveDevices"][0]
            profile = device["rfConfig"]["rlProfiles"][0]["rlProfileCfg_t"]
            frameCfg = device["rfConfig"]["rlFrameCfg_t"]

        # signal
        to_param = lambda x, s: float(f"{x*s:e}")
        self.numADCSample = int(profile["numAdcSamples"])
        self.adcSampleRate = to_param(profile["digOutSampleRate"], 1e3)  # Hz/s
        self.startFreqConst = to_param(profile["startFreqConst_GHz"], 1e9)  # Hz
        self.chirpSlope = to_param(profile["freqSlopeConst_MHz_usec"], 1e12)  # Hz/s
        self.chirpIdleTime = to_param(profile["idleTimeConst_usec"], 1e-6)  # s
        self.adcStartTimeConst = to_param(profile["adcStartTimeConst_usec"], 1e-6)  # s
        self.chirpRampEndTime = to_param(profile["rampEndTime_usec"], 1e-6)  # s
        self.RxGain_dB = float(literal_eval(profile["rxGain_dB"]))
        self.speedOfLight = 3e8

        # frame
        self.numTx = 12
        self.numRxPerDevice = 4
        self.numDevice = 4
        self.numRx = self.numRxPerDevice * self.numDevice
        self.numChirp = int(frameCfg["numLoops"])
        self.numChirpPerFrame = self.numChirp * self.numTx
        self.numChirpsPerVirAnt = self.numChirp

        # chirp
        self.chirpRampTime = self.numADCSample / self.adcSampleRate
        self.chirpBandwidth = self.chirpSlope * self.chirpRampTime  # Hz
        self.chirpInterval = self.chirpRampEndTime + self.chirpIdleTime
        self.carrierFrequency = (
            self.startFreqConst
            + (self.adcStartTimeConst + self.chirpRampTime / 2) * self.chirpSlope
        )
        self.centerFrequency = (
            self.startFreqConst
            + self.numADCSample / self.adcSampleRate * self.chirpSlope / 2
        )
        self.cascade_antenna_designFreq = 76.8e9
        self.antenna_dis = 0.5 * self.centerFrequency / self.cascade_antenna_designFreq

        # antenna layout
        # ref: https://www.ti.com/lit/ug/swru553a/swru553a.pdf
        self.RxOrder = [12, 13, 14, 15, 0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7]
        self.TxOrder = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        cascade_tx_position_azi = np.array([11, 10, 9, 32, 28, 24, 20, 16, 12, 8, 4, 0])
        cascade_tx_position_ele = np.array([6, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        cascade_rx_position_azi = np.array(
            ([11, 12, 13, 14] + [50, 51, 52, 53] + [46, 47, 48, 49] + [0, 1, 2, 3])
        )
        cascade_rx_position_ele = np.array([0 for _ in range(16)])
        self.dTx_azi = cascade_tx_position_azi[self.TxOrder]
        self.dTx_ele = cascade_tx_position_ele[self.TxOrder]
        self.dRX_azi = cascade_rx_position_azi[self.RxOrder]
        self.dRX_ele = cascade_rx_position_ele[self.RxOrder]
        self.antenna_azi = np.sum(
            np.meshgrid(self.dRX_azi, self.dTx_azi, indexing="ij"),
            axis=0,
            dtype=np.int32,
        )
        self.antenna_ele = np.sum(
            np.meshgrid(self.dRX_ele, self.dTx_ele, indexing="ij"),
            axis=0,
            dtype=np.int32,
        )

        # range
        self.maxRange = (
            self.speedOfLight
            * self.adcSampleRate
            * self.chirpRampTime
            / (2 * self.chirpBandwidth)
        )
        self.numSamplePerChirp = round(self.chirpRampTime * self.adcSampleRate)
        self.rangeFFTSize = int(2 ** (np.ceil(np.log2(self.numSamplePerChirp))))
        self.rangeResolution = self.speedOfLight / 2 / self.chirpBandwidth
        self.rangeBinSize = (
            self.rangeResolution * self.numSamplePerChirp / self.rangeFFTSize
        )

        # velocity
        self.lambda_ = self.speedOfLight / self.carrierFrequency
        self.velocityResolution = self.lambda_ / (
            2 * self.numChirp * self.chirpInterval * self.numTx
        )
        self.DopplerFFTSize = int(2 ** (np.ceil(np.log2(self.numChirp))))
        self.velocityBinSize = (
            self.velocityResolution * self.numChirpsPerVirAnt / self.DopplerFFTSize
        )

    def __str__(self) -> str:
        dict_info = self.__dict__.copy()
        for key in list(dict_info.keys()):
            if isinstance(dict_info[key], np.ndarray):
                dict_info.pop(key)

        repl_func = lambda match: " ".join(match.group().split())
        info = json.dumps(dict_info, indent=4)
        info = re.sub(r"(?<=\[)[^\[\]]+(?=])", repl_func, info)
        return info

    @staticmethod
    def get_freq(
        distance: np.ndarray, chirpSlope: float, speedOfLight: float = 3e8
    ) -> np.ndarray:
        return chirpSlope * 2 * distance / speedOfLight

    @staticmethod
    def get_phase(
        distance: np.ndarray,
        startFreqConst: float,
        chirpSlope: float,
        speedOfLight: float = 3e8,
    ) -> np.ndarray:
        return startFreqConst * 2 * distance / speedOfLight - 2 * chirpSlope * (
            (distance / speedOfLight) ** 2
        )
