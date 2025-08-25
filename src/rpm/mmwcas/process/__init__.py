"""Initialization module for the mmwcas.process package."""

from jaxtyping import install_import_hook

with install_import_hook("rpm.mmwcas.process", "beartype.beartype"):
    from .angle_fft import AngleFFT
    from .angle_of_arrival import AOAPC
    from .azimuth_fft import AzimuthFFT
    from .cfar import CFAR
    from .doppler_fft import DopplerFFT
    from .pattern import PatternAZI, PatternELE
    from .pipeline import (
        RadarCubeProc,
        RadarPCProc,
        RangeAzimuthCFARProc,
        RangeAzimuthProc,
        RangeDopplerCFARProc,
        RangeProc,
    )
    from .range_fft import RangeFFT

__all__ = [
    "RangeFFT",
    "DopplerFFT",
    "CFAR",
    "AngleFFT",
    "AzimuthFFT",
    "AOAPC",
    "RangeProc",
    "RangeDopplerCFARProc",
    "RangeAzimuthProc",
    "RangeAzimuthCFARProc",
    "RadarPCProc",
    "RadarCubeProc",
    "PatternELE",
    "PatternAZI",
]
