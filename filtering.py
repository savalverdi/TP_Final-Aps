"""Filtro pasabanda para la senal ECG (equivalente a butter+filtfilt de MATLAB)."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low_hz: float = 0.5,
    high_hz: float = 30.0,
    order: int = 4,
) -> np.ndarray:
    """Filtro Butterworth pasabanda de fase cero (filtfilt)."""
    nyquist = fs / 2
    b, a = butter(order, [low_hz / nyquist, high_hz / nyquist], btype="bandpass")
    return filtfilt(b, a, signal)
