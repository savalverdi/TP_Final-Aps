"""Control automatico de ganancia (AGC) por ventanas, igual que el original.

Normaliza la amplitud de la senal en bloques no solapados de
`window_seconds` segundos, dividiendo cada bloque por su valor
absoluto maximo (o dejandolo intacto si el bloque es todo cero).
"""

from __future__ import annotations

import numpy as np


def apply_agc(signal: np.ndarray, fs: float, window_seconds: float = 5.0) -> np.ndarray:
    window_size = int(window_seconds * fs)
    if window_size <= 0:
        raise ValueError("window_seconds * fs debe ser positivo")

    agc_signal = np.empty_like(signal)
    n = len(signal)
    for start in range(0, n, window_size):
        end = min(start + window_size, n)
        window = signal[start:end]
        amplitud_max = np.max(np.abs(window))
        ganancia = 1.0 / amplitud_max if amplitud_max > 0 else 1.0
        agc_signal[start:end] = window * ganancia
    return agc_signal
