"""Deteccion de las ondas R, Q, S y P sobre la senal con AGC aplicado.

Es un port del algoritmo original de MATLAB (findpeaks + busqueda de
minimos/maximos locales en ventanas fijas), con dos mejoras dirigidas a
los problemas que el propio trabajo reporto en su seccion de
Discusion:

1. Bug de indices en 0: en MATLAB, cuando un latido esta demasiado
   cerca del inicio de la senal para tener ventana de busqueda de Q/P,
   el codigo original deja el indice en 0, lo que en MATLAB puede
   hacer fallar el `plot` posterior (indice invalido). Aqui se usa
   `NaN` explicito para "onda no encontrada", y quien consuma los
   resultados decide que hacer (excluir del grafico, del calculo de
   intervalos, etc.) en vez de fallar o graficar un punto erroneo en
   el origen.

2. Deteccion "ciega" de la onda P: el original toma el maximo de la
   ventana sin verificar que sea un pico real, por lo que en
   grabaciones con ruido o latidos ectopicos (ej. PVC, que no tienen
   onda P) el algoritmo igual "detecta" un punto cualquiera. Aca se
   exige una prominencia minima real (`find_peaks`); si no hay un pico
   genuino, la onda P se marca como no detectada en vez de inventar
   una posicion. Esto es justamente lo que se observo en las
   grabaciones 107-109 y 116-118 del trabajo original.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


@dataclass
class WavePeaks:
    r: np.ndarray  # indices de muestra (int), siempre presentes
    q: np.ndarray  # indices (float, NaN si no se detecto), mismo largo que r
    s: np.ndarray  # indices (float, NaN si no se detecto), mismo largo que r
    p: np.ndarray  # indices (float, NaN si no se detecto), mismo largo que r

    def valid_mask(self, key: str) -> np.ndarray:
        return ~np.isnan(getattr(self, key))

    def valid_indices(self, key: str) -> np.ndarray:
        arr = getattr(self, key)
        return arr[self.valid_mask(key)].astype(int)


def detect_r_peaks(
    signal: np.ndarray,
    fs: float,
    min_height: float = 0.4,
    min_distance_s: float = 0.2,
) -> np.ndarray:
    """Equivalente a findpeaks(ecg_agc, 'MinPeakHeight', ..., 'MinPeakDistance', ...)."""
    distance = max(1, int(round(min_distance_s * fs)))
    peaks, _ = find_peaks(signal, height=min_height, distance=distance)
    return peaks


def detect_qs_waves(
    signal: np.ndarray,
    fs: float,
    r_peaks: np.ndarray,
    s_window_s: float = 0.15,
    q_window_s: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Onda S: minimo tras cada R. Onda Q: minimo antes de cada R."""
    n = len(signal)
    s_window = max(1, int(round(s_window_s * fs)))
    q_window = max(1, int(round(q_window_s * fs)))

    q_idx = np.full(len(r_peaks), np.nan)
    s_idx = np.full(len(r_peaks), np.nan)

    for i, r in enumerate(r_peaks):
        s_end = min(r + s_window, n - 1)
        s_idx[i] = r + np.argmin(signal[r : s_end + 1])

        if r > q_window:
            q_start = max(0, r - q_window)
            q_idx[i] = q_start + np.argmin(signal[q_start : r + 1])
        # si r <= q_window no hay espacio suficiente antes: se deja NaN

    return q_idx, s_idx


def detect_p_waves(
    signal: np.ndarray,
    fs: float,
    r_peaks: np.ndarray,
    q_idx: np.ndarray,
    base_window_s: float = 0.2,
    min_prominence: float = 0.02,
) -> np.ndarray:
    """Onda P: pico real (con prominencia minima) antes de cada onda Q.

    La ventana de busqueda se acota tambien por el intervalo RR previo
    (40% del RR anterior) para no invadir la onda T del latido
    anterior en frecuencias cardiacas altas.
    """
    base_window = max(1, int(round(base_window_s * fs)))
    p_idx = np.full(len(r_peaks), np.nan)

    for i, q in enumerate(q_idx):
        if np.isnan(q) or q <= 0:
            continue
        q = int(q)

        window = base_window
        if i > 0:
            rr_prev = r_peaks[i] - r_peaks[i - 1]
            window = min(window, max(1, int(0.4 * rr_prev)))

        start = max(0, q - window)
        if start >= q:
            continue

        segment = signal[start : q + 1]
        local_peaks, props = find_peaks(segment, prominence=min_prominence)
        if len(local_peaks) == 0:
            continue  # no hay un pico genuino: onda P no detectada (ej. latido ectopico)

        best = local_peaks[np.argmax(props["prominences"])]
        p_idx[i] = start + best

    return p_idx


def detect_waves(
    signal: np.ndarray,
    fs: float,
    min_r_height: float = 0.4,
    min_r_distance_s: float = 0.2,
    s_window_s: float = 0.15,
    q_window_s: float = 0.2,
    p_window_s: float = 0.2,
    p_min_prominence: float = 0.02,
) -> WavePeaks:
    r_peaks = detect_r_peaks(signal, fs, min_r_height, min_r_distance_s)
    q_idx, s_idx = detect_qs_waves(signal, fs, r_peaks, s_window_s, q_window_s)
    p_idx = detect_p_waves(signal, fs, r_peaks, q_idx, p_window_s, p_min_prominence)
    return WavePeaks(r=r_peaks, q=q_idx, s=s_idx, p=p_idx)
