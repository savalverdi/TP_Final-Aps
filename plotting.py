"""Graficos de la senal ECG con las ondas P, Q, R y S marcadas."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .detection import WavePeaks


def plot_waves(
    time: np.ndarray,
    signal: np.ndarray,
    waves: WavePeaks,
    xlim: tuple[float, float] | None = (1180, 1200),
    ylim: tuple[float, float] | None = (-0.4, 1.0),
    title: str = "Deteccion de ondas P, Q, R y S en el ECG",
    save_path: str | None = None,
):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, signal, label="ECG", linewidth=0.8)

    ax.plot(time[waves.r], signal[waves.r], "*r", label="Onda R")

    s_idx = waves.valid_indices("s")
    ax.plot(time[s_idx], signal[s_idx], "*k", label="Onda S")

    q_idx = waves.valid_indices("q")
    ax.plot(time[q_idx], signal[q_idx], "*g", label="Onda Q")

    p_idx = waves.valid_indices("p")
    ax.plot(time[p_idx], signal[p_idx], "*m", label="Onda P")

    ax.set_title(title)
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Voltaje [mV]")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, ax
