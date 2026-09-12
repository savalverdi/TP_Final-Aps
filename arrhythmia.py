"""Calculo de intervalos RR/PP, frecuencia cardiaca y arritmias.

Port de la seccion final del script de MATLAB. Una diferencia
deliberada respecto al original: los intervalos PP solo se calculan
entre ondas P *consecutivas y efectivamente detectadas*. El script de
MATLAB hacia `diff(t(picos_P))` sobre el vector completo, por lo que
si una onda P no se detectaba (dejando un 0 "de relleno"), el
intervalo PP calculado ahi quedaba corrupto y podia disparar un falso
"bloqueo sinusal". Aqui, si a alguno de los dos latidos consecutivos
le falta la onda P detectada, ese intervalo simplemente no entra en el
calculo (no se inventa un numero).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .detection import WavePeaks


@dataclass
class ArrhythmiaReport:
    n_beats: int
    rr_intervals_s: np.ndarray
    heart_rate_bpm: np.ndarray
    mean_heart_rate_bpm: float
    tachycardia_mask: np.ndarray
    bradycardia_mask: np.ndarray
    pp_intervals_s: np.ndarray
    sinus_block_mask: np.ndarray
    n_p_waves_detected: int = field(default=0)

    @property
    def n_tachycardia(self) -> int:
        return int(np.sum(self.tachycardia_mask))

    @property
    def n_bradycardia(self) -> int:
        return int(np.sum(self.bradycardia_mask))

    @property
    def n_sinus_block(self) -> int:
        return int(np.sum(self.sinus_block_mask))


def analyze_arrhythmias(
    waves: WavePeaks,
    fs: float,
    tachycardia_threshold_bpm: float = 100.0,
    bradycardia_threshold_bpm: float = 60.0,
    sinus_block_ratio: float = 1.5,
) -> ArrhythmiaReport:
    r_times = waves.r / fs
    rr = np.diff(r_times)
    heart_rate = 60.0 / rr if len(rr) else np.array([])
    mean_hr = float(np.mean(heart_rate)) if len(heart_rate) else float("nan")

    tachycardia_mask = heart_rate > tachycardia_threshold_bpm
    bradycardia_mask = heart_rate < bradycardia_threshold_bpm

    p_times = waves.p / fs  # NaN donde no se detecto
    both_valid = ~np.isnan(p_times[:-1]) & ~np.isnan(p_times[1:]) if len(p_times) > 1 else np.array([], dtype=bool)
    pp = (p_times[1:] - p_times[:-1])[both_valid] if len(p_times) > 1 else np.array([])

    if len(pp):
        sinus_block_mask = pp > np.mean(pp) * sinus_block_ratio
    else:
        sinus_block_mask = np.array([], dtype=bool)

    return ArrhythmiaReport(
        n_beats=len(rr),
        rr_intervals_s=rr,
        heart_rate_bpm=heart_rate,
        mean_heart_rate_bpm=mean_hr,
        tachycardia_mask=tachycardia_mask,
        bradycardia_mask=bradycardia_mask,
        pp_intervals_s=pp,
        sinus_block_mask=sinus_block_mask,
        n_p_waves_detected=int(np.sum(~np.isnan(waves.p))),
    )


def format_report(report: ArrhythmiaReport, record_name: str = "") -> str:
    header = f"Resultados ({record_name})" if record_name else "Resultados"
    lines = [
        header,
        f"Cantidad de latidos (RR intervals): {report.n_beats}",
        f"Frecuencia cardiaca media: {report.mean_heart_rate_bpm:.2f} l.p.m.",
        f"Casos de taquicardia detectados: {report.n_tachycardia}",
        f"Casos de bradicardia detectados: {report.n_bradycardia}",
        f"Cantidad de bloqueos sinusales: {report.n_sinus_block}",
        f"Ondas P detectadas: {report.n_p_waves_detected} de {report.n_beats + 1} latidos",
    ]
    return "\n".join(lines)
