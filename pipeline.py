"""Pipeline completo: carga -> filtrado -> AGC -> deteccion de ondas -> arritmias."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .agc import apply_agc
from .arrhythmia import ArrhythmiaReport, analyze_arrhythmias
from .data import EcgRecord, load_record
from .detection import WavePeaks, detect_waves
from .filtering import bandpass_filter


@dataclass
class AnalysisResult:
    record: EcgRecord
    filtered_signal: np.ndarray
    agc_signal: np.ndarray
    waves: WavePeaks
    report: ArrhythmiaReport


def analyze_record(
    record_name: str = "103",
    lead: str = "MLII",
    pn_dir: str = "mitdb",
    bandpass_low_hz: float = 0.5,
    bandpass_high_hz: float = 30.0,
    bandpass_order: int = 4,
    agc_window_s: float = 5.0,
    min_r_height: float = 0.4,
    min_r_distance_s: float = 0.2,
    s_window_s: float = 0.15,
    q_window_s: float = 0.2,
    p_window_s: float = 0.2,
    p_min_prominence: float = 0.02,
    tachycardia_threshold_bpm: float = 100.0,
    bradycardia_threshold_bpm: float = 60.0,
    sinus_block_ratio: float = 1.5,
) -> AnalysisResult:
    record = load_record(record_name=record_name, lead=lead, pn_dir=pn_dir)

    filtered = bandpass_filter(
        record.signal, record.fs, bandpass_low_hz, bandpass_high_hz, bandpass_order
    )
    agc_signal = apply_agc(filtered, record.fs, agc_window_s)

    waves = detect_waves(
        agc_signal,
        record.fs,
        min_r_height=min_r_height,
        min_r_distance_s=min_r_distance_s,
        s_window_s=s_window_s,
        q_window_s=q_window_s,
        p_window_s=p_window_s,
        p_min_prominence=p_min_prominence,
    )

    report = analyze_arrhythmias(
        waves,
        record.fs,
        tachycardia_threshold_bpm=tachycardia_threshold_bpm,
        bradycardia_threshold_bpm=bradycardia_threshold_bpm,
        sinus_block_ratio=sinus_block_ratio,
    )

    return AnalysisResult(
        record=record,
        filtered_signal=filtered,
        agc_signal=agc_signal,
        waves=waves,
        report=report,
    )
