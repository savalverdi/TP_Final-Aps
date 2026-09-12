"""Regresion contra los resultados publicados por el script original de MATLAB
(LecturaECG-1.pdf, pagina 3) para el registro 103, derivacion MLII:

    Cantidad de latidos (RR intervals): 2080
    Frecuencia cardiaca media: 69.41 l.p.m.
    Casos de taquicardia detectados: 0
    Casos de bradicardia detectados: 4
    Cantidad de bloqueos sinusales: 1
"""

from ecg_arrhythmia.pipeline import analyze_record


def test_record_103_matches_original_matlab_output():
    result = analyze_record(record_name="103", lead="MLII")
    report = result.report

    assert report.n_beats == 2080
    assert round(report.mean_heart_rate_bpm, 2) == 69.41
    assert report.n_tachycardia == 0
    assert report.n_bradycardia == 4
    assert report.n_sinus_block == 1
