import numpy as np

from ecg_arrhythmia.arrhythmia import analyze_arrhythmias
from ecg_arrhythmia.detection import WavePeaks


def _waves(r_seconds, p_seconds, fs):
    r = (np.array(r_seconds) * fs).astype(int)
    p = np.array([np.nan if v is None else v * fs for v in p_seconds])
    q = r - 5.0
    s = r + 5.0
    return WavePeaks(r=r, q=q, s=s, p=p)


def test_heart_rate_and_tachy_bradycardia():
    fs = 360.0
    # RR: 1.0s -> 60 lpm; 0.5s -> 120 lpm (taqui); 1.2s -> 50 lpm (bradi)
    r_seconds = [0.0, 1.0, 1.5, 2.7]
    p_seconds = [t - 0.16 for t in r_seconds]
    waves = _waves(r_seconds, p_seconds, fs)

    report = analyze_arrhythmias(waves, fs)

    assert report.n_beats == 3
    np.testing.assert_allclose(report.heart_rate_bpm, [60.0, 120.0, 50.0], atol=1e-6)
    assert report.n_tachycardia == 1
    assert report.n_bradycardia == 1


def test_sinus_block_ignores_gaps_from_missing_p_detections():
    fs = 360.0
    r_seconds = [0.0, 1.0, 2.0, 3.0, 4.0]
    # P del segundo latido no se detecto (None): el intervalo PP que la
    # involucraria NO debe entrar en el calculo de bloqueo sinusal.
    p_seconds = [0.0 - 0.16, None, 2.0 - 0.16, 3.0 - 0.16, 4.0 - 0.16]
    waves = _waves(r_seconds, p_seconds, fs)

    report = analyze_arrhythmias(waves, fs)

    # solo hay 2 intervalos PP validos (entre latidos 3-4 y 4-5), ambos de 1s
    assert len(report.pp_intervals_s) == 2
    assert report.n_sinus_block == 0


def test_no_crash_with_single_beat():
    fs = 360.0
    waves = _waves([0.0], [None], fs)
    report = analyze_arrhythmias(waves, fs)
    assert report.n_beats == 0
    assert report.n_tachycardia == 0
    assert report.n_sinus_block == 0
