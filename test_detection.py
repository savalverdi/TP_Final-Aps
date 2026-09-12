import numpy as np

from ecg_arrhythmia.detection import detect_p_waves, detect_qs_waves, detect_r_peaks, detect_waves


def _gaussian(t, center, width, amp):
    return amp * np.exp(-0.5 * ((t - center) / width) ** 2)


def make_synthetic_ecg(fs=360.0, n_beats=10, beat_period=1.0):
    """Senal ECG sintetica y limpia con P, Q, R, S en posiciones conocidas."""
    duration = n_beats * beat_period + 1.0
    t = np.arange(0, duration, 1 / fs)
    signal = np.zeros_like(t)

    r_times = []
    q_times = []
    s_times = []
    p_times = []

    for i in range(n_beats):
        r_center = 0.5 + i * beat_period
        p_center = r_center - 0.16
        q_center = r_center - 0.02
        s_center = r_center + 0.02

        signal += _gaussian(t, p_center, 0.02, 0.15)
        signal += _gaussian(t, q_center, 0.008, -0.15)
        signal += _gaussian(t, r_center, 0.012, 1.0)
        signal += _gaussian(t, s_center, 0.012, -0.25)

        r_times.append(r_center)
        q_times.append(q_center)
        s_times.append(s_center)
        p_times.append(p_center)

    return t, signal, fs, {
        "r": np.array(r_times),
        "q": np.array(q_times),
        "s": np.array(s_times),
        "p": np.array(p_times),
    }


def test_r_peak_detection_matches_expected_positions():
    t, signal, fs, expected = make_synthetic_ecg()
    r_peaks = detect_r_peaks(signal, fs)

    assert len(r_peaks) == len(expected["r"])
    detected_times = r_peaks / fs
    assert np.allclose(detected_times, expected["r"], atol=2 / fs)


def test_qs_wave_detection_matches_expected_positions():
    t, signal, fs, expected = make_synthetic_ecg()
    r_peaks = detect_r_peaks(signal, fs)
    q_idx, s_idx = detect_qs_waves(signal, fs, r_peaks)

    assert not np.any(np.isnan(q_idx))
    assert not np.any(np.isnan(s_idx))
    assert np.allclose(q_idx / fs, expected["q"], atol=3 / fs)
    assert np.allclose(s_idx / fs, expected["s"], atol=3 / fs)


def test_p_wave_detection_matches_expected_positions():
    t, signal, fs, expected = make_synthetic_ecg()
    r_peaks = detect_r_peaks(signal, fs)
    q_idx, _ = detect_qs_waves(signal, fs, r_peaks)
    p_idx = detect_p_waves(signal, fs, r_peaks, q_idx)

    assert not np.any(np.isnan(p_idx))
    assert np.allclose(p_idx / fs, expected["p"], atol=3 / fs)


def test_p_wave_not_forced_when_absent():
    """Si un latido no tiene onda P real (ej. ectopico), debe quedar NaN, no un valor inventado."""
    t, signal, fs, expected = make_synthetic_ecg(n_beats=5)
    r_peaks = detect_r_peaks(signal, fs)
    q_idx, _ = detect_qs_waves(signal, fs, r_peaks)

    # quitamos la onda P del primer latido restando la misma gaussiana que se sumo
    p_center = expected["p"][0]
    signal_no_p = signal - _gaussian(t, p_center, 0.02, 0.15)

    p_idx = detect_p_waves(signal_no_p, fs, r_peaks, q_idx)
    assert np.isnan(p_idx[0])


def test_detect_waves_end_to_end():
    t, signal, fs, expected = make_synthetic_ecg()
    waves = detect_waves(signal, fs)

    assert len(waves.r) == len(expected["r"])
    assert waves.valid_mask("q").all()
    assert waves.valid_mask("s").all()
    assert waves.valid_mask("p").all()
