import numpy as np

from ecg_arrhythmia.filtering import bandpass_filter


def test_bandpass_removes_dc_and_low_freq_drift():
    fs = 360.0
    t = np.arange(0, 10, 1 / fs)
    drift = 0.5 * np.sin(2 * np.pi * 0.05 * t)  # 0.05 Hz, fuera de banda
    signal_in_band = np.sin(2 * np.pi * 5 * t)  # 5 Hz, dentro de banda
    raw = drift + signal_in_band + 2.0  # + DC

    filtered = bandpass_filter(raw, fs)

    assert abs(np.mean(filtered)) < 0.05
    # la componente dentro de banda debe conservar la mayor parte de su energia
    assert np.std(filtered) > 0.5 * np.std(signal_in_band)


def test_bandpass_attenuates_high_freq_noise():
    fs = 360.0
    t = np.arange(0, 5, 1 / fs)
    low = np.sin(2 * np.pi * 5 * t)
    noise = 0.8 * np.sin(2 * np.pi * 100 * t)  # 100 Hz, fuera de banda
    raw = low + noise

    filtered = bandpass_filter(raw, fs)

    # la potencia deberia acercarse a la de la senal limpia, no a la de raw+noise
    assert np.std(filtered - low) < np.std(noise)
