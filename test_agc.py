import numpy as np

from ecg_arrhythmia.agc import apply_agc


def test_agc_normalizes_each_window_to_unit_amplitude():
    fs = 10.0
    window_seconds = 1.0
    # dos ventanas de amplitudes claramente distintas: 2.0 y 5.0
    window1 = 2.0 * np.sin(np.linspace(0, 2 * np.pi, 10))
    window2 = 5.0 * np.sin(np.linspace(0, 2 * np.pi, 10))
    signal = np.concatenate([window1, window2])

    agc = apply_agc(signal, fs, window_seconds)

    assert np.isclose(np.max(np.abs(agc[:10])), 1.0, atol=1e-9)
    assert np.isclose(np.max(np.abs(agc[10:])), 1.0, atol=1e-9)


def test_agc_handles_all_zero_window():
    fs = 10.0
    signal = np.zeros(10)
    agc = apply_agc(signal, fs, window_seconds=1.0)
    assert np.all(agc == 0)


def test_agc_last_partial_window():
    fs = 10.0
    signal = np.concatenate([np.ones(10) * 3.0, np.ones(4) * 7.0])
    agc = apply_agc(signal, fs, window_seconds=1.0)
    assert np.allclose(agc[:10], 1.0)
    assert np.allclose(agc[10:], 1.0)
