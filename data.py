"""Carga de senales ECG.

El .mat original (103m_MLII.mat) no estaba disponible, asi que las
senales se descargan directamente de la MIT-BIH Arrhythmia Database en
PhysioNet mediante `wfdb`, que es la fuente que el propio trabajo cita
como origen de los datos. Cada registro se cachea localmente en
`data/` para no volver a descargarlo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class EcgRecord:
    record_name: str
    lead: str
    fs: float
    signal: np.ndarray  # mV
    time: np.ndarray  # segundos

    def __len__(self) -> int:
        return len(self.signal)


def load_record(
    record_name: str = "103",
    lead: str = "MLII",
    pn_dir: str = "mitdb",
    cache_dir: Path | str = DATA_DIR,
) -> EcgRecord:
    """Carga un registro de ECG (por defecto de MIT-BIH) para una derivacion dada.

    Usa un cache local en `cache_dir/<record_name>.npz`; si no existe,
    descarga el registro de PhysioNet con wfdb.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{record_name}_{lead}.npz"

    if cache_file.exists():
        cached = np.load(cache_file)
        fs = float(cached["fs"])
        signal = cached["signal"]
    else:
        record = wfdb.rdrecord(record_name, pn_dir=pn_dir)
        if lead not in record.sig_name:
            raise ValueError(
                f"La derivacion '{lead}' no esta en el registro {record_name} "
                f"(disponibles: {record.sig_name})"
            )
        channel = record.sig_name.index(lead)
        signal = record.p_signal[:, channel].astype(np.float64)
        fs = float(record.fs)
        np.savez(cache_file, signal=signal, fs=fs)

    time = np.arange(len(signal)) / fs
    return EcgRecord(record_name=record_name, lead=lead, fs=fs, signal=signal, time=time)
