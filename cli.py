"""CLI: python -m ecg_arrhythmia --record 103 --lead MLII --xlim 1180 1200"""

from __future__ import annotations

import argparse
from pathlib import Path

from .arrhythmia import format_report
from .pipeline import analyze_record
from .plotting import plot_waves


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", default="103", help="Nombre del registro MIT-BIH (ej. 103)")
    parser.add_argument("--lead", default="MLII", help="Derivacion a usar (ej. MLII)")
    parser.add_argument("--pn-dir", default="mitdb", help="Base de datos de PhysioNet")
    parser.add_argument(
        "--xlim", type=float, nargs=2, default=(1180, 1200), metavar=("INICIO", "FIN"),
        help="Ventana de tiempo (s) a graficar",
    )
    parser.add_argument("--no-plot", action="store_true", help="No generar el grafico")
    parser.add_argument(
        "--output", default="output", help="Carpeta donde guardar el grafico (PNG)"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = analyze_record(record_name=args.record, lead=args.lead, pn_dir=args.pn_dir)
    print(format_report(result.report, record_name=f"{args.record} ({args.lead})"))

    if not args.no_plot:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"{args.record}_{args.lead}_waves.png"
        plot_waves(
            result.record.time,
            result.agc_signal,
            result.waves,
            xlim=tuple(args.xlim),
            save_path=str(save_path),
        )
        print(f"Grafico guardado en: {save_path}")


if __name__ == "__main__":
    main()
