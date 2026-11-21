"""Calibra sy e pixel_size da integração a partir do ground truth de um dataset.

Uso:
    python scripts/calibrate_dataset.py --data_dir data/raw/hybrid_stereo_new/lamb-gls
    python scripts/calibrate_dataset.py --data_dir <dataset> \\
        --results_dir data/results/hybrid_stereo/<run>   # testa também as normais do run

Imprime as medições e um trecho de YAML para colar em hybrid.integration.
Detalhes do método: hybrid_stereo_method/evaluation/calibration.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile

from hybrid_stereo_method.evaluation.calibration import calibrate, format_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_dir", required=True, help="Pasta do dataset (com sharp/ e zf*/shrp.png).")
    parser.add_argument(
        "--results_dir", default=None, help="Run do pipeline, para testar também as normais estimadas."
    )
    parser.add_argument("--json", default=None, help="Salva o relatório em JSON neste caminho.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    with tempfile.TemporaryDirectory(prefix="calibrate_") as workdir:
        report = calibrate(args.data_dir, workdir, args.results_dir)
    print(format_report(report))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2, default=float)


if __name__ == "__main__":
    main()
