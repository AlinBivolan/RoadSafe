from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data_loader import (
    RAW_RELEVANT_COLUMNS,
    clean_accidents_dataframe,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curăță CSV-ul brut și păstrează doar coloanele relevante pentru proiect."
    )
    parser.add_argument(
        "--input",
        default=str(ROOT / "data" / "accidents.csv"),
        help="CSV-ul brut de intrare. Implicit: data/accidents.csv",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "accidents_clean.csv"),
        help="CSV-ul curățat de ieșire. Implicit: data/accidents_clean.csv",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Nu există fișierul de intrare: {input_path}")

    header = pd.read_csv(input_path, nrows=0)
    available_cols = [c.strip() for c in header.columns]
    usecols = [c for c in RAW_RELEVANT_COLUMNS if c in available_cols]

    print(f"Citesc: {input_path}")
    print(f"Păstrez coloanele: {', '.join(usecols)}")

    raw_df = pd.read_csv(input_path, usecols=usecols)
    clean_df = clean_accidents_dataframe(raw_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output_path, index=False)

    print(f"CSV curățat salvat în: {output_path}")
    print(f"Rânduri salvate: {len(clean_df)}")
    print(f"Coloane salvate: {len(clean_df.columns)}")
    print("Gata. Aplicația va folosi automat data/accidents_clean.csv dacă există.")


if __name__ == "__main__":
    main()
