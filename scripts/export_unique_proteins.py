import argparse
import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_CSVS = [
    "dataset/kcat-km_processed_0.4_data_10fold.csv",
    "dataset/ph_largeset_data_clustered_0.4_5fold.csv",
    "dataset/topt_data_clustered_0.4_5fold.csv",
]


def _protein_hash(sequence: str) -> str:
    return "seq_" + hashlib.sha1(sequence.encode("utf-8")).hexdigest()[:16]


def _clean_uniprot(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return text if text else ""


def _iter_rows(csv_paths: Iterable[Path]):
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if "Sequence" not in df.columns:
            raise ValueError(f"{csv_path} 缺失 Sequence 列")
        for _, row in df.dropna(subset=["Sequence"]).iterrows():
            sequence = str(row["Sequence"]).strip().replace(" ", "")
            if not sequence:
                continue
            yield {
                "source_csv": str(csv_path),
                "sequence": sequence,
                "uniprot_id": _clean_uniprot(row["UniProtID"]) if "UniProtID" in df.columns else "",
            }


def _wrap_fasta(sequence: str, width: int = 80) -> str:
    return "\n".join(sequence[i : i + width] for i in range(0, len(sequence), width))


def export_unique_proteins(csv_paths: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records_by_sequence: dict[str, dict[str, object]] = {}

    for row in _iter_rows(csv_paths):
        sequence = row["sequence"]
        existing = records_by_sequence.setdefault(
            sequence,
            {
                "sequence": sequence,
                "uniprot_ids": set(),
                "source_csvs": set(),
                "source_count": 0,
            },
        )
        existing["source_count"] = int(existing["source_count"]) + 1
        existing["source_csvs"].add(row["source_csv"])
        if row["uniprot_id"]:
            existing["uniprot_ids"].add(row["uniprot_id"])

    rows = []
    fasta_lines = []
    used_ids: dict[str, str] = {}
    for sequence, record in sorted(records_by_sequence.items(), key=lambda item: item[1]["source_count"], reverse=True):
        uniprot_ids = sorted(record["uniprot_ids"])
        protein_id = uniprot_ids[0] if uniprot_ids else _protein_hash(sequence)
        if protein_id in used_ids and used_ids[protein_id] != sequence:
            protein_id = f"{protein_id}_{hashlib.sha1(sequence.encode('utf-8')).hexdigest()[:8]}"
        used_ids[protein_id] = sequence
        source_csvs = sorted(record["source_csvs"])
        rows.append(
            {
                "protein_id": protein_id,
                "uniprot_ids": ";".join(uniprot_ids),
                "source_csvs": ";".join(source_csvs),
                "source_count": int(record["source_count"]),
                "sequence_length": len(sequence),
                "sequence": sequence,
            }
        )
        fasta_lines.append(f">{protein_id}")
        fasta_lines.append(_wrap_fasta(sequence))

    index_path = output_dir / "protein_index.csv"
    fasta_path = output_dir / "unique_proteins.fasta"
    pd.DataFrame(rows).to_csv(index_path, index=False)
    fasta_path.write_text("\n".join(fasta_lines) + "\n", encoding="utf-8")
    print(f"[OK] proteins={len(rows)} index={index_path} fasta={fasta_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export unique protein sequences for ColabFold.")
    parser.add_argument("--csv", nargs="*", default=DEFAULT_CSVS, help="Input CSV files.")
    parser.add_argument("--output_dir", default="data/protein3d", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_paths = [Path(path) for path in args.csv]
    missing = [str(path) for path in csv_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"CSV 不存在: {missing}")
    export_unique_proteins(csv_paths=csv_paths, output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
