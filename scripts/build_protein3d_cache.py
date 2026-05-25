import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
AA_LIST = list("ARNDCQEGHILKMFPSTWYV") + ["X"]


def _one_hot(value: str, allowed: list[str]) -> list[float]:
    out = [0.0] * len(allowed)
    out[allowed.index(value if value in allowed else "X")] = 1.0
    return out


def _find_pdb(colabfold_dir: Path, protein_id: str) -> Optional[Path]:
    patterns = [
        f"{protein_id}*rank_001*.pdb",
        f"{protein_id}*rank_1*.pdb",
        f"{protein_id}*ranked_0*.pdb",
        f"{protein_id}*.pdb",
    ]
    for pattern in patterns:
        matches = sorted(colabfold_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _load_pae(colabfold_dir: Path, protein_id: str) -> Optional[np.ndarray]:
    matches = sorted(colabfold_dir.glob(f"{protein_id}*predicted_aligned_error*.json"))
    if not matches:
        matches = sorted(colabfold_dir.glob(f"{protein_id}*pae*.json"))
    if not matches:
        return None
    payload = json.loads(matches[0].read_text(encoding="utf-8"))
    if isinstance(payload, list) and payload:
        payload = payload[0]
    pae = None
    if isinstance(payload, dict):
        pae = payload.get("predicted_aligned_error") or payload.get("pae")
    if pae is None:
        return None
    return np.asarray(pae, dtype=np.float32)


def _unit_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm


def _unit_vectors(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / np.clip(norm, 1e-8, None)


def _dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = _unit_vector(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.arctan2(y, x))


def _rbf(dist: np.ndarray, num_basis: int = 16, cutoff: float = 20.0) -> np.ndarray:
    centers = np.linspace(0.0, cutoff, num_basis, dtype=np.float32)
    gamma = 1.0 / max(float(centers[1] - centers[0]), 1e-6)
    return np.exp(-gamma * (dist[..., None] - centers) ** 2).astype(np.float32)


def _parse_residues(pdb_path: Path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    residues = []
    for residue in structure.get_residues():
        if residue.id[0] != " " or "CA" not in residue:
            continue
        aa = AA3_TO_1.get(residue.resname, "X")
        atoms = {}
        for atom_name in ("N", "CA", "C", "O", "CB"):
            if atom_name in residue:
                atoms[atom_name] = residue[atom_name].coord.astype(np.float32)
        plddt = float(residue["CA"].bfactor) if "CA" in residue else 0.0
        residues.append({"aa": aa, "atoms": atoms, "plddt": plddt})
    return residues


def _build_features(residues: list[dict], k_neighbors: int, cutoff: float, pae: Optional[np.ndarray]):
    ca = np.stack([res["atoms"]["CA"] for res in residues], axis=0).astype(np.float32)
    length = ca.shape[0]
    diff = ca[:, None, :] - ca[None, :, :]
    dist = np.linalg.norm(diff, axis=-1).astype(np.float32)

    phi = np.zeros(length, dtype=np.float32)
    psi = np.zeros(length, dtype=np.float32)
    for idx in range(length):
        atoms = residues[idx]["atoms"]
        if idx > 0 and {"C"} <= residues[idx - 1]["atoms"].keys() and {"N", "CA", "C"} <= atoms.keys():
            phi[idx] = _dihedral(residues[idx - 1]["atoms"]["C"], atoms["N"], atoms["CA"], atoms["C"])
        if idx + 1 < length and {"N", "CA", "C"} <= atoms.keys() and {"N"} <= residues[idx + 1]["atoms"].keys():
            psi[idx] = _dihedral(atoms["N"], atoms["CA"], atoms["C"], residues[idx + 1]["atoms"]["N"])

    curvature = np.zeros(length, dtype=np.float32)
    if length > 2:
        v1 = ca[1:-1] - ca[:-2]
        v2 = ca[2:] - ca[1:-1]
        curvature[1:-1] = np.linalg.norm(_unit_vectors(v2) - _unit_vectors(v1), axis=-1)

    neighbor_count = (dist < 10.0).sum(axis=1).astype(np.float32) - 1.0
    neighbor_density = neighbor_count / max(float(neighbor_count.max()), 1.0)
    asa_proxy = 1.0 - neighbor_density
    rel_pos = np.linspace(0.0, 1.0, length, dtype=np.float32)
    sse_feat = np.zeros((length, 3), dtype=np.float32)
    sse_feat[:, 2] = 1.0

    residue_feat = []
    for idx, res in enumerate(residues):
        feat = []
        feat += _one_hot(res["aa"], AA_LIST)
        feat += [
            res["plddt"] / 100.0,
            rel_pos[idx],
            np.sin(phi[idx]),
            np.cos(phi[idx]),
            np.sin(psi[idx]),
            np.cos(psi[idx]),
            curvature[idx],
            neighbor_density[idx],
            asa_proxy[idx],
        ]
        feat += sse_feat[idx].tolist()
        residue_feat.append(feat)
    residue_feat = np.asarray(residue_feat, dtype=np.float32)

    edges = []
    attrs = []
    for src in range(length):
        order = np.argsort(dist[src])
        candidate = [int(dst) for dst in order[1 : k_neighbors + 1] if dist[src, dst] <= cutoff]
        for dst in candidate:
            edge_dist = dist[src, dst]
            seq_sep = min(abs(src - dst), 512) / 512.0
            pae_value = 0.0
            if pae is not None and src < pae.shape[0] and dst < pae.shape[1]:
                pae_value = min(float(pae[src, dst]) / 30.0, 1.0)
            attrs.append(np.concatenate([_rbf(np.asarray(edge_dist)), np.asarray([seq_sep, pae_value], dtype=np.float32)]))
            edges.append([src, dst])

    if edges:
        edge_index = np.asarray(edges, dtype=np.int64).T
        edge_attr = np.stack(attrs, axis=0).astype(np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, 18), dtype=np.float32)

    return {
        "residue_pos": ca,
        "residue_feat": residue_feat,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "sse_feat": sse_feat,
        "asa_feat": asa_proxy[:, None].astype(np.float32),
        "plddt": np.asarray([res["plddt"] for res in residues], dtype=np.float32),
        "valid_length": np.asarray([length], dtype=np.int64),
    }


def build_cache(index_path: Path, colabfold_dir: Path, cache_dir: Path, k_neighbors: int, cutoff: float) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(index_path)
    rows = []
    for _, row in df.iterrows():
        protein_id = str(row["protein_id"])
        sequence = str(row.get("sequence", "")).strip()
        pdb_path = _find_pdb(colabfold_dir, protein_id)
        if pdb_path is None:
            rows.append(
                {
                    "protein_id": protein_id,
                    "sequence": sequence,
                    "cache_path": "",
                    "status": "missing_pdb",
                    "num_residues": 0,
                }
            )
            continue
        residues = _parse_residues(pdb_path)
        if not residues:
            rows.append(
                {
                    "protein_id": protein_id,
                    "sequence": sequence,
                    "cache_path": "",
                    "status": "empty_pdb",
                    "num_residues": 0,
                }
            )
            continue
        pae = _load_pae(colabfold_dir, protein_id)
        payload = _build_features(residues=residues, k_neighbors=k_neighbors, cutoff=cutoff, pae=pae)
        out_path = cache_dir / f"{protein_id}.npz"
        np.savez_compressed(out_path, **payload)
        rows.append(
            {
                "protein_id": protein_id,
                "sequence": sequence,
                "cache_path": str(out_path),
                "status": "ok",
                "num_residues": int(payload["valid_length"][0]),
                "pdb_path": str(pdb_path),
            }
        )
    out_index = cache_dir.parent / "protein3d_index.csv"
    pd.DataFrame(rows).to_csv(out_index, index=False)
    ok = sum(1 for row in rows if row["status"] == "ok")
    print(f"[OK] cached={ok}/{len(rows)} index={out_index}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build residue-level protein 3D cache from ColabFold outputs.")
    parser.add_argument("--protein_index", default="data/protein3d/protein_index.csv")
    parser.add_argument("--colabfold_dir", default="data/protein3d/colabfold_out")
    parser.add_argument("--cache_dir", default="data/protein3d/cache")
    parser.add_argument("--k_neighbors", type=int, default=16)
    parser.add_argument("--cutoff", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_cache(
        index_path=Path(args.protein_index),
        colabfold_dir=Path(args.colabfold_dir),
        cache_dir=Path(args.cache_dir),
        k_neighbors=args.k_neighbors,
        cutoff=args.cutoff,
    )


if __name__ == "__main__":
    main()
