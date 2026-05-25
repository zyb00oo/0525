from typing import Optional

from rdkit import Chem


def smiles_to_mol_safe(smiles: str) -> Optional[Chem.Mol]:
    """
    Parse SMILES with conservative cleanup to avoid unstable RDKit warning paths.
    """
    if not isinstance(smiles, str):
        return None
    smiles = smiles.strip()
    if not smiles:
        return None

    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    # Remove detached hydrogen fragments (can trigger RDKit RemoveHs warnings).
    detached_h = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 1 and atom.GetDegree() == 0 and mol.GetNumAtoms() > 1
    ]
    if detached_h:
        rw_mol = Chem.RWMol(mol)
        for idx in sorted(detached_h, reverse=True):
            rw_mol.RemoveAtom(idx)
        mol = rw_mol.GetMol()

    sanitize_ops = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS
    try:
        Chem.SanitizeMol(mol, sanitizeOps=sanitize_ops)
    except Exception:
        return None
    return mol
