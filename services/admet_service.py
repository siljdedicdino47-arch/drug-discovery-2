"""
ADMET prediction service using RDKit descriptors.
Predicts drug-likeness properties: Absorption, Distribution,
Metabolism, Excretion, Toxicity.
"""
from typing import Optional

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, QED, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def validate_smiles(smiles: str) -> bool:
    """Return True if the SMILES string is chemically valid."""
    if not RDKIT_AVAILABLE:
        return True  # Skip validation in envs without rdkit
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def predict_admet(smiles: str) -> dict:
    """
    Compute a suite of ADMET-relevant molecular descriptors from SMILES.
    Returns a structured dict of properties with interpretations.
    """
    if not RDKIT_AVAILABLE:
        return {"error": "RDKit not installed. Install with: pip install rdkit"}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES: {smiles}"}

    mw = Descriptors.ExactMolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rot_bonds = Lipinski.NumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()
    fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    qed_score = QED.qed(mol)

    # Lipinski Rule of Five evaluation
    ro5_violations = sum([
        mw > 500,
        logp > 5,
        hbd > 5,
        hba > 10,
    ])

    # Veber rules (oral bioavailability)
    veber_pass = rot_bonds <= 10 and tpsa <= 140

    # Blood-brain barrier estimate (simplified CNS MPO heuristic)
    bbb_score = _estimate_bbb(mw, logp, tpsa, hbd)

    # Solubility estimate (ESOL-like)
    log_sol = _estimate_solubility(logp, mw, rot_bonds, aromatic_rings)

    return {
        "smiles": smiles,
        "molecular_weight": round(mw, 2),
        "logP": round(logp, 2),
        "h_bond_donors": hbd,
        "h_bond_acceptors": hba,
        "tpsa": round(tpsa, 2),
        "rotatable_bonds": rot_bonds,
        "rings": rings,
        "aromatic_rings": aromatic_rings,
        "heavy_atoms": heavy_atoms,
        "fsp3": round(fsp3, 3),
        "qed_score": round(qed_score, 3),
        "lipinski": {
            "violations": ro5_violations,
            "pass": ro5_violations <= 1,
            "interpretation": _ro5_interpretation(ro5_violations),
        },
        "veber_rules": {
            "pass": veber_pass,
            "interpretation": "Likely orally bioavailable" if veber_pass else "Potential oral bioavailability issues",
        },
        "bbb_penetration": bbb_score,
        "estimated_solubility": {
            "log_s": round(log_sol, 2),
            "category": _solubility_category(log_sol),
        },
        "drug_likeness": {
            "qed": round(qed_score, 3),
            "interpretation": _qed_interpretation(qed_score),
        },
    }


def _estimate_bbb(mw: float, logp: float, tpsa: float, hbd: int) -> dict:
    """Simplified CNS penetration heuristic (not a validated QSAR model)."""
    score = 0
    if mw < 450:
        score += 1
    if 1 <= logp <= 3:
        score += 1
    if tpsa < 90:
        score += 1
    if hbd <= 3:
        score += 1

    if score >= 3:
        prediction = "Likely CNS penetrant"
    elif score == 2:
        prediction = "Moderate CNS penetration"
    else:
        prediction = "Unlikely to penetrate BBB"

    return {"score": score, "max": 4, "prediction": prediction}


def _estimate_solubility(logp: float, mw: float, rot_bonds: int, aromatic_rings: int) -> float:
    """ESOL approximation (Delaney, 2004): LogS estimate."""
    return 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rot_bonds - 0.74 * aromatic_rings


def _solubility_category(log_s: float) -> str:
    if log_s > -1:
        return "Highly soluble"
    elif log_s > -2:
        return "Soluble"
    elif log_s > -3:
        return "Moderately soluble"
    elif log_s > -4:
        return "Low solubility"
    else:
        return "Poorly soluble"


def _ro5_interpretation(violations: int) -> str:
    if violations == 0:
        return "Excellent drug-likeness (Ro5 fully satisfied)"
    elif violations == 1:
        return "Good drug-likeness (1 Ro5 violation — borderline)"
    elif violations == 2:
        return "Reduced drug-likeness (2 Ro5 violations)"
    else:
        return "Poor drug-likeness (>2 Ro5 violations — unlikely orally bioavailable)"


def _qed_interpretation(qed: float) -> str:
    if qed >= 0.7:
        return "High drug-likeness"
    elif qed >= 0.5:
        return "Moderate drug-likeness"
    elif qed >= 0.3:
        return "Low drug-likeness"
    else:
        return "Very low drug-likeness"
