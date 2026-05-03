"""
Project 12 — Drug Discovery Dashboard
Streamlit app: paste a SMILES or pick from 40+ real drugs to get
molecule visualisation, drug-likeness, ADMET, covalent warheads,
and similarity search — all explained in plain English.

Deploy free: https://share.streamlit.io
"""

import io
import streamlit as st

try:
    from rdkit import Chem
    from rdkit.Chem import (
        Descriptors, Lipinski, QED,
        rdMolDescriptors, AllChem, DataStructs,
    )
    from rdkit.Chem.Draw import rdMolDraw2D
    from PIL import Image
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False

from services.admet_service import predict_admet, validate_smiles

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drug Discovery Dashboard",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .pass  { color:#4ade80; font-weight:700; }
  .fail  { color:#f87171; font-weight:700; }
  .warn  { color:#facc15; font-weight:700; }
  .compound-name {
      font-size:2rem; font-weight:800;
      background:linear-gradient(90deg,#818cf8,#38bdf8);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent;
      margin-bottom:0;
  }
  .explain-box {
      background:#1e293b; border-left:3px solid #38bdf8;
      padding:10px 14px; border-radius:6px;
      margin:6px 0 14px 0; font-size:0.88rem; line-height:1.6;
  }
  .rule-row {
      display:flex; align-items:baseline; gap:10px;
      padding:6px 0; border-bottom:1px solid #1e293b;
  }
  .rule-label { min-width:160px; font-weight:600; }
  .rule-value { min-width:80px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Compound library
# ─────────────────────────────────────────────────────────────────────────────
EXAMPLES: dict[str, str] = {
    # Pain / Anti-inflammatory
    "Aspirin":              "CC(=O)Oc1ccccc1C(=O)O",
    "Ibuprofen":            "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
    "Naproxen":             "COc1ccc2cc([C@@H](C)C(=O)O)ccc2c1",
    "Diclofenac":           "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
    "Celecoxib":            "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
    "Paracetamol":          "CC(=O)Nc1ccc(O)cc1",
    "Morphine":             "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@@H]3[C@@H]1C5",
    "Tramadol":             "OC1(c2ccccc2OC)CCCC[C@@H]1CN(C)C",
    # Antibiotics
    "Amoxicillin":          "CC1(C)S[C@@H]2[C@H](NC(=O)[C@@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O",
    "Ampicillin":           "CC1(C)S[C@@H]2[C@H](NC(=O)[C@@H](N)c3ccccc3)C(=O)N2[C@H]1C(=O)O",
    "Ciprofloxacin":        "OC(=O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "Levofloxacin":         "C[C@@H]1COc2c(N3CCN(C)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23",
    "Azithromycin":         "CC[C@@H]1OC(=O)[C@H](C)[C@@H](O[C@@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@H]2C[C@H](N(C)C)[C@@H](O)[C@H](C)O2)[C@](C)(O)C[C@@H]1C",
    "Vancomycin":           "CC[C@H](C)[C@@H]1NC(=O)[C@@H]2Cc3cc4cc(O[C@H]5C[C@H](N)[C@@H](O)[C@H](C)O5)c(Cl)c(O4)c3OC(=O)[C@H](CC(N)=O)NC(=O)[C@H](CC(=O)O)NC(=O)[C@@H]3Cc4c(Cl)c(O[C@@H]5C[C@@H](O[C@H]6CC(=C)O[C@@H](C)C6=O)[C@H](NC(C)=O)[C@@H](O5)C(=O)O)cc(c4O)NC(=O)[C@H](NC1=O)[C@@H](O)c1ccc(O)c(c1)-c1c(O)cc2",
    "Penicillin G":         "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O",
    # CNS / Psychiatry
    "Caffeine":             "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Theophylline":         "Cn1cnc2c1c(=O)[nH]c(=O)n2C",
    "Diazepam":             "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
    "Fluoxetine":           "CNCC(COc1ccc(C(F)(F)F)cc1)c1ccccc1",
    "Sertraline":           "CNC1CC(c2ccc(Cl)c(Cl)c2)c2ccccc21",
    "Haloperidol":          "OC1(CCc2ccc(Cl)cc2)CCN(CCCC(=O)c2ccc(F)cc2)CC1",
    "Lithium carbonate":    "[Li+].[Li+].[O-]C([O-])=O",
    # Cardiovascular
    "Atorvastatin":         "CC(C)c1c(C(=O)Nc2ccccc2F)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O",
    "Simvastatin":          "CCC(C)(C)C(=O)O[C@H]1C[C@@H](CC2[C@@H]1[C@@H]1CC[C@H](O1)CC2=O)C",
    "Amlodipine":           "CCOC(=O)C1=C(CCN)NC(C)=C(C(=O)OCC)C1c1ccccc1Cl",
    "Metoprolol":           "COCCC(=O)CCOc1ccc(C[C@@H](O)CNC(C)C)cc1",
    "Atenolol":             "CC(C)NC[C@@H](O)COc1ccc(CC(N)=O)cc1",
    "Losartan":             "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1",
    "Captopril":            "CC(CS)C(=O)N1CCC[C@H]1C(=O)O",
    "Warfarin":             "OC(=O)CCCC(=O)c1ccc2ccccc2c1",
    "Clopidogrel":          "COC(=O)[C@@H]1CCCN1Cc1ccc(Cl)cc1",
    # Oncology
    "Imatinib":             "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "Erlotinib":            "C#Cc1cccc(Nc2ncnc3cc(OCCO)c(OCCO)cc23)c1",
    "Ibrutinib":            "O=C(/C=C/c1ccccc1)N1CC[C@@H](n2nc(-c3ccc(Oc4ccccc4)cc3)c3c(N)ncnc32)C1",
    "Methotrexate":         "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc1",
    "Tamoxifen":            "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
    "Doxorubicin":          "COc1cccc2C(=O)c3c(O)c4C[C@](O)(CC(=O)CO)C[C@@H](O[C@H]5C[C@H](N)[C@@H](O)[C@H](C)O5)[C@@H]4c(O)c3C(=O)c12",
    # Metabolic / Diabetes
    "Metformin":            "CN(C)C(=N)NC(=N)N",
    "Glipizide":            "Cc1cnc(C(=O)NCCc2ccc(S(=O)(=O)NC(=O)NC3CCCCC3)cc2)s1",
    "Sitagliptin":          "Fc1cc(CC(N)CC(=O)N2CC[C@@H](N3NC(=O)CC3=O)C2)c(F)cc1F",
    # Other
    "Sildenafil":           "CCCC1=NN(C)C(=O)c2[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc21",
    "Oseltamivir":          "CCOC(=O)[C@@H]1C[C@H](OC(CC)CC)[C@@H](NC(C)=O)[C@H](N)C1",
    "Omeprazole":           "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",
    "Hydroxychloroquine":   "CCN(CCO)CCCC(C)Nc1ccnc2cc(Cl)ccc12",
}

SMILES_TO_NAME = {v: k for k, v in EXAMPLES.items()}

# ─────────────────────────────────────────────────────────────────────────────
# Drug similarity database
# ─────────────────────────────────────────────────────────────────────────────
DRUG_DB: dict[str, str] = {
    "Aspirin":            "CC(=O)Oc1ccccc1C(=O)O",
    "Ibuprofen":          "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
    "Naproxen":           "COc1ccc2cc([C@@H](C)C(=O)O)ccc2c1",
    "Diclofenac":         "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
    "Celecoxib":          "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
    "Paracetamol":        "CC(=O)Nc1ccc(O)cc1",
    "Caffeine":           "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Theophylline":       "Cn1cnc2c1c(=O)[nH]c(=O)n2C",
    "Diazepam":           "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
    "Fluoxetine":         "CNCC(COc1ccc(C(F)(F)F)cc1)c1ccccc1",
    "Sertraline":         "CNC1CC(c2ccc(Cl)c(Cl)c2)c2ccccc21",
    "Haloperidol":        "OC1(CCc2ccc(Cl)cc2)CCN(CCCC(=O)c2ccc(F)cc2)CC1",
    "Atorvastatin":       "CC(C)c1c(C(=O)Nc2ccccc2F)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O",
    "Simvastatin":        "CCC(C)(C)C(=O)O[C@H]1C[C@@H](CC2[C@@H]1[C@@H]1CC[C@H](O1)CC2=O)C",
    "Amlodipine":         "CCOC(=O)C1=C(CCN)NC(C)=C(C(=O)OCC)C1c1ccccc1Cl",
    "Metoprolol":         "COCCC(=O)CCOc1ccc(C[C@@H](O)CNC(C)C)cc1",
    "Atenolol":           "CC(C)NC[C@@H](O)COc1ccc(CC(N)=O)cc1",
    "Losartan":           "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1",
    "Captopril":          "CC(CS)C(=O)N1CCC[C@H]1C(=O)O",
    "Warfarin":           "OC(=O)CCCC(=O)c1ccc2ccccc2c1",
    "Imatinib":           "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "Erlotinib":          "C#Cc1cccc(Nc2ncnc3cc(OCCO)c(OCCO)cc23)c1",
    "Methotrexate":       "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc1",
    "Tamoxifen":          "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
    "Sildenafil":         "CCCC1=NN(C)C(=O)c2[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc21",
    "Omeprazole":         "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",
    "Amoxicillin":        "CC1(C)S[C@@H]2[C@H](NC(=O)[C@@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O",
    "Ciprofloxacin":      "OC(=O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "Metformin":          "CN(C)C(=N)NC(=N)N",
    "Sitagliptin":        "Fc1cc(CC(N)CC(=O)N2CC[C@@H](N3NC(=O)CC3=O)C2)c(F)cc1F",
    "Ibrutinib":          "O=C(/C=C/c1ccccc1)N1CC[C@@H](n2nc(-c3ccc(Oc4ccccc4)cc3)c3c(N)ncnc32)C1",
    "Oseltamivir":        "CCOC(=O)[C@@H]1C[C@H](OC(CC)CC)[C@@H](NC(C)=O)[C@H](N)C1",
    "Hydroxychloroquine": "CCN(CCO)CCCC(C)Nc1ccnc2cc(Cl)ccc12",
    "Glipizide":          "Cc1cnc(C(=O)NCCc2ccc(S(=O)(=O)NC(=O)NC3CCCCC3)cc2)s1",
    "Penicillin G":       "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O",
}

# ─────────────────────────────────────────────────────────────────────────────
# Covalent warhead SMARTS
# ─────────────────────────────────────────────────────────────────────────────
WARHEADS: list[tuple[str, str, str, str]] = [
    ("Acrylamide (Michael acceptor)",
     "[CX3;H1,H2]=[CX3]C(=O)[NX3]", "moderate",
     "Reacts with cysteine residues via 1,4-addition. Used intentionally in drugs like ibrutinib."),
    ("Vinyl ketone (strong Michael acceptor)",
     "[CX3;H1,H2]=[CX3]C(=O)[#6]", "high",
     "Highly reactive electrophile — non-selective towards Cys, Lys, and His. Often a toxicity flag."),
    ("α,β-unsaturated carbonyl",
     "[CX3]=[CX3][CX3](=[OX1])", "moderate",
     "Softer electrophile; can covalently modify proteins. Common in natural products and some drugs."),
    ("Epoxide",
     "C1OC1", "high",
     "Ring-strain-driven alkylation of nucleophiles (Cys, Lys, DNA). A classic toxicophore."),
    ("Acyl chloride",
     "C(=O)Cl", "high",
     "Extremely reactive acylating agent. Rare in drugs; often a synthetic intermediate."),
    ("Aldehyde",
     "[CX3H1](=O)[#6]", "moderate",
     "Forms reversible Schiff bases with Lys. Some drugs exploit this (e.g., retinal)."),
    ("Isocyanate",
     "[NX2]=[CX2]=[OX1]", "high",
     "Reacts with Lys, Cys, Tyr. Rarely found in drugs; often a metabolite concern."),
    ("Vinyl sulfone",
     "[CX3]=[CX3]S(=O)(=O)", "high",
     "Strong irreversible Cys modifier. Used in research probes; rarely in approved drugs."),
    ("Aziridine",
     "C1NC1", "moderate",
     "Ring-strain electrophile similar to epoxide. Found in some antibiotics (mitomycin C)."),
    ("Chloroacetamide",
     "ClCC(=O)N", "high",
     "Irreversible Cys alkylator. Occasionally used in targeted covalent drugs."),
    ("Vinyl halide",
     "[CX3]=[CX3][FClBrI]", "moderate",
     "Can undergo elimination or substitution in vivo. Context-dependent reactivity."),
    ("Disulfide",
     "SSC", "low",
     "Forms reversible mixed-disulfide with Cys. Low concern — many biologics contain this motif."),
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def mol_to_png(mol, width: int = 420, height: int = 310) -> Image.Image:
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    drawer.drawOptions().addStereoAnnotation = True
    drawer.drawOptions().padding = 0.08
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText()))


def _fp(mol):
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    return gen.GetFingerprint(mol)


def find_similar_drugs(query_mol, top_n: int = 8) -> list[dict]:
    qfp = _fp(query_mol)
    results = []
    for name, smi in DRUG_DB.items():
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        sim = DataStructs.TanimotoSimilarity(qfp, _fp(m))
        results.append({"name": name, "smiles": smi, "similarity": round(sim, 4)})
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_n]


def detect_warheads(mol) -> list[dict]:
    hits = []
    for label, smarts, reactivity, mechanism in WARHEADS:
        patt = Chem.MolFromSmarts(smarts)
        if patt and mol.HasSubstructMatch(patt):
            hits.append({"label": label, "reactivity": reactivity, "mechanism": mechanism})
    return hits


def _badge(text: str, kind: str) -> str:
    cls = {"pass": "pass", "fail": "fail", "warn": "warn"}.get(kind, "")
    icon = {"pass": "✅", "fail": "❌", "warn": "⚠️"}.get(kind, "•")
    return f'<span class="{cls}">{icon} {text}</span>'


def _explain(text: str) -> None:
    st.markdown(f'<div class="explain-box">{text}</div>', unsafe_allow_html=True)


def _rule_row(label: str, value, passed: bool | None, unit: str = "") -> str:
    if passed is True:
        badge = _badge("Pass", "pass")
    elif passed is False:
        badge = _badge("Fail", "fail")
    else:
        badge = _badge("Note", "warn")
    return (
        f'<div class="rule-row">'
        f'<span class="rule-label">{label}</span>'
        f'<span class="rule-value"><b>{value}</b>{" " + unit if unit else ""}</span>'
        f'{badge}'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚗️ Drug Discovery")
    st.caption("Paste any SMILES or pick from 40+ real drugs to analyse it instantly.")

    category_map = {
        "── Pain / Anti-inflammatory ──": [
            "Aspirin", "Ibuprofen", "Naproxen", "Diclofenac",
            "Celecoxib", "Paracetamol", "Morphine", "Tramadol",
        ],
        "── Antibiotics ──": [
            "Amoxicillin", "Ampicillin", "Ciprofloxacin",
            "Levofloxacin", "Azithromycin", "Vancomycin", "Penicillin G",
        ],
        "── CNS / Psychiatry ──": [
            "Caffeine", "Theophylline", "Diazepam",
            "Fluoxetine", "Sertraline", "Haloperidol", "Lithium carbonate",
        ],
        "── Cardiovascular ──": [
            "Atorvastatin", "Simvastatin", "Amlodipine", "Metoprolol",
            "Atenolol", "Losartan", "Captopril", "Warfarin", "Clopidogrel",
        ],
        "── Oncology ──": [
            "Imatinib", "Erlotinib", "Ibrutinib",
            "Methotrexate", "Tamoxifen", "Doxorubicin",
        ],
        "── Metabolic / Diabetes ──": [
            "Metformin", "Glipizide", "Sitagliptin",
        ],
        "── Other ──": [
            "Sildenafil", "Oseltamivir", "Omeprazole", "Hydroxychloroquine",
        ],
    }

    flat_options = ["— pick an example —"]
    for header, drugs in category_map.items():
        flat_options.append(header)
        flat_options.extend([f"  {d}" for d in drugs])

    raw_choice = st.selectbox("Pick an example", flat_options)
    chosen_name = raw_choice.strip()
    is_header = chosen_name.startswith("──") or chosen_name == "— pick an example —"
    default_smiles = "" if is_header else EXAMPLES.get(chosen_name, "")

    smiles_input = st.text_area(
        "SMILES",
        value=default_smiles,
        height=90,
        placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O",
        help="Simplified Molecular Input Line Entry System notation",
    )

    st.button("Analyse", type="primary", use_container_width=True)

    st.divider()
    st.markdown("**How to read this dashboard**")
    st.markdown(
        "- **Drug-likeness** — can it be a pill?\n"
        "- **ADMET** — how does the body handle it?\n"
        "- **Covalent** — does it react with proteins?\n"
        "- **Similar Drugs** — what approved drugs look like it?\n\n"
        "Results are heuristic estimates, not validated QSAR models."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Guard
# ─────────────────────────────────────────────────────────────────────────────
if not smiles_input.strip():
    st.markdown(
        "## Welcome to the Drug Discovery Dashboard\n\n"
        "Pick a compound from the sidebar dropdown or paste your own SMILES string to begin.\n\n"
        "**What is SMILES?** A one-line text code that represents a molecule. "
        "`CC(=O)Oc1ccccc1C(=O)O` is aspirin. "
        "Copy SMILES from [PubChem](https://pubchem.ncbi.nlm.nih.gov/) or "
        "[ChEMBL](https://www.ebi.ac.uk/chembl/)."
    )
    st.stop()

if not RDKIT_OK:
    st.error("RDKit is not installed. Run `pip install rdkit` and restart.")
    st.stop()

mol = Chem.MolFromSmiles(smiles_input.strip())
if mol is None:
    st.error(
        f"❌ **Invalid SMILES:** `{smiles_input.strip()}`\n\n"
        "Check for typos or copy the SMILES directly from PubChem."
    )
    st.stop()

data = predict_admet(smiles_input.strip())
if "error" in data:
    st.error(data["error"])
    st.stop()

compound_name = (
    chosen_name if (not is_header and chosen_name in EXAMPLES)
    else SMILES_TO_NAME.get(smiles_input.strip(), "Custom Compound")
)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f'<p class="compound-name">{compound_name}</p>', unsafe_allow_html=True)
st.caption(f"`{smiles_input.strip()}`")
st.divider()

col_img, col_kpi = st.columns([1, 2], gap="large")

with col_img:
    try:
        img = mol_to_png(mol)
        st.image(img, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render molecule: {e}")

with col_kpi:
    st.markdown("### At a Glance")
    k1, k2, k3 = st.columns(3)
    k1.metric("Mol. Weight",  f"{data['molecular_weight']} Da",
              help="Molecular mass. Oral drugs typically 150–500 Da.")
    k2.metric("LogP",          data["logP"],
              help="Lipophilicity. Ideal −1 to 5.")
    k3.metric("QED Score",     data["drug_likeness"]["qed"],
              help="Quantitative Estimate of Drug-likeness. 0=bad, 1=perfect.")

    k4, k5, k6 = st.columns(3)
    k4.metric("TPSA",          f"{data['tpsa']} Ų",
              help="Topological Polar Surface Area. <90 Ų → good oral absorption.")
    k5.metric("HBD / HBA",     f"{data['h_bond_donors']} / {data['h_bond_acceptors']}",
              help="H-bond donors (≤5) and acceptors (≤10).")
    k6.metric("Rot. Bonds",    data["rotatable_bonds"],
              help="≤10 for good oral bioavailability.")

    ro5   = data["lipinski"]
    veber = data["veber_rules"]
    st.markdown(
        "**Lipinski Ro5:** " +
        (_badge("Pass", "pass") if ro5["pass"] else _badge(f"{ro5['violations']} violation(s)", "fail")) +
        "&nbsp;&nbsp;|&nbsp;&nbsp;**Veber:** " +
        (_badge("Pass", "pass") if veber["pass"] else _badge("Fail", "fail")) +
        "&nbsp;&nbsp;|&nbsp;&nbsp;**QED:** " +
        (_badge(data["drug_likeness"]["interpretation"],
                "pass" if data["drug_likeness"]["qed"] >= 0.5 else "warn")),
        unsafe_allow_html=True,
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_dl, tab_admet, tab_cov, tab_sim = st.tabs(
    ["💊 Drug-likeness", "🔬 ADMET", "⚡ Covalent Warheads", "🔗 Similar Drugs"]
)

mw   = data["molecular_weight"]
logp = data["logP"]
hbd  = data["h_bond_donors"]
hba  = data["h_bond_acceptors"]
tpsa = data["tpsa"]

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Drug-likeness
# ═══════════════════════════════════════════════════════════════════════════
with tab_dl:
    st.subheader("Drug-likeness Analysis")
    _explain(
        "Drug-likeness rules are filters developed from thousands of approved oral drugs. "
        "They estimate whether a molecule has the right size, polarity, and flexibility to "
        "<b>survive the journey from pill to bloodstream</b>. Failing one rule doesn't doom a drug — "
        "many approved drugs break one rule — but multiple failures are a strong warning sign."
    )

    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("#### Lipinski Rule of Five")
        _explain(
            "Developed by Pfizer's Christopher Lipinski in 1997 by analysing oral drugs. "
            "The idea: your molecule must fit through biological membranes, which means "
            "it can't be too big, too greasy, or too polar. "
            "All four thresholds are multiples of 5 — hence 'Rule of Five.'"
        )

        st.markdown(_rule_row("Molecular Weight", mw, mw <= 500, "Da"), unsafe_allow_html=True)
        if mw <= 500:
            _explain(
                f"✅ <b>{mw} Da</b> — within the 500 Da limit. "
                "Light enough to diffuse across cell membranes and gut walls. "
                "Most tablets contain an active ingredient between 150–500 Da."
            )
        else:
            _explain(
                f"❌ <b>{mw} Da</b> — exceeds 500 Da. "
                "Large molecules can't passively cross membranes so oral absorption is poor. "
                "This compound may need injection delivery, or its size must be reduced by "
                "removing non-essential groups."
            )

        st.markdown(_rule_row("LogP (lipophilicity)", logp, logp <= 5), unsafe_allow_html=True)
        if logp <= 5:
            if logp < -1:
                _explain(
                    f"⚠️ <b>LogP {logp}</b> — technically passes Ro5 but very hydrophilic. "
                    "Extremely low LogP can mean the drug can't cross lipid membranes to reach "
                    "intracellular targets."
                )
            else:
                _explain(
                    f"✅ <b>LogP {logp}</b> — good lipophilicity balance. "
                    "The molecule can dissolve in water AND cross fatty membranes. "
                    "LogP 0–3 is ideal: soluble enough to stay in blood, "
                    "lipophilic enough to enter tissues."
                )
        else:
            _explain(
                f"❌ <b>LogP {logp}</b> — too lipophilic. "
                "Greasy molecules accumulate in fat tissue, are metabolised rapidly by the liver, "
                "and often show toxicity. Consider adding polar groups to bring LogP below 5."
            )

        st.markdown(_rule_row("H-Bond Donors (HBD)", hbd, hbd <= 5), unsafe_allow_html=True)
        if hbd <= 5:
            _explain(
                f"✅ <b>{hbd} H-bond donor(s)</b> — acceptable. "
                "OH and NH groups help dissolve the drug in water but slow membrane crossing. "
                "Up to 5 is fine for most oral drugs."
            )
        else:
            _explain(
                f"❌ <b>{hbd} H-bond donors</b> — too many. "
                "Each OH/NH traps the molecule in water. It can't desolvate to cross "
                "the lipid bilayer of the gut wall."
            )

        st.markdown(_rule_row("H-Bond Acceptors (HBA)", hba, hba <= 10), unsafe_allow_html=True)
        if hba <= 10:
            _explain(
                f"✅ <b>{hba} H-bond acceptor(s)</b> — acceptable. "
                "Oxygens and nitrogens add polarity and water solubility. "
                "The ≤10 threshold gives more room than donors because acceptors "
                "are slightly less penalising for membrane permeability."
            )
        else:
            _explain(
                f"❌ <b>{hba} H-bond acceptors</b> — too many polar groups overall, "
                "making the molecule too polar to efficiently cross biological membranes."
            )

        violations = ro5["violations"]
        if violations == 0:
            st.success("🏆 Perfect Lipinski score — 0 violations. Excellent oral drug candidate.")
        elif violations == 1:
            st.info("ℹ️ 1 violation — borderline. Many approved drugs break one rule. Acceptable.")
        elif violations == 2:
            st.warning("⚠️ 2 violations — reduced drug-likeness. Oral bioavailability is uncertain.")
        else:
            st.error("🚫 3+ violations — poor oral drug-likeness. Likely requires IV or alternative delivery.")

    with c_right:
        st.markdown("#### Veber Rules (oral bioavailability)")
        _explain(
            "Proposed by Veber et al. in 2002 using rat bioavailability data. "
            "Two rules focusing on flexibility and polar surface area — "
            "complementary to Lipinski and particularly important for predicting gut absorption."
        )

        rb = data["rotatable_bonds"]
        st.markdown(_rule_row("Rotatable Bonds", rb, rb <= 10), unsafe_allow_html=True)
        if rb <= 10:
            _explain(
                f"✅ <b>{rb} rotatable bond(s)</b> — flexible enough but not too floppy. "
                "Too many rotatable bonds means the molecule becomes entropic — it loses "
                "too much conformational freedom when binding, lowering bioavailability."
            )
        else:
            _explain(
                f"❌ <b>{rb} rotatable bonds</b> — too flexible. "
                "Consider constraining the structure with rings or double bonds "
                "to lock it into a productive conformation."
            )

        st.markdown(_rule_row("TPSA", tpsa, tpsa <= 140, "Ų"), unsafe_allow_html=True)
        if tpsa <= 60:
            _explain(
                f"✅ <b>TPSA {tpsa} Ų</b> — very low polar surface area. "
                "Excellent membrane permeability. May also cross the blood-brain barrier "
                "(CNS drugs typically need TPSA < 60–90 Ų)."
            )
        elif tpsa <= 90:
            _explain(
                f"✅ <b>TPSA {tpsa} Ų</b> — ideal range (60–90 Ų). "
                "Good balance: polar enough for water solubility, non-polar enough for membranes. "
                "This range is associated with high oral absorption AND potential CNS activity."
            )
        elif tpsa <= 140:
            _explain(
                f"✅ <b>TPSA {tpsa} Ų</b> — passes Veber but CNS penetration is unlikely. "
                "Oral absorption is still feasible for peripheral drugs (gut, blood)."
            )
        else:
            _explain(
                f"❌ <b>TPSA {tpsa} Ų</b> — too polar. "
                "Large polar surface area blocks passive diffusion through membranes. "
                "Reduce polar groups (amides, acids, hydroxyls) to improve absorption."
            )

        st.markdown("#### QED — Quantitative Estimate of Drug-likeness")
        _explain(
            "QED (Bickerton et al., 2012) combines 8 descriptors — MW, logP, HBD, HBA, TPSA, "
            "rotatable bonds, aromatic rings, and alerts — into a single 0–1 score. "
            "Think of it as a composite grade. The average QED of approved oral drugs is ~0.67."
        )

        qed_val = data["drug_likeness"]["qed"]
        st.progress(qed_val, text=f"QED = {qed_val}  ({data['drug_likeness']['interpretation']})")

        if qed_val >= 0.7:
            _explain(
                f"✅ <b>QED {qed_val}</b> — high drug-likeness. "
                "This molecule's property profile closely resembles approved oral drugs."
            )
        elif qed_val >= 0.5:
            _explain(
                f"⚠️ <b>QED {qed_val}</b> — moderate drug-likeness. "
                "Some properties are drug-like but others deviate. Check which Ro5/Veber "
                "rules failed above — those drag the QED score down."
            )
        elif qed_val >= 0.3:
            _explain(
                f"⚠️ <b>QED {qed_val}</b> — low drug-likeness. "
                "Multiple properties fall outside typical drug space. "
                "Significant structural optimisation would be needed."
            )
        else:
            _explain(
                f"❌ <b>QED {qed_val}</b> — very low drug-likeness. "
                "Far from approved drug space. May be a useful research tool or natural product, "
                "but unlikely to succeed as an oral drug without major redesign."
            )

        st.markdown("#### Additional Descriptors")
        extras = {
            "Heavy atoms":    (data["heavy_atoms"],    "Total non-hydrogen atoms. Correlates with MW. >30 is large."),
            "Rings":          (data["rings"],           "Total ring count. Rings add rigidity, which can improve target selectivity."),
            "Aromatic rings": (data["aromatic_rings"],  "Flat aromatic systems are important for binding but >3 increases metabolic instability."),
            "Fsp³":           (data["fsp3"],            "Fraction of sp³ carbons. Higher Fsp³ (>0.25) means more 3D shape — often better selectivity and solubility."),
        }
        for prop, (val, explanation) in extras.items():
            with st.expander(f"{prop}: **{val}**"):
                st.write(explanation)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — ADMET
# ═══════════════════════════════════════════════════════════════════════════
with tab_admet:
    st.subheader("ADMET Predictions")
    _explain(
        "ADMET = <b>A</b>bsorption, <b>D</b>istribution, <b>M</b>etabolism, "
        "<b>E</b>xcretion, <b>T</b>oxicity — the five processes that determine a drug's fate "
        "in the body. Over 40% of drug candidates fail in clinical trials due to poor ADMET, "
        "not lack of efficacy. These are heuristic estimates from molecular descriptors, "
        "not experimental measurements."
    )

    a1, a2 = st.columns(2)

    with a1:
        st.markdown("#### 🫁 Absorption")
        _explain(
            "How much of an oral dose reaches the bloodstream. Depends on solubility "
            "(can it dissolve in gut fluid?), permeability (can it cross the gut wall?), "
            "and stability (does stomach acid destroy it?)."
        )

        sol = data["estimated_solubility"]
        log_s, sol_cat = sol["log_s"], sol["category"]
        sol_pass = "poorly" not in sol_cat.lower() and "low" not in sol_cat.lower()
        st.markdown(
            _rule_row("Aqueous Solubility (ESOL)", f"LogS = {log_s}", sol_pass) +
            f'<span style="font-size:0.85rem;margin-left:8px;color:#94a3b8;">{sol_cat}</span>',
            unsafe_allow_html=True
        )
        sol_msgs = {
            "Highly soluble":     f"✅ LogS {log_s} — dissolves rapidly and completely in gut fluid. No solubility barrier to oral absorption.",
            "Soluble":            f"✅ LogS {log_s} — adequate dissolution for oral delivery.",
            "Moderately soluble": f"⚠️ LogS {log_s} — may need formulation help (salt formation, micronisation) for reliable absorption.",
            "Low solubility":     f"⚠️ LogS {log_s} — dissolution is rate-limiting (BCS Class II). Lipid-based formulations or nanosizing often used.",
            "Poorly soluble":     f"❌ LogS {log_s} — &lt;10 µg/mL aqueous solubility. Oral bioavailability will be low and variable.",
        }
        _explain(sol_msgs.get(sol_cat, f"LogS = {log_s}"))

        st.markdown(_rule_row("Lipinski Ro5 (permeability proxy)", "Pass" if ro5["pass"] else "Fail", ro5["pass"]), unsafe_allow_html=True)
        st.markdown(_rule_row("Veber Rules (gut absorption)", "Pass" if veber["pass"] else "Fail", veber["pass"]), unsafe_allow_html=True)

        st.markdown("#### 🔄 Metabolism")
        _explain(
            "Most oral drugs are metabolised by cytochrome P450 enzymes in the liver "
            "(first-pass effect) before reaching circulation. High LogP, many aromatic rings, "
            "and reactive groups all predict faster metabolism."
        )
        aromatic = data["aromatic_rings"]
        fsp3     = data["fsp3"]
        if logp > 3 and aromatic >= 3:
            st.markdown(_badge("Potential rapid hepatic metabolism", "warn"), unsafe_allow_html=True)
            _explain(
                f"⚠️ High LogP ({logp}) with {aromatic} aromatic ring(s) suggests strong CYP450 substrate potential. "
                "The molecule may be cleared quickly. Consider reducing lipophilicity."
            )
        elif fsp3 < 0.2:
            st.markdown(_badge("Flat molecule — metabolic risk", "warn"), unsafe_allow_html=True)
            _explain(
                f"⚠️ Fsp³ = {fsp3} — highly aromatic/flat. Flat molecules are well-recognised by "
                "CYP enzymes. Increasing Fsp³ by adding sp³ carbons improves metabolic stability."
            )
        else:
            st.markdown(_badge("Moderate metabolic liability", "pass"), unsafe_allow_html=True)
            _explain("✅ No strong flags for rapid hepatic clearance.")

    with a2:
        st.markdown("#### 🩸 Distribution (BBB)")
        _explain(
            "After absorption, a drug distributes through blood and tissues. "
            "The key question: does it cross the <b>blood-brain barrier (BBB)</b>? "
            "Essential for CNS drugs; potentially harmful for peripheral drugs."
        )

        bbb       = data["bbb_penetration"]
        bbb_score = bbb["score"]
        st.markdown(
            _rule_row("BBB Penetration Score", f"{bbb_score}/4", bbb_score >= 3) +
            f'<span style="font-size:0.85rem;margin-left:8px;color:#94a3b8;">{bbb["prediction"]}</span>',
            unsafe_allow_html=True
        )
        bbb_criteria = [
            (f"MW < 450 Da  ({mw} Da)",    mw < 450),
            (f"LogP 1–3  ({logp})",         1 <= logp <= 3),
            (f"TPSA < 90 Ų  ({tpsa} Ų)",  tpsa < 90),
            (f"HBD ≤ 3  ({hbd})",           hbd <= 3),
        ]
        for criterion, met in bbb_criteria:
            st.markdown("  " + _badge(criterion, "pass" if met else "fail"), unsafe_allow_html=True)

        if bbb_score >= 3:
            _explain(
                f"✅ Score {bbb_score}/4 — likely CNS penetrant. "
                "<b>CNS drug:</b> excellent. <b>Peripheral drug:</b> may cause neurological side effects."
            )
        elif bbb_score == 2:
            _explain(f"⚠️ Score {bbb_score}/4 — moderate BBB penetration. Not reliable for CNS targets.")
        else:
            _explain(
                f"ℹ️ Score {bbb_score}/4 — unlikely to penetrate the BBB. "
                "For peripheral drugs this is desirable — no unwanted CNS effects."
            )

        st.markdown("#### 🔃 Excretion")
        _explain(
            "Drugs leave the body mainly through renal (urine) or hepatic (bile) excretion. "
            "Low MW + low logP → renal. High logP → hepatic metabolism first."
        )
        if mw < 300 and logp < 1:
            _explain(
                f"✅ MW {mw} Da, LogP {logp} — small and hydrophilic. "
                "Likely direct renal clearance. May need frequent dosing due to rapid elimination."
            )
        elif logp > 4:
            _explain(
                f"⚠️ LogP {logp} — highly lipophilic drugs require hepatic metabolism to more polar "
                "metabolites before renal excretion. Watch for active or toxic metabolites."
            )
        else:
            _explain("✅ Balanced lipophilicity — mixed renal/hepatic excretion. Common in approved oral drugs.")

        st.divider()
        with st.expander("📋 Full descriptor table"):
            import pandas as pd
            rows = [
                ("Molecular Weight",   mw,                                      "Da",  "≤ 500"),
                ("LogP",               logp,                                    "",    "−1 to 5"),
                ("H-Bond Donors",      hbd,                                     "",    "≤ 5"),
                ("H-Bond Acceptors",   hba,                                     "",    "≤ 10"),
                ("TPSA",               tpsa,                                    "Ų",  "≤ 140"),
                ("Rotatable Bonds",    data["rotatable_bonds"],                  "",    "≤ 10"),
                ("Rings",              data["rings"],                            "",    ""),
                ("Aromatic Rings",     aromatic,                                 "",    "≤ 3 preferred"),
                ("Heavy Atoms",        data["heavy_atoms"],                      "",    "< 30 preferred"),
                ("Fsp³",              fsp3,                                     "",    "> 0.25 preferred"),
                ("QED",                data["drug_likeness"]["qed"],             "",    "0–1 (higher=better)"),
                ("LogS (solubility)",  data["estimated_solubility"]["log_s"],   "",    "> −3 preferred"),
                ("BBB Score",          f"{bbb_score}/4",                         "",    "≥ 3 = likely CNS"),
            ]
            df = pd.DataFrame(rows, columns=["Property", "Value", "Unit", "Drug-like Range"])
            st.dataframe(df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Covalent Warheads
# ═══════════════════════════════════════════════════════════════════════════
with tab_cov:
    st.subheader("Covalent Warhead Detection")
    _explain(
        "A <b>covalent warhead</b> is a reactive group that forms a permanent bond with an amino "
        "acid in a protein. Sometimes this is <i>intentional</i> — aspirin, omeprazole, and "
        "ibrutinib are all covalent drugs. But non-selective reactivity causes toxicity. "
        "This screen flags groups worth investigating using SMARTS substructure matching."
    )

    hits = detect_warheads(mol)
    reactivity_icon = {"high": "🔴", "moderate": "🟡", "low": "🟢"}

    if not hits:
        st.success(f"✅ No covalent warheads detected in {compound_name}.")
        _explain(
            "None of the 12 screened reactive motifs are present. "
            "This does not mean the molecule is non-toxic — toxicity has many mechanisms — "
            "but it is not flagged as a potential covalent modifier by this screen."
        )
    else:
        for hit in hits:
            icon = reactivity_icon.get(hit["reactivity"], "•")
            st.markdown(f"**{icon} {hit['label']}** — Reactivity: `{hit['reactivity'].upper()}`")
            _explain(hit["mechanism"])

        st.divider()
        _explain(
            "<b>If covalent binding is intentional</b>: verify selectivity via competitive ABPP "
            "(activity-based protein profiling) to ensure only the intended residue is modified.<br><br>"
            "<b>If NOT intended</b>: consider replacing the reactive group — e.g. replace an "
            "acrylamide with a propionamide, or block the Michael acceptor with a methyl group."
        )

    st.divider()
    st.markdown("**Reactivity guide**")
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.markdown("🔴 **High** — likely reacts non-selectively with Cys, Lys, His. Major toxicity risk unless intentional.")
    col_r2.markdown("🟡 **Moderate** — reversible or slow-reacting. Common in approved drugs (e.g. ibrutinib's acrylamide).")
    col_r3.markdown("🟢 **Low** — mild electrophile. Generally lower toxicity risk.")

    with st.expander("All screened motifs"):
        import pandas as pd
        motif_df = pd.DataFrame([
            {"Warhead": w[0], "Reactivity": w[2], "Mechanism": w[3]}
            for w in WARHEADS
        ])
        st.dataframe(motif_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — Similar Drugs
# ═══════════════════════════════════════════════════════════════════════════
with tab_sim:
    st.subheader("Similar Approved Drugs")
    _explain(
        "Structural similarity to known drugs helps in drug discovery: "
        "<b>scaffold hopping</b> finds new patents, <b>bioisostere replacement</b> improves a "
        "property while keeping activity, and similar approved drugs tell you what ADMET profile "
        "to expect. Similarity is computed using <b>Morgan fingerprints</b> (radius=2, 2048 bits) "
        "and <b>Tanimoto coefficient</b> — the most widely used metric in medicinal chemistry."
    )

    col_guide1, col_guide2, col_guide3, col_guide4 = st.columns(4)
    col_guide1.metric("≥ 0.85", "Same scaffold")
    col_guide1.caption("Likely same pharmacophore")
    col_guide2.metric("0.7–0.85", "Closely related")
    col_guide2.caption("Similar binding pocket expected")
    col_guide3.metric("0.4–0.7", "Structurally related")
    col_guide3.caption("May share mechanism of action")
    col_guide4.metric("< 0.4", "Dissimilar")
    col_guide4.caption("Potentially novel chemical space")

    st.divider()

    similars = find_similar_drugs(mol, top_n=8)

    for i, item in enumerate(similars):
        sim_val = item["similarity"]
        if sim_val >= 0.85:
            icon, tier = "🟢", "Same scaffold"
        elif sim_val >= 0.7:
            icon, tier = "🟢", "Closely related"
        elif sim_val >= 0.4:
            icon, tier = "🟡", "Structurally related"
        else:
            icon, tier = "🔴", "Dissimilar"

        c_name, c_bar, c_score = st.columns([2, 4, 1])
        with c_name:
            st.markdown(f"**{i+1}. {item['name']}**")
            st.caption(tier)
        with c_bar:
            st.progress(sim_val)
        with c_score:
            st.markdown(f"**{sim_val:.3f}**")

    st.divider()
    top = similars[0] if similars else None
    if top and top["similarity"] >= 0.4:
        _explain(
            f"<b>Most structurally similar approved drug: {top['name']}</b> "
            f"(Tanimoto = {top['similarity']:.3f}). "
            "Look up its known ADMET properties, patent status, approved indications, "
            "and toxicity profile — this gives you a head start on predicting yours."
        )
    else:
        _explain(
            "No close structural relatives found in the database. "
            "This may indicate <b>novel chemical space</b> — good for IP, but less prior "
            "ADMET knowledge to draw on."
        )

    with st.expander("How Morgan fingerprints work"):
        st.markdown(
            "Each atom broadcasts its chemical environment outward to radius 2 "
            "(atoms up to 2 bonds away). This neighbourhood is hashed into one of 2048 bit positions. "
            "Tanimoto = **|A ∩ B| / |A ∪ B|** — the fraction of shared bits. "
            "Identical molecules = 1.0; completely different ≈ 0.0."
        )
