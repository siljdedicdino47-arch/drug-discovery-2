"""
Project 12 — Drug Discovery Dashboard (v5)
New: drug repurposing finder, Lipinski fix suggestions, Murcko scaffold,
     stereochemistry analyzer, tautomer enumeration, PubMed literature,
     similarity network graph, rate limiter, method citations, fixed databases
"""

import io, urllib.parse, base64, time, math
import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ── Optional heavy dependencies ───────────────────────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import (Descriptors, Lipinski, QED,
                            rdMolDescriptors, AllChem, DataStructs)
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.EnumerateStereoisomers import (
        EnumerateStereoisomers, StereoEnumerationOptions)
    from PIL import Image
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False

try:
    import py3Dmol
    PY3DMOL_OK = True
except ImportError:
    PY3DMOL_OK = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, Image as RLImage)
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Rate limiter  (per-session, protects external API calls)
# ─────────────────────────────────────────────────────────────────────────────

_RATE_WINDOW = 60
_RATE_MAX    = 15   # calls per window per session

def _allowed(key: str = "api") -> bool:
    """Returns True if the call is within rate limits."""
    now = time.time()
    sk  = f"_rl_{key}"
    if sk not in st.session_state:
        st.session_state[sk] = []
    calls = [t for t in st.session_state[sk] if now - t < _RATE_WINDOW]
    if len(calls) >= _RATE_MAX:
        return False
    calls.append(now)
    st.session_state[sk] = calls
    return True


# ─────────────────────────────────────────────────────────────────────────────
# ADMET engine
# ─────────────────────────────────────────────────────────────────────────────

def predict_admet(smiles: str) -> dict:
    if not RDKIT_OK:
        return {"error": "RDKit not installed."}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES: {smiles}"}
    mw   = Descriptors.ExactMolWt(mol);   logp = Descriptors.MolLogP(mol)
    hbd  = Lipinski.NumHDonors(mol);      hba  = Lipinski.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol);         rb   = Lipinski.NumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    ar    = rdMolDescriptors.CalcNumAromaticRings(mol)
    ha    = mol.GetNumHeavyAtoms()
    fsp3  = rdMolDescriptors.CalcFractionCSP3(mol)
    qed   = QED.qed(mol)
    ro5v  = sum([mw>500, logp>5, hbd>5, hba>10])
    veber = rb<=10 and tpsa<=140
    bbb   = sum([mw<450, 1<=logp<=3, tpsa<90, hbd<=3])
    bbb_p = ("Likely CNS penetrant" if bbb>=3
             else "Moderate CNS penetration" if bbb==2
             else "Unlikely to penetrate BBB")
    log_s = 0.16 - 0.63*logp - 0.0062*mw + 0.066*rb - 0.74*ar
    sol_c = ("Highly soluble" if log_s>-1 else "Soluble" if log_s>-2
             else "Moderately soluble" if log_s>-3
             else "Low solubility" if log_s>-4 else "Poorly soluble")
    return {
        "smiles": smiles, "molecular_weight": round(mw,2), "logP": round(logp,2),
        "h_bond_donors": hbd, "h_bond_acceptors": hba, "tpsa": round(tpsa,2),
        "rotatable_bonds": rb, "rings": rings, "aromatic_rings": ar,
        "heavy_atoms": ha, "fsp3": round(fsp3,3), "qed_score": round(qed,3),
        "lipinski":  {"violations": ro5v, "pass": ro5v<=1,
                      "interpretation": ["Excellent (Ro5 satisfied)","Good (1 violation — borderline)",
                                         "Reduced (2 violations)","Poor (>2 violations)"][min(ro5v,3)]},
        "veber_rules": {"pass": veber,
                        "interpretation": "Likely orally bioavailable" if veber else "Oral bioavailability issues"},
        "bbb_penetration": {"score": bbb, "max": 4, "prediction": bbb_p},
        "estimated_solubility": {"log_s": round(log_s,2), "category": sol_c},
        "drug_likeness": {"qed": round(qed,3),
                          "interpretation": ("High" if qed>=0.7 else "Moderate" if qed>=0.5
                                             else "Low" if qed>=0.3 else "Very low") + " drug-likeness"},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Complexity Score
# ─────────────────────────────────────────────────────────────────────────────

def synthetic_complexity(mol) -> dict:
    stereo     = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    spiro      = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    bridge     = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    rings_info = mol.GetRingInfo().AtomRings()
    large_r    = sum(1 for r in rings_info if 7 < len(r) < 12)
    macro_r    = sum(1 for r in rings_info if len(r) >= 12)
    ha         = mol.GetNumHeavyAtoms()
    raw = (1.0 + stereo*1.0 + spiro*1.5 + bridge*1.0
           + large_r*1.5 + macro_r*2.5 + max(0, ha-25)*0.08)
    score = round(min(10.0, max(1.0, raw)), 1)
    if score <= 3:   label, color = "Easy", "pass"
    elif score <= 5: label, color = "Moderate", "pass"
    elif score <= 7: label, color = "Challenging", "warn"
    else:            label, color = "Very difficult", "fail"
    factors = []
    if stereo:  factors.append(f"{stereo} stereocentre(s)")
    if spiro:   factors.append(f"{spiro} spiro atom(s)")
    if bridge:  factors.append(f"{bridge} bridgehead atom(s)")
    if large_r: factors.append(f"{large_r} medium ring(s) (8–11)")
    if macro_r: factors.append(f"{macro_r} macrocycle(s) (≥12)")
    if ha > 25: factors.append(f"{ha} heavy atoms")
    return {"score": score, "label": label, "color": color,
            "factors": factors, "stereo": stereo, "spiro": spiro,
            "bridge": bridge, "large_rings": large_r, "macrocycles": macro_r}


# ─────────────────────────────────────────────────────────────────────────────
# Lipinski fix suggestions
# ─────────────────────────────────────────────────────────────────────────────

def lipinski_fix_suggestions(data: dict) -> list:
    """Actionable chemistry suggestions for each Ro5 violation."""
    out = []
    mw, logp, hbd, hba = (data["molecular_weight"], data["logP"],
                           data["h_bond_donors"], data["h_bond_acceptors"])
    if mw > 500:
        n = max(1, round((mw-500)/14))
        out.append({"prop": "Molecular Weight", "val": f"{mw} Da", "fixes": [
            f"Remove ~{n} non-pharmacophoric CH₂/CH₃ groups (≈14 Da each)",
            "Replace a phenyl ring with pyridyl, cyclopropyl, or azetidine (saves 28–50 Da)",
            "Shorten or remove flexible linkers — replace with direct bonds",
            "Prodrug strategy: synthesise a lighter precursor activated in vivo",
        ]})
    if logp > 5:
        out.append({"prop": "LogP", "val": str(logp), "fixes": [
            "Add a hydroxyl (−OH) to an aromatic carbon (≈ −0.7 LogP units)",
            "Replace a phenyl with pyridyl or morpholinyl ring (−0.5 to −1.0 units)",
            "Replace a propyl chain with a methyl-ether (≈ −0.5 units)",
            "Add carboxamide or sulfonamide to a non-H-bonded position (≈ −1.0 units)",
        ]})
    if hbd > 5:
        out.append({"prop": "H-Bond Donors", "val": str(hbd), "fixes": [
            "N-methylate amines (R-NH₂ → R-NHMe) — removes 1 HBD per site",
            "Convert carboxylic acids to esters (prodrug) — removes 1 HBD",
            "Replace hydroxyl substituents with halogens or methyl groups",
            "Cyclise an OH+NH pair into an oxazoline or lactam (removes both)",
        ]})
    if hba > 10:
        out.append({"prop": "H-Bond Acceptors", "val": str(hba), "fixes": [
            "Replace ether oxygens (−O−) with methylene (−CH₂−)",
            "Merge two HBA groups into a single ring system bioisostere",
            "Replace carbonyl C=O with C=C where pharmacophore allows",
        ]})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Murcko scaffold
# ─────────────────────────────────────────────────────────────────────────────

def get_murcko_scaffold(mol):
    try:
        scaf    = MurckoScaffold.GetScaffoldForMol(mol)
        generic = MurckoScaffold.MakeScaffoldGeneric(scaf)
        return scaf, generic, Chem.MolToSmiles(scaf), Chem.MolToSmiles(generic)
    except Exception:
        return None, None, None, None


def scaffold_drug_matches(scaf_smi: str, drug_db: dict) -> list:
    matches = []
    scaf_mol = Chem.MolFromSmiles(scaf_smi)
    if scaf_mol is None:
        return matches
    for name, smi in drug_db.items():
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        try:
            s = MurckoScaffold.GetScaffoldForMol(m)
            if Chem.MolToSmiles(s) == scaf_smi:
                matches.append(name)
        except Exception:
            continue
    return matches


# ─────────────────────────────────────────────────────────────────────────────
# Tautomers
# ─────────────────────────────────────────────────────────────────────────────

def enumerate_tautomers(mol, max_t: int = 8):
    try:
        te = rdMolStandardize.TautomerEnumerator()
        te.SetMaxTautomers(max_t)
        tauts  = list(te.Enumerate(mol))
        canon  = te.Canonicalize(mol)
        return [(Chem.MolToSmiles(t), t) for t in tauts], Chem.MolToSmiles(canon)
    except Exception:
        return [], None


# ─────────────────────────────────────────────────────────────────────────────
# Stereoisomer enumeration
# ─────────────────────────────────────────────────────────────────────────────

def enumerate_stereoisomers(mol, max_isomers: int = 16) -> list:
    """Return list of (smiles, mol) for all unique stereoisomers."""
    try:
        opts = StereoEnumerationOptions(unique=True, onlyUnassigned=False,
                                        maxIsomers=max_isomers)
        isomers = list(EnumerateStereoisomers(mol, options=opts))
        return [(Chem.MolToSmiles(iso, isomericSmiles=True), iso) for iso in isomers]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Stereochemistry
# ─────────────────────────────────────────────────────────────────────────────

def analyze_stereo(mol) -> dict:
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    defined = [(i, c) for i, c in centers if c in ("R", "S")]
    undef   = [(i, c) for i, c in centers if c not in ("R", "S")]
    db_stereo = []
    for bond in mol.GetBonds():
        if bond.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ):
            db_stereo.append((bond.GetIdx(),
                              "E" if bond.GetStereo() == Chem.BondStereo.STEREOE else "Z"))
    return {
        "is_chiral":   len(centers) > 0,
        "total":       len(centers),
        "defined":     defined,
        "undefined":   undef,
        "db_stereo":   db_stereo,
        "max_isomers": 2**len(undef) if undef else 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Similarity network graph (requires plotly)
# ─────────────────────────────────────────────────────────────────────────────

def similarity_network(qmol, qname: str, db: dict, threshold: float = 0.2):
    if not PLOTLY_OK:
        return None
    qfp = _fp(qmol)
    drug_sims = []
    for name, smi in db.items():
        m = Chem.MolFromSmiles(smi)
        if m:
            s = DataStructs.TanimotoSimilarity(qfp, _fp(m))
            if s >= threshold:
                drug_sims.append((name, round(s, 4)))
    if not drug_sims:
        return None
    drug_sims.sort(key=lambda x: -x[1])
    # Circular layout — more similar drugs placed closer to centre
    pos = {qname: (0.0, 0.0)}
    n = len(drug_sims)
    for i, (name, sim) in enumerate(drug_sims):
        angle = 2 * math.pi * i / n
        r = 2.5 - sim * 1.5
        pos[name] = (r * math.cos(angle), r * math.sin(angle))
    ex, ey = [], []
    for name, sim in drug_sims:
        x0, y0 = pos[qname]; x1, y1 = pos[name]
        ex += [x0, x1, None]; ey += [y0, y1, None]
    edge_trace = go.Scatter(x=ex, y=ey, mode="lines",
                            line=dict(color="#334155", width=1), hoverinfo="none")
    all_nodes  = [(qname, None)] + drug_sims
    nx_ = [pos[n][0] for n, _ in all_nodes]
    ny_ = [pos[n][1] for n, _ in all_nodes]
    colors  = ["#38bdf8"] + ["#4ade80" if s>=0.7 else "#facc15" if s>=0.4 else "#94a3b8"
                              for _, s in drug_sims]
    sizes   = [24] + [8 + s*16 for _, s in drug_sims]
    htexts  = [f"<b>{qname}</b><br>(query molecule)"] + \
              [f"<b>{n}</b><br>Tanimoto: {s:.3f}" for n, s in drug_sims]
    labels  = [n for n, _ in all_nodes]
    node_trace = go.Scatter(
        x=nx_, y=ny_, mode="markers+text", text=labels,
        textposition="top center", hoverinfo="text", hovertext=htexts,
        marker=dict(size=sizes, color=colors, line=dict(color="#0f172a", width=1))
    )
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        showlegend=False, hovermode="closest",
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0", size=9),
        margin=dict(b=20, l=5, r=5, t=50),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        title=dict(text="Structural Similarity Network · node size ∝ Tanimoto score · "
                        "🔵 query  🟢 similar  🟡 related  ⚫ distant",
                   font=dict(color="#38bdf8", size=12)),
        height=520,
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3D viewer
# ─────────────────────────────────────────────────────────────────────────────

def render_3d(mol, style="Stick + sphere", width=700, height=460):
    if not PY3DMOL_OK:
        st.info("Install py3Dmol: `pip install py3Dmol`")
        return
    try:
        mol3d = Chem.AddHs(mol)
        params = AllChem.ETKDGv3(); params.randomSeed = 42
        if AllChem.EmbedMolecule(mol3d, params) == -1:
            st.warning("Could not generate 3D coordinates for this molecule.")
            return
        AllChem.MMFFOptimizeMolecule(mol3d)
        mb = Chem.MolToMolBlock(mol3d)
        viewer = py3Dmol.view(width=width, height=height)
        viewer.addModel(mb, "mol")
        if style == "Stick only":
            viewer.setStyle({"stick": {"colorscheme": "Jmol", "radius": 0.15}})
        elif style == "Ball and stick":
            viewer.setStyle({"stick": {"colorscheme": "Jmol", "radius": 0.12},
                             "sphere": {"colorscheme": "Jmol", "scale": 0.35}})
        elif style == "Spheres (CPK)":
            viewer.setStyle({"sphere": {"colorscheme": "Jmol"}})
        elif style == "Surface":
            viewer.setStyle({"stick": {"colorscheme": "Jmol", "radius": 0.1}})
            viewer.addSurface("VDW", {"opacity": 0.6, "colorscheme": "Jmol"})
        else:
            viewer.setStyle({"stick": {"colorscheme": "Jmol", "radius": 0.15},
                             "sphere": {"colorscheme": "Jmol", "radius": 0.3}})
        viewer.setBackgroundColor("#0f172a")
        viewer.zoomTo()
        components.html(viewer.write_html(), height=height + 10)
    except Exception as e:
        st.warning(f"3D rendering failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PDF report
# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf(compound_name, smiles, data, sc, mol_img_bytes) -> bytes | None:
    if not REPORTLAB_OK:
        return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=30, bottomMargin=30,
                            leftMargin=40, rightMargin=40)
    styles = getSampleStyleSheet(); story = []
    story.append(Paragraph(f"Drug Discovery Report: {compound_name}", styles["Title"]))
    story.append(Paragraph(f"SMILES: {smiles}", styles["Normal"]))
    story.append(Spacer(1, 12))
    if mol_img_bytes:
        story.append(RLImage(io.BytesIO(mol_img_bytes), width=200, height=150))
        story.append(Spacer(1, 12))
    story.append(Paragraph("Key Properties", styles["Heading2"]))
    props = [
        ["Property","Value","Range"],
        ["Molecular Weight",f"{data['molecular_weight']} Da","≤ 500"],
        ["LogP",str(data["logP"]),"−1 to 5"],
        ["H-Bond Donors",str(data["h_bond_donors"]),"≤ 5"],
        ["H-Bond Acceptors",str(data["h_bond_acceptors"]),"≤ 10"],
        ["TPSA",f"{data['tpsa']} Ų","≤ 140"],
        ["Rotatable Bonds",str(data["rotatable_bonds"]),"≤ 10"],
        ["QED Score",str(data["drug_likeness"]["qed"]),"0–1"],
        ["BBB Prediction",data["bbb_penetration"]["prediction"],""],
        ["Solubility",data["estimated_solubility"]["category"],""],
        ["Synth. Complexity",f"{sc['score']}/10 ({sc['label']})","1=easy"],
        ["Lipinski Ro5","PASS" if data["lipinski"]["pass"] else "FAIL","≤1 violation"],
        ["Veber Rules","PASS" if data["veber_rules"]["pass"] else "FAIL","both criteria"],
    ]
    tbl = Table(props, colWidths=[160, 160, 140])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),rl_colors.HexColor("#1e40af")),
        ("TEXTCOLOR",(0,0),(-1,0),rl_colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[rl_colors.HexColor("#f1f5f9"),rl_colors.white]),
        ("GRID",(0,0),(-1,-1),0.5,rl_colors.HexColor("#94a3b8")),
        ("FONTSIZE",(0,0),(-1,-1),9),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ]))
    story.append(tbl); story.append(Spacer(1,12))
    story.append(Paragraph(f"Interpretation: {data['lipinski']['interpretation']}", styles["Normal"]))
    story.append(Spacer(1,6))
    story.append(Paragraph("Generated by Drug Discovery Dashboard · For research use only.", styles["Normal"]))
    doc.build(story)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Batch processing
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    smiles_col = next((c for c in df.columns if c.lower() in ("smiles","smile","smi")), df.columns[0])
    name_col   = next((c for c in df.columns if c.lower() in ("name","compound","id")), None)
    for i, row in df.iterrows():
        smi  = str(row[smiles_col]).strip()
        name = str(row[name_col]) if name_col else f"Compound_{i+1}"
        d    = predict_admet(smi)
        if "error" in d:
            results.append({"Name": name, "SMILES": smi, "Error": d["error"]}); continue
        mol = Chem.MolFromSmiles(smi)
        sc  = synthetic_complexity(mol) if mol else {}
        results.append({
            "Name": name, "SMILES": smi,
            "MW (Da)": d["molecular_weight"], "LogP": d["logP"],
            "HBD": d["h_bond_donors"], "HBA": d["h_bond_acceptors"],
            "TPSA": d["tpsa"], "Rot. Bonds": d["rotatable_bonds"],
            "QED": d["drug_likeness"]["qed"],
            "Lipinski Pass": d["lipinski"]["pass"],
            "Veber Pass": d["veber_rules"]["pass"],
            "Ro5 Violations": d["lipinski"]["violations"],
            "BBB": d["bbb_penetration"]["prediction"],
            "LogS": d["estimated_solubility"]["log_s"],
            "Solubility": d["estimated_solubility"]["category"],
            "Synth. Complexity": sc.get("score",""),
            "Synth. Label": sc.get("label",""),
        })
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# PubChem / ChEMBL / PubMed
# ─────────────────────────────────────────────────────────────────────────────

PC = "MolecularFormula,MolecularWeight,IUPACName,CanonicalSMILES,IsomericSMILES,InChIKey,CID"
_HEADERS = {"User-Agent": "DrugDiscoveryDashboard/5.0 (research)"}

@st.cache_data(show_spinner=False, ttl=300)
def pubchem_by_name(name: str):
    try:
        encoded = urllib.parse.quote(name.strip(), safe="")
        r = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/{PC}/JSON",
            timeout=15, headers=_HEADERS,
        )
        if not r.ok:
            return None
        return r.json()["PropertyTable"]["Properties"][0]
    except Exception:
        return None

def pubchem_by_smiles(smiles: str):
    """POST first; fall back to GET if POST fails. No cache — button handler uses session_state."""
    # Method 1: POST (handles brackets, @, # in SMILES)
    try:
        r = requests.post(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/property/{PC}/JSON",
            data={"smiles": smiles},
            timeout=15, headers=_HEADERS,
        )
        if r.ok:
            return r.json()["PropertyTable"]["Properties"][0]
    except Exception:
        pass
    # Method 2: GET with URL-encoded SMILES
    try:
        encoded = urllib.parse.quote(smiles, safe="")
        r2 = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
            f"{encoded}/property/{PC}/JSON",
            timeout=15, headers=_HEADERS,
        )
        if r2.ok:
            return r2.json()["PropertyTable"]["Properties"][0]
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False, ttl=300)
def pubchem_synonyms(cid: int):
    try:
        r = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON",
            timeout=15, headers=_HEADERS,
        )
        if not r.ok:
            return []
        return r.json()["InformationList"]["Information"][0].get("Synonym", [])[:8]
    except Exception:
        return []

def chembl_by_inchikey(ik: str, name_hint: str = ""):
    """Search ChEMBL by InChIKey, with name-based fallback. No cache — session_state managed."""
    # Method 1: InChIKey (most precise)
    if ik:
        try:
            r = requests.get(
                f"https://www.ebi.ac.uk/chembl/api/data/molecule"
                f"?molecule_structures__standard_inchi_key={ik}&format=json&limit=1",
                timeout=15, headers=_HEADERS,
            )
            if r.ok:
                mols = r.json().get("molecules", [])
                if mols:
                    return mols[0]
        except Exception:
            pass
    # Method 2: preferred name (fallback when InChIKey fails or is absent)
    if name_hint and name_hint not in ("Custom Compound", "—"):
        try:
            r2 = requests.get(
                f"https://www.ebi.ac.uk/chembl/api/data/molecule"
                f"?pref_name__iexact={urllib.parse.quote(name_hint, safe='')}"
                f"&format=json&limit=1",
                timeout=15, headers=_HEADERS,
            )
            if r2.ok:
                mols = r2.json().get("molecules", [])
                if mols:
                    return mols[0]
        except Exception:
            pass
    return None

@st.cache_data(show_spinner=False, ttl=600)
def pubmed_search(query: str, max_results: int = 5):
    """Query PubMed via NCBI E-utilities."""
    try:
        r1 = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db":"pubmed","term":query,"retmax":max_results,
                    "retmode":"json","sort":"relevance"},
            timeout=10, headers=_HEADERS,
        )
        if not r1.ok:
            return []
        ids = r1.json().get("esearchresult",{}).get("idlist",[])
        if not ids:
            return []
        r2 = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db":"pubmed","id":",".join(ids),"retmode":"json"},
            timeout=10, headers=_HEADERS,
        )
        if not r2.ok:
            return []
        res  = r2.json().get("result", {})
        arts = []
        for pmid in ids:
            art = res.get(pmid, {})
            if not art:
                continue
            authors = art.get("authors", [])
            arts.append({
                "pmid":    pmid,
                "title":   art.get("title", "—"),
                "authors": (authors[0].get("name","") + " et al.") if authors else "Unknown",
                "journal": art.get("fulljournalname", art.get("source","")),
                "year":    art.get("pubdate","")[:4],
                "url":     f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })
        return arts
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# PubChem GHS / Safety data
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=600)
def pubchem_ghs(cid: int):
    """Fetch GHS classification and experimental physical properties from PubChem PUG View."""
    try:
        r = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
            f"?heading=Safety+and+Hazards",
            timeout=15, headers=_HEADERS,
        )
        if not r.ok:
            return None

        def _find(node, heading):
            out = []
            if isinstance(node, dict):
                if node.get("TOCHeading") == heading:
                    out.append(node)
                for v in node.values():
                    out.extend(_find(v, heading))
            elif isinstance(node, list):
                for item in node:
                    out.extend(_find(item, heading))
            return out

        def _strings(section):
            return [swm.get("String","")
                    for info in section.get("Information",[])
                    for swm  in info.get("Value",{}).get("StringWithMarkup",[])
                    if swm.get("String","")]

        record   = r.json().get("Record", {})
        hazards  = []
        precauts = []
        signal   = None
        pictograms = []

        for sec in _find(record, "GHS Classification"):
            for sub in sec.get("Section", []):
                h = sub.get("TOCHeading","")
                if "Hazard Statement"   in h: hazards.extend(_strings(sub))
                elif "Precautionary"   in h: precauts.extend(_strings(sub))
                elif "Signal"          in h:
                    s = _strings(sub)
                    if s: signal = s[0]
                elif "Pictogram"       in h: pictograms.extend(_strings(sub))

        # Experimental physical properties
        phys = {}
        prop_map = {
            "Boiling Point": "boiling_point",  "Melting Point":  "melting_point",
            "Flash Point":   "flash_point",    "Solubility":     "solubility",
            "Density":       "density",        "Vapor Pressure": "vapor_pressure",
            "LogP":          "logp_exp",       "pKa":            "pka",
        }
        for sec in _find(record, "Experimental Properties"):
            for sub in sec.get("Section", []):
                key = prop_map.get(sub.get("TOCHeading",""))
                if key:
                    vals = _strings(sub)
                    if vals:
                        phys[key] = vals[0]

        return {
            "signal":      signal,
            "hazards":     hazards[:12],
            "precautions": precauts[:12],
            "pictograms":  pictograms[:8],
            "phys":        phys,
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Computed MSDS safety profile (structure-based, no API needed)
# ─────────────────────────────────────────────────────────────────────────────

def compute_msds_profile(mol, data: dict, tox_hits: list) -> dict:
    logp = data["logP"]; mw = data["molecular_weight"]
    tpsa = data["tpsa"]; hbd = data["h_bond_donors"]
    rb   = data["rotatable_bonds"]

    high_tox = [h for h in tox_hits if h["severity"] == "high"]
    mod_tox  = [h for h in tox_hits if h["severity"] == "moderate"]

    # GHS hazard statement estimates
    ghs_h = []
    signal = "Warning"

    if high_tox:
        ghs_h.append("H301 Toxic if swallowed [structural estimate]")
        ghs_h.append("H311 Toxic in contact with skin [structural estimate]")
        signal = "Danger"
    elif mod_tox:
        ghs_h.append("H302 Harmful if swallowed [structural estimate]")

    if hbd >= 2 and tpsa > 60:
        ghs_h.append("H315 Causes skin irritation [structural estimate]")
        ghs_h.append("H319 Causes serious eye irritation [structural estimate]")

    if any(h["label"] in ("Nitroaromatic","Aromatic amine") for h in tox_hits):
        ghs_h.append("H351 Suspected of causing cancer [structural alert]")
        signal = "Danger"

    if any(h["label"] == "Hydrazine" for h in tox_hits):
        ghs_h.append("H341 Suspected of causing genetic defects [structural alert]")
        signal = "Danger"

    if logp > 4:
        ghs_h.append("H411 Toxic to aquatic life with long lasting effects")
    elif logp > 2:
        ghs_h.append("H412 Harmful to aquatic life with long lasting effects")

    if not ghs_h:
        ghs_h.append("No specific GHS hazard classification estimated from structure")

    # PPE
    ppe = ["Safety spectacles or goggles", "Lab coat", "Nitrile gloves (0.1 mm min)"]
    if signal == "Danger" or high_tox:
        ppe += ["Chemical-resistant gloves", "Work in certified fume hood",
                "Consider supplied-air respirator if generating aerosols"]
    if logp < -1 and mw < 200:
        ppe.append("Respiratory protection — small, hydrophilic molecules can be inhaled")

    # Storage
    storage = [
        "Store in a cool (≤25°C), dry, well-ventilated area away from direct sunlight",
        "Keep container tightly sealed — use amber glass for light-sensitive compounds",
        "Segregate from oxidisers, strong acids, and strong bases",
    ]
    if any(h["label"] == "Peroxide" for h in tox_hits):
        storage.append("⚠️ Peroxide-forming compound — inspect for peroxide formation before use")
    if logp > 4:
        storage.append("High logP — avoid prolonged contact with plastic containers; use glass")

    # First aid
    first_aid = {
        "Skin contact":  "Remove contaminated clothing. Wash with soap and water for ≥15 min. "
                         "Seek medical attention if irritation persists.",
        "Eye contact":   "Flush with water for ≥15 min, holding eyelids open. "
                         "Seek immediate medical attention.",
        "Ingestion":     "Do NOT induce vomiting. Rinse mouth with water. "
                         "Seek immediate medical attention. Show SDS to physician.",
        "Inhalation":    "Move to fresh air immediately. If breathing is difficult, give oxygen. "
                         "Seek medical attention if symptoms persist.",
    }

    # Physical property estimates (computed)
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
    ha      = mol.GetNumHeavyAtoms()
    env_risk = "High" if logp > 4 else "Moderate" if logp > 2 else "Low"

    # Estimated volatility (very rough — small, nonpolar = more volatile)
    if mw < 150 and logp > 1:
        volatility = "Potentially volatile — handle in fume hood"
    elif mw < 250 and logp > 2:
        volatility = "Low-to-moderate volatility"
    else:
        volatility = "Low volatility expected (MW and logP suggest low vapour pressure)"

    return {
        "signal":        signal,
        "ghs_hazards":   ghs_h,
        "ppe":           ppe,
        "storage":       storage,
        "first_aid":     first_aid,
        "env_risk":      env_risk,
        "formula":       formula,
        "volatility":    volatility,
        "heavy_atoms":   ha,
        "note": ("⚠️ Structure-based estimates only. Always consult the official supplier SDS "
                 "before handling any chemical in a laboratory setting."),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Toxicity
# ─────────────────────────────────────────────────────────────────────────────

TOXICOPHORES = [
    ("Nitroaromatic","[N+](=O)[O-]~a","high",
     "Forms reactive nitroso/hydroxylamine metabolites via CYP enzymes — DNA-damaging and hepatotoxic."),
    ("Aromatic amine","[NH2,NH1]~a","high",
     "Oxidised to reactive nitrenium ions by CYP1A2. Classic carcinogen (bladder cancer) and hepatotoxin."),
    ("Quinone","O=C1C=CC(=O)C=C1","high",
     "Strong Michael acceptors that deplete glutathione, cause oxidative stress and mitochondrial toxicity."),
    ("Hydrazine","[NX3;H1,H2][NX3;H1,H2]","high",
     "Forms reactive radicals; associated with hepatotoxicity (isoniazid) and carcinogenicity."),
    ("Azo compound","[#6]N=N[#6]","moderate",
     "Gut bacteria reduce azo groups to aromatic amines. Toxicity depends on resulting amine."),
    ("Thiocarbonyl","[#6]C(=S)[#6,#7,#8]","moderate",
     "Oxidised to sulfenic acid intermediates that alkylate proteins. Associated with agranulocytosis."),
    ("Alkyl halide","[CX4][Cl,Br,I]","moderate",
     "Potential alkylating agent. Reactivity depends on leaving group and sterics."),
    ("Peroxide","OO","high",
     "Generates ROS that damage lipids, proteins, and DNA. Artemisinin's peroxide is the known exception."),
    ("Michael acceptor","[CX3]=[CX3]C(=O)","moderate",
     "Reacts with glutathione and protein nucleophiles. Risk depends on reactivity context."),
    ("Aldehyde","[CX3H1](=O)[#6]","moderate",
     "Forms Schiff bases with Lys; can cause protein cross-linking."),
    ("Epoxide","C1OC1","high",
     "Alkylates proteins and DNA. Often generated as a reactive metabolite."),
]

WARHEADS = [
    ("Acrylamide (Michael acceptor)","[CX3;H1,H2]=[CX3]C(=O)[NX3]","moderate",
     "Reacts with Cys via 1,4-addition. Intentional in ibrutinib."),
    ("Vinyl ketone","[CX3;H1,H2]=[CX3]C(=O)[#6]","high",
     "Non-selective — reacts with Cys, Lys, His. Toxicity flag."),
    ("α,β-unsaturated carbonyl","[CX3]=[CX3][CX3](=[OX1])","moderate",
     "Softer electrophile; modifies nucleophilic protein residues."),
    ("Epoxide","C1OC1","high","Ring-strain alkylation of nucleophiles."),
    ("Acyl chloride","C(=O)Cl","high","Extremely reactive acylating agent. Rarely in approved drugs."),
    ("Aldehyde","[CX3H1](=O)[#6]","moderate","Reversible Schiff bases with Lys."),
    ("Vinyl sulfone","[CX3]=[CX3]S(=O)(=O)","high","Irreversible Cys modifier."),
    ("Chloroacetamide","ClCC(=O)N","high","Irreversible Cys alkylator."),
    ("Disulfide","SSC","low","Reversible mixed-disulfide. Low concern."),
]

def run_pains(mol):
    try:
        p = FilterCatalogParams(); p.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        cat = FilterCatalog(p); e = cat.GetFirstMatch(mol)
        return [e.GetDescription()] if e else []
    except Exception:
        return []

def detect_tox(mol):
    hits = []
    for label, smarts, sev, expl in TOXICOPHORES:
        patt = Chem.MolFromSmarts(smarts)
        if patt and mol.HasSubstructMatch(patt):
            hits.append({"label": label, "severity": sev, "explanation": expl})
    return hits

def detect_warheads(mol):
    hits = []
    for label, smarts, react, mech in WARHEADS:
        patt = Chem.MolFromSmarts(smarts)
        if patt and mol.HasSubstructMatch(patt):
            hits.append({"label": label, "reactivity": react, "mechanism": mech})
    return hits


# ─────────────────────────────────────────────────────────────────────────────
# Similarity / drug database
# ─────────────────────────────────────────────────────────────────────────────

DRUG_DB = {
    "Aspirin":"CC(=O)Oc1ccccc1C(=O)O","Ibuprofen":"CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
    "Naproxen":"COc1ccc2cc([C@@H](C)C(=O)O)ccc2c1","Diclofenac":"OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
    "Celecoxib":"Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
    "Paracetamol":"CC(=O)Nc1ccc(O)cc1","Caffeine":"Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Diazepam":"CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21","Fluoxetine":"CNCC(COc1ccc(C(F)(F)F)cc1)c1ccccc1",
    "Sertraline":"CNC1CC(c2ccc(Cl)c(Cl)c2)c2ccccc21","Haloperidol":"OC1(CCc2ccc(Cl)cc2)CCN(CCCC(=O)c2ccc(F)cc2)CC1",
    "Atorvastatin":"CC(C)c1c(C(=O)Nc2ccccc2F)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O",
    "Amlodipine":"CCOC(=O)C1=C(CCN)NC(C)=C(C(=O)OCC)C1c1ccccc1Cl",
    "Metoprolol":"COCCC(=O)CCOc1ccc(C[C@@H](O)CNC(C)C)cc1",
    "Losartan":"CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1",
    "Captopril":"CC(CS)C(=O)N1CCC[C@H]1C(=O)O","Warfarin":"OC(=O)CCCC(=O)c1ccc2ccccc2c1",
    "Imatinib":"Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "Erlotinib":"C#Cc1cccc(Nc2ncnc3cc(OCCO)c(OCCO)cc23)c1",
    "Methotrexate":"CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc1",
    "Tamoxifen":"CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
    "Sildenafil":"CCCC1=NN(C)C(=O)c2[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc21",
    "Amoxicillin":"CC1(C)S[C@@H]2[C@H](NC(=O)[C@@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O",
    "Ciprofloxacin":"OC(=O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "Metformin":"CN(C)C(=N)NC(=N)N","Ibrutinib":"O=C(/C=C/c1ccccc1)N1CC[C@@H](n2nc(-c3ccc(Oc4ccccc4)cc3)c3c(N)ncnc32)C1",
    "Oseltamivir":"CCOC(=O)[C@@H]1C[C@H](OC(CC)CC)[C@@H](NC(C)=O)[C@H](N)C1",
    "Hydroxychloroquine":"CCN(CCO)CCCC(C)Nc1ccnc2cc(Cl)ccc12",
    "Penicillin G":"CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O",
    "Omeprazole":"COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",
    "Doxorubicin":"COc1cccc2C(=O)c3c(O)c4C[C@](O)(CC(=O)CO)C[C@@H](O[C@H]5C[C@H](N)[C@@H](O)[C@H](C)O5)[C@@H]4c(O)c3C(=O)c12",
    "Sitagliptin":"Fc1cc(CC(N)CC(=O)N2CC[C@@H](N3NC(=O)CC3=O)C2)c(F)cc1F",
}

DRUG_TARGETS = {
    "Aspirin":         {"class":"NSAID",            "targets":["COX-1","COX-2"],         "indications":["Pain","Antiplatelet","Anti-inflammatory"]},
    "Ibuprofen":       {"class":"NSAID",            "targets":["COX-1","COX-2"],         "indications":["Pain","Anti-inflammatory","Antipyretic"]},
    "Naproxen":        {"class":"NSAID",            "targets":["COX-1","COX-2"],         "indications":["Pain","Anti-inflammatory"]},
    "Diclofenac":      {"class":"NSAID",            "targets":["COX-1","COX-2","LOX"],   "indications":["Pain","Anti-inflammatory"]},
    "Celecoxib":       {"class":"COX-2 selective",  "targets":["COX-2"],                 "indications":["Pain","OA","Cancer prevention"]},
    "Paracetamol":     {"class":"Analgesic",        "targets":["COX-3","CB1"],           "indications":["Pain","Antipyretic"]},
    "Caffeine":        {"class":"Stimulant",        "targets":["A1/A2A adenosine"],      "indications":["CNS stimulant","Headache"]},
    "Diazepam":        {"class":"Benzodiazepine",   "targets":["GABA-A"],                "indications":["Anxiety","Seizure","Muscle relaxant"]},
    "Fluoxetine":      {"class":"SSRI",             "targets":["SERT"],                  "indications":["Depression","OCD","Bulimia"]},
    "Sertraline":      {"class":"SSRI",             "targets":["SERT"],                  "indications":["Depression","PTSD","OCD"]},
    "Haloperidol":     {"class":"Antipsychotic",    "targets":["D2","D3"],               "indications":["Schizophrenia","Tourette's"]},
    "Atorvastatin":    {"class":"Statin",           "targets":["HMG-CoA reductase"],     "indications":["Hypercholesterolaemia","CVD prevention"]},
    "Amlodipine":      {"class":"CCB",              "targets":["L-type Ca²⁺ channel"],   "indications":["Hypertension","Angina"]},
    "Metoprolol":      {"class":"Beta-blocker",     "targets":["β1-adrenoreceptor"],     "indications":["Hypertension","Heart failure"]},
    "Losartan":        {"class":"ARB",              "targets":["AT1 receptor"],          "indications":["Hypertension","Heart failure"]},
    "Captopril":       {"class":"ACE inhibitor",    "targets":["ACE"],                   "indications":["Hypertension","Heart failure"]},
    "Warfarin":        {"class":"Anticoagulant",    "targets":["VKORC1"],                "indications":["VTE","AF","PE"]},
    "Imatinib":        {"class":"TKI",              "targets":["BCR-ABL","c-KIT","PDGFR"],"indications":["CML","GIST"]},
    "Erlotinib":       {"class":"EGFR TKI",         "targets":["EGFR"],                  "indications":["NSCLC","Pancreatic cancer"]},
    "Methotrexate":    {"class":"Antifolate",       "targets":["DHFR"],                  "indications":["Cancer","RA","Psoriasis"]},
    "Tamoxifen":       {"class":"SERM",             "targets":["ER-α","ER-β"],           "indications":["Breast cancer"]},
    "Sildenafil":      {"class":"PDE5i",            "targets":["PDE5"],                  "indications":["ED","PAH"]},
    "Amoxicillin":     {"class":"Beta-lactam",      "targets":["PBP"],                   "indications":["Bacterial infection"]},
    "Ciprofloxacin":   {"class":"Fluoroquinolone",  "targets":["DNA gyrase","Topo IV"],  "indications":["Bacterial infection"]},
    "Metformin":       {"class":"Biguanide",        "targets":["AMPK","Complex I"],      "indications":["T2DM","PCOS"]},
    "Ibrutinib":       {"class":"BTK inhibitor",    "targets":["BTK"],                   "indications":["CLL","MCL","WM"]},
    "Oseltamivir":     {"class":"Neuraminidase inh.","targets":["Influenza NA"],          "indications":["Influenza"]},
    "Hydroxychloroquine":{"class":"Antimalarial",   "targets":["Heme polymerization","TLR"],"indications":["Malaria","RA","SLE"]},
    "Omeprazole":      {"class":"PPI",              "targets":["H+/K+-ATPase"],          "indications":["GERD","PUD"]},
    "Doxorubicin":     {"class":"Anthracycline",    "targets":["Topoisomerase II"],      "indications":["Cancer"]},
    "Sitagliptin":     {"class":"DPP-4i",           "targets":["DPP-4"],                 "indications":["T2DM"]},
    "Penicillin G":    {"class":"Beta-lactam",      "targets":["PBP"],                   "indications":["Bacterial infection"]},
}

EXAMPLES = {
    "Aspirin":"CC(=O)Oc1ccccc1C(=O)O","Ibuprofen":"CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
    "Naproxen":"COc1ccc2cc([C@@H](C)C(=O)O)ccc2c1","Diclofenac":"OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
    "Celecoxib":"Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
    "Paracetamol":"CC(=O)Nc1ccc(O)cc1",
    "Morphine":"CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@@H]3[C@@H]1C5",
    "Tramadol":"OC1(c2ccccc2OC)CCCC[C@@H]1CN(C)C",
    "Amoxicillin":"CC1(C)S[C@@H]2[C@H](NC(=O)[C@@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O",
    "Ciprofloxacin":"OC(=O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "Azithromycin":"CC[C@@H]1OC(=O)[C@H](C)[C@@H](O[C@@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@H]2C[C@H](N(C)C)[C@@H](O)[C@H](C)O2)[C@](C)(O)C[C@@H]1C",
    "Penicillin G":"CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O",
    "Caffeine":"Cn1cnc2c1c(=O)n(C)c(=O)n2C","Theophylline":"Cn1cnc2c1c(=O)[nH]c(=O)n2C",
    "Diazepam":"CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
    "Fluoxetine":"CNCC(COc1ccc(C(F)(F)F)cc1)c1ccccc1","Sertraline":"CNC1CC(c2ccc(Cl)c(Cl)c2)c2ccccc21",
    "Haloperidol":"OC1(CCc2ccc(Cl)cc2)CCN(CCCC(=O)c2ccc(F)cc2)CC1",
    "Atorvastatin":"CC(C)c1c(C(=O)Nc2ccccc2F)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O",
    "Amlodipine":"CCOC(=O)C1=C(CCN)NC(C)=C(C(=O)OCC)C1c1ccccc1Cl",
    "Metoprolol":"COCCC(=O)CCOc1ccc(C[C@@H](O)CNC(C)C)cc1",
    "Losartan":"CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1",
    "Captopril":"CC(CS)C(=O)N1CCC[C@H]1C(=O)O","Warfarin":"OC(=O)CCCC(=O)c1ccc2ccccc2c1",
    "Imatinib":"Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "Erlotinib":"C#Cc1cccc(Nc2ncnc3cc(OCCO)c(OCCO)cc23)c1",
    "Ibrutinib":"O=C(/C=C/c1ccccc1)N1CC[C@@H](n2nc(-c3ccc(Oc4ccccc4)cc3)c3c(N)ncnc32)C1",
    "Methotrexate":"CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc1",
    "Tamoxifen":"CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
    "Doxorubicin":"COc1cccc2C(=O)c3c(O)c4C[C@](O)(CC(=O)CO)C[C@@H](O[C@H]5C[C@H](N)[C@@H](O)[C@H](C)O5)[C@@H]4c(O)c3C(=O)c12",
    "Metformin":"CN(C)C(=N)NC(=N)N","Glipizide":"Cc1cnc(C(=O)NCCc2ccc(S(=O)(=O)NC(=O)NC3CCCCC3)cc2)s1",
    "Sitagliptin":"Fc1cc(CC(N)CC(=O)N2CC[C@@H](N3NC(=O)CC3=O)C2)c(F)cc1F",
    "Sildenafil":"CCCC1=NN(C)C(=O)c2[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc21",
    "Oseltamivir":"CCOC(=O)[C@@H]1C[C@H](OC(CC)CC)[C@@H](NC(C)=O)[C@H](N)C1",
    "Omeprazole":"COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",
    "Hydroxychloroquine":"CCN(CCO)CCCC(C)Nc1ccnc2cc(Cl)ccc12",
}
SMILES_TO_NAME = {v: k for k, v in EXAMPLES.items()}

def _fp(mol):
    return AllChem.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)

def find_similar(qmol, top=8):
    qfp = _fp(qmol)
    res = []
    for name, smi in DRUG_DB.items():
        m = Chem.MolFromSmiles(smi)
        if m:
            res.append({"name": name,
                        "similarity": round(DataStructs.TanimotoSimilarity(qfp, _fp(m)), 4)})
    return sorted(res, key=lambda x: -x["similarity"])[:top]


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def mol_to_png(mol, w=420, h=310):
    d = rdMolDraw2D.MolDraw2DCairo(w, h)
    d.drawOptions().addStereoAnnotation = True
    d.drawOptions().padding = 0.08
    d.DrawMolecule(mol); d.FinishDrawing()
    return d.GetDrawingText()

def _b(text, kind):
    cls  = {"pass":"pass","fail":"fail","warn":"warn"}.get(kind,"")
    icon = {"pass":"✅","fail":"❌","warn":"⚠️"}.get(kind,"•")
    return f'<span class="{cls}">{icon} {text}</span>'

def _xp(text):
    st.markdown(f'<div class="explain-box">{text}</div>', unsafe_allow_html=True)

def _rr(label, value, passed, unit=""):
    badge = (_b("Pass","pass") if passed is True else
             _b("Fail","fail") if passed is False else _b("Note","warn"))
    return (f'<div class="rule-row"><span class="rule-label">{label}</span>'
            f'<span class="rule-value"><b>{value}</b>{" "+unit if unit else ""}</span>{badge}</div>')


# ─────────────────────────────────────────────────────────────────────────────
# Page config + CSS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Drug Discovery Dashboard", page_icon="⚗️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
  .pass{color:#4ade80;font-weight:700}.fail{color:#f87171;font-weight:700}.warn{color:#facc15;font-weight:700}
  .compound-name{font-size:2rem;font-weight:800;
    background:linear-gradient(90deg,#818cf8,#38bdf8);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0}
  .explain-box{background:#1e293b;border-left:3px solid #38bdf8;
    padding:10px 14px;border-radius:6px;margin:6px 0 14px 0;font-size:0.88rem;line-height:1.6}
  .rule-row{display:flex;align-items:baseline;gap:10px;padding:6px 0;border-bottom:1px solid #1e293b}
  .rule-label{min-width:160px;font-weight:600}.rule-value{min-width:80px}
  .hist-item{background:#1e293b;border-radius:6px;padding:6px 10px;margin:3px 0;
    font-size:0.8rem;cursor:pointer}
  .buy-card{background:#1e293b;border-radius:8px;padding:12px 16px;margin:6px 0}
  .fix-box{background:#1e3a1e;border-left:3px solid #4ade80;
    padding:10px 14px;border-radius:6px;margin:6px 0 10px 0;font-size:0.86rem}
  .cite{font-size:0.75rem;color:#64748b;font-style:italic;margin-top:4px}
</style>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

for key, default in [("smiles_override",""), ("lookup_name",""),
                     ("history",[]), ("compare_list",[])]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚗️ Drug Discovery")

    st.markdown("**Search by name**")
    name_q = st.text_input("", placeholder="ibuprofen, aspirin…", label_visibility="collapsed")
    if st.button("Look up on PubChem", use_container_width=True):
        if name_q.strip():
            if _allowed("sidebar"):
                with st.spinner("Querying PubChem…"):
                    pc_res = pubchem_by_name(name_q.strip())
                if pc_res:
                    st.session_state.smiles_override = (pc_res.get("IsomericSMILES")
                                                        or pc_res.get("CanonicalSMILES",""))
                    st.session_state.lookup_name     = pc_res.get("IUPACName", name_q)
                    st.success(f"Found CID {pc_res.get('CID')}")
                else:
                    st.error("Not found. Try a common name (e.g. 'aspirin', 'ibuprofen').")
            else:
                st.warning("Rate limit reached. Wait 60 s.")

    st.divider()
    st.markdown("**Or pick an example**")
    cat_map = {
        "── Pain / Anti-inflammatory ──":["Aspirin","Ibuprofen","Naproxen","Diclofenac","Celecoxib","Paracetamol","Morphine","Tramadol"],
        "── Antibiotics ──":             ["Amoxicillin","Ciprofloxacin","Azithromycin","Penicillin G"],
        "── CNS / Psychiatry ──":        ["Caffeine","Theophylline","Diazepam","Fluoxetine","Sertraline","Haloperidol"],
        "── Cardiovascular ──":          ["Atorvastatin","Amlodipine","Metoprolol","Losartan","Captopril","Warfarin"],
        "── Oncology ──":                ["Imatinib","Erlotinib","Ibrutinib","Methotrexate","Tamoxifen","Doxorubicin"],
        "── Metabolic ──":               ["Metformin","Glipizide","Sitagliptin"],
        "── Other ──":                   ["Sildenafil","Oseltamivir","Omeprazole","Hydroxychloroquine"],
    }
    flat = ["— pick an example —"]
    for hdr, drugs in cat_map.items():
        flat.append(hdr); flat.extend([f"  {d}" for d in drugs])
    raw    = st.selectbox("", flat, label_visibility="collapsed")
    chosen = raw.strip()
    is_hdr = chosen.startswith("──") or chosen == "— pick an example —"
    ex_smi = "" if is_hdr else EXAMPLES.get(chosen,"")

    active_smi  = st.session_state.smiles_override or ex_smi
    smiles_input = st.text_area("SMILES", value=active_smi, height=90,
                                placeholder="CC(=O)Oc1ccccc1C(=O)O")
    if smiles_input != st.session_state.smiles_override:
        st.session_state.smiles_override = ""
        st.session_state.lookup_name     = ""

    st.button("Analyse", type="primary", use_container_width=True)

    st.divider()
    if st.session_state.history:
        st.markdown("**Recent compounds**")
        for entry in reversed(st.session_state.history[-10:]):
            st.markdown(
                f'<div class="hist-item">🧪 <b>{entry["name"]}</b><br>'
                f'<span style="color:#94a3b8;font-size:0.75rem">'
                f'QED {entry["qed"]} · MW {entry["mw"]}</span></div>',
                unsafe_allow_html=True)
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []; st.rerun()

    st.divider()
    st.markdown("**Tabs**")
    st.markdown("💊 Drug-likeness · 🔬 ADMET · ☠️ Toxicity\n\n"
                "⚡ Covalent · 🧊 3D · 🔗 Similar · 🔄 Repurposing\n\n"
                "🧬 Stereo & Tautomers · 🧪 MSDS & Safety\n\n"
                "📊 Compare · 📦 Batch · 🔍 Databases")


# ─────────────────────────────────────────────────────────────────────────────
# Guard
# ─────────────────────────────────────────────────────────────────────────────

if not smiles_input.strip():
    st.markdown("## Welcome to the Drug Discovery Dashboard\n\n"
                "**Three ways to start:**\n"
                "1. Type a name → **Look up on PubChem**\n"
                "2. Pick from the **example dropdown**\n"
                "3. Paste a **SMILES** string directly\n\n"
                "Then explore 11 analysis tabs including 3D viewer, batch processing, "
                "drug repurposing, stereochemistry analysis, similarity networks, and more.")
    st.stop()

if not RDKIT_OK:
    st.error("RDKit not installed. Check requirements.txt and packages.txt."); st.stop()

mol = Chem.MolFromSmiles(smiles_input.strip())
if mol is None:
    st.error(f"❌ Invalid SMILES: `{smiles_input.strip()}`"); st.stop()

data = predict_admet(smiles_input.strip())
if "error" in data: st.error(data["error"]); st.stop()

sc = synthetic_complexity(mol)

if st.session_state.lookup_name:
    compound_name = st.session_state.lookup_name
elif not is_hdr and chosen in EXAMPLES:
    compound_name = chosen
else:
    compound_name = SMILES_TO_NAME.get(smiles_input.strip(), "Custom Compound")

hist_entry = {"name": compound_name, "smiles": smiles_input.strip(),
              "qed": data["drug_likeness"]["qed"], "mw": data["molecular_weight"]}
if not st.session_state.history or st.session_state.history[-1]["smiles"] != smiles_input.strip():
    st.session_state.history.append(hist_entry)
    if len(st.session_state.history) > 20:
        st.session_state.history = st.session_state.history[-20:]

mw   = data["molecular_weight"]; logp = data["logP"]
hbd  = data["h_bond_donors"];    hba  = data["h_bond_acceptors"]
tpsa = data["tpsa"]; ro5 = data["lipinski"]; veber = data["veber_rules"]


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(f'<p class="compound-name">{compound_name}</p>', unsafe_allow_html=True)
st.caption(f"`{smiles_input.strip()}`")
st.divider()

col_img, col_kpi = st.columns([1,2], gap="large")
mol_png_bytes = mol_to_png(mol)
with col_img:
    try: st.image(Image.open(io.BytesIO(mol_png_bytes)), use_container_width=True)
    except Exception as e: st.warning(f"Render error: {e}")

with col_kpi:
    st.markdown("### At a Glance")
    k1,k2,k3 = st.columns(3)
    k1.metric("Mol. Weight",f"{mw} Da"); k2.metric("LogP",logp); k3.metric("QED",data["drug_likeness"]["qed"])
    k4,k5,k6 = st.columns(3)
    k4.metric("TPSA",f"{tpsa} Ų"); k5.metric("HBD/HBA",f"{hbd}/{hba}"); k6.metric("Synth. Score",f"{sc['score']}/10")
    st.markdown(
        "**Ro5:** " + (_b("Pass","pass") if ro5["pass"] else _b(f"{ro5['violations']} viol.","fail")) +
        " &nbsp;|&nbsp; **Veber:** " + (_b("Pass","pass") if veber["pass"] else _b("Fail","fail")) +
        " &nbsp;|&nbsp; **Synth.:** " + _b(sc["label"], sc["color"]),
        unsafe_allow_html=True)
    if REPORTLAB_OK:
        pdf_bytes = generate_pdf(compound_name, smiles_input.strip(), data, sc, mol_png_bytes)
        if pdf_bytes:
            st.download_button("📄 Download PDF Report", pdf_bytes,
                               file_name=f"{compound_name.replace(' ','_')}_report.pdf",
                               mime="application/pdf")
    else:
        st.caption("Install reportlab for PDF export.")

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

(tab_dl, tab_admet, tab_tox, tab_cov, tab_3d, tab_sim,
 tab_rep, tab_stereo, tab_msds, tab_cmp, tab_batch, tab_db) = st.tabs([
    "💊 Drug-likeness","🔬 ADMET","☠️ Toxicity","⚡ Covalent",
    "🧊 3D Structure","🔗 Similar Drugs","🔄 Repurposing",
    "🧬 Stereo & Tautomers","🧪 MSDS & Safety","📊 Compare","📦 Batch","🔍 Databases"
])


# ═══ TAB 1 — Drug-likeness ═══════════════════════════════════════════════════
with tab_dl:
    st.subheader("Drug-likeness Analysis")
    _xp("Estimates whether a molecule can survive the journey from pill to bloodstream. "
        "Failing one rule doesn't doom a drug — many approved drugs break one — but multiple failures are a warning sign.")
    cl, cr = st.columns(2)
    with cl:
        st.markdown("#### Lipinski Rule of Five")
        st.markdown('<p class="cite">Lipinski et al., Adv. Drug Deliv. Rev. 1997 — Experimental and computational approaches to estimate solubility and permeability in drug discovery.</p>', unsafe_allow_html=True)
        for label, val, passed, expl_pass, expl_fail in [
            ("Molecular Weight", f"{mw} Da", mw<=500,
             f"✅ {mw} Da — within 500 Da. Light enough to cross cell membranes and gut walls.",
             f"❌ {mw} Da — too heavy. Can't passively cross membranes. Trim non-essential groups or use injection."),
            ("LogP", logp, logp<=5,
             f"✅ LogP {logp} — {'very hydrophilic: may struggle to enter cells.' if logp<-1 else 'good balance: dissolves in water AND crosses fatty membranes.'}",
             f"❌ LogP {logp} — too greasy. Accumulates in fat, rapidly metabolised, often toxic."),
            ("H-Bond Donors", hbd, hbd<=5,
             f"✅ {hbd} donor(s) — acceptable. OH/NH groups help solubility without blocking membrane crossing.",
             f"❌ {hbd} donors — too many. Each OH/NH traps the molecule in water; it can't cross the gut wall."),
            ("H-Bond Acceptors", hba, hba<=10,
             f"✅ {hba} acceptor(s) — acceptable.",
             f"❌ {hba} acceptors — too many polar groups. Too polar to cross membranes efficiently."),
        ]:
            st.markdown(_rr(label, val, passed), unsafe_allow_html=True)
            _xp(expl_pass if passed else expl_fail)

        v = ro5["violations"]
        [st.success,st.info,st.warning,st.error][min(v,3)](
            ["🏆 0 violations — perfect oral candidate.",
             "ℹ️ 1 violation — borderline, many approved drugs break one.",
             "⚠️ 2 violations — oral bioavailability uncertain.",
             "🚫 3+ violations — likely needs IV delivery."][min(v,3)])

        # Lipinski fix suggestions
        fixes = lipinski_fix_suggestions(data)
        if fixes:
            st.markdown("#### 🔧 Chemistry Fix Suggestions")
            _xp("Concrete suggestions to address each violation. These are common medicinal chemistry transformations — "
                "actual potency effects require experimental validation.")
            for fix in fixes:
                with st.expander(f"Fix {fix['prop']} ({fix['val']})"):
                    for i, suggestion in enumerate(fix["fixes"], 1):
                        st.markdown(f'<div class="fix-box">{i}. {suggestion}</div>',
                                    unsafe_allow_html=True)
        else:
            st.success("✅ No Lipinski violations — no fixes needed.")

    with cr:
        st.markdown("#### Veber Rules")
        st.markdown('<p class="cite">Veber et al., J. Med. Chem. 2002 — Molecular properties that influence the oral bioavailability of drug candidates.</p>', unsafe_allow_html=True)
        rb = data["rotatable_bonds"]
        st.markdown(_rr("Rotatable Bonds", rb, rb<=10), unsafe_allow_html=True)
        _xp(f"{'✅' if rb<=10 else '❌'} {rb} bond(s) — "
            + ("not too floppy. Good oral absorption." if rb<=10
               else "too flexible. Constrain with rings or double bonds."))
        st.markdown(_rr("TPSA", f"{tpsa} Ų", tpsa<=140, ""), unsafe_allow_html=True)
        _xp(f"{'✅' if tpsa<=60 else '✅' if tpsa<=90 else '✅' if tpsa<=140 else '❌'} TPSA {tpsa} Ų — "
            + ("very low: excellent permeability, potential CNS activity." if tpsa<=60
               else "ideal 60–90 Ų: high oral absorption and possible CNS activity." if tpsa<=90
               else "passes Veber but CNS penetration unlikely. Fine for peripheral drugs." if tpsa<=140
               else "too polar. Reduce amides, acids, and hydroxyls."))

        st.markdown("#### QED Score")
        st.markdown('<p class="cite">Bickerton et al., Nat. Chem. 2012 — Quantifying the chemical beauty of drugs.</p>', unsafe_allow_html=True)
        qed_val = data["drug_likeness"]["qed"]
        st.progress(qed_val, text=f"QED = {qed_val}  ({data['drug_likeness']['interpretation']})")
        _xp(f"Combines 8 descriptors into a 0–1 score (approved oral drug average ≈ 0.67). "
            + (f"✅ {qed_val} — closely resembles approved drugs." if qed_val>=0.7
               else f"⚠️ {qed_val} — some properties deviate from drug space." if qed_val>=0.5
               else f"⚠️ {qed_val} — significant optimisation needed." if qed_val>=0.3
               else f"❌ {qed_val} — far from approved drug space."))

        st.markdown("#### Synthetic Complexity")
        st.markdown('<p class="cite">Ertl & Schuffenhauer, J. Cheminform. 2009 — Estimation of synthetic accessibility score of drug-like molecules.</p>', unsafe_allow_html=True)
        _xp("Estimates synthesis difficulty (1=trivial, 10=near-impossible) from stereocentres, "
            "spiro atoms, bridgeheads, ring sizes, and heavy atom count.")
        st.progress(sc["score"]/10, text=f"Score {sc['score']}/10 — {sc['label']}")
        if sc["factors"]:
            _xp("Complexity driven by: " + ", ".join(sc["factors"]) + ".")
        else:
            _xp("✅ No major complexity drivers. Straightforward synthetic target.")

        st.markdown("#### Murcko Scaffold")
        _xp("The Murcko scaffold is the ring systems + linkers stripped of all substituents. "
            "Drugs sharing a scaffold often share a target class — the basis of scaffold hopping.")
        scaf_mol, gen_mol, scaf_smi, gen_smi = get_murcko_scaffold(mol)
        if scaf_smi:
            st.code(scaf_smi, language=None)
            st.caption(f"Generic scaffold: `{gen_smi}`")
            matches = scaffold_drug_matches(scaf_smi, DRUG_DB)
            if matches:
                st.success(f"Approved drugs sharing this exact scaffold: **{', '.join(matches)}**")
            else:
                st.info("No exact scaffold match in the approved drug database. Potentially novel scaffold.")
            try:
                scaf_img = mol_to_png(scaf_mol, w=280, h=200)
                st.image(Image.open(io.BytesIO(scaf_img)), width=200)
            except Exception:
                pass
        else:
            st.info("Scaffold extraction failed (molecule may be acyclic).")


# ═══ TAB 2 — ADMET ═══════════════════════════════════════════════════════════
with tab_admet:
    st.subheader("ADMET Predictions")
    _xp("ADMET = Absorption, Distribution, Metabolism, Excretion, Toxicity. "
        "Over 40% of drug candidates fail clinical trials due to poor ADMET. See ☠️ Toxicity tab for the T.")
    a1,a2 = st.columns(2)
    with a1:
        st.markdown("#### 🫁 Absorption")
        sol = data["estimated_solubility"]
        sol_pass = sol["category"] in ("Highly soluble","Soluble","Moderately soluble")
        st.markdown(_rr("Solubility (ESOL)",f"LogS={sol['log_s']}",sol_pass)+
                    f'<span style="color:#94a3b8;font-size:0.85rem;margin-left:8px">{sol["category"]}</span>',
                    unsafe_allow_html=True)
        st.markdown('<p class="cite">ESOL model: Delaney, J. Chem. Inf. Comput. Sci. 2004</p>', unsafe_allow_html=True)
        _xp({"Highly soluble":f"✅ LogS {sol['log_s']} — rapid dissolution, no barrier.",
             "Soluble":f"✅ LogS {sol['log_s']} — adequate for oral delivery.",
             "Moderately soluble":f"⚠️ LogS {sol['log_s']} — may need salt formation or micronisation.",
             "Low solubility":f"⚠️ LogS {sol['log_s']} — BCS Class II. Lipid formulations often needed.",
             "Poorly soluble":f"❌ LogS {sol['log_s']} — &lt;10 µg/mL. Low and variable oral bioavailability."
             }.get(sol["category"],f"LogS {sol['log_s']}"))
        st.markdown(_rr("Lipinski Ro5","Pass" if ro5["pass"] else "Fail",ro5["pass"]),unsafe_allow_html=True)
        st.markdown(_rr("Veber Rules","Pass" if veber["pass"] else "Fail",veber["pass"]),unsafe_allow_html=True)
        st.markdown("#### 🔄 Metabolism")
        ar = data["aromatic_rings"]; fsp3 = data["fsp3"]
        if logp>3 and ar>=3:
            st.markdown(_b("Rapid hepatic metabolism risk","warn"),unsafe_allow_html=True)
            _xp(f"⚠️ LogP {logp} + {ar} aromatic rings → strong CYP450 substrate. Consider reducing lipophilicity.")
        elif fsp3<0.2:
            st.markdown(_b("Flat molecule — metabolic risk","warn"),unsafe_allow_html=True)
            _xp(f"⚠️ Fsp³ {fsp3} — flat molecules metabolised faster. Add sp³ carbons.")
        else:
            st.markdown(_b("Moderate metabolic liability","pass"),unsafe_allow_html=True)
            _xp("✅ No strong flags for rapid hepatic clearance.")
    with a2:
        st.markdown("#### 🩸 Distribution (BBB)")
        bbb = data["bbb_penetration"]; bs = bbb["score"]
        st.markdown(_rr("BBB Score",f"{bs}/4",bs>=3)+
                    f'<span style="color:#94a3b8;font-size:0.85rem;margin-left:8px">{bbb["prediction"]}</span>',
                    unsafe_allow_html=True)
        for crit,met in [(f"MW<450 ({mw}Da)",mw<450),(f"LogP 1–3 ({logp})",1<=logp<=3),
                         (f"TPSA<90 ({tpsa})",tpsa<90),(f"HBD≤3 ({hbd})",hbd<=3)]:
            st.markdown(" "+_b(crit,"pass" if met else "fail"),unsafe_allow_html=True)
        _xp(f"Score {bs}/4 — " +
            ("✅ Likely CNS penetrant. Great for CNS drugs; may cause neurological side effects in peripheral drugs." if bs>=3
             else "⚠️ Moderate penetration — not reliable for CNS targets." if bs==2
             else "ℹ️ Unlikely to cross BBB. Good for peripheral drugs."))
        st.markdown("#### 🔃 Excretion")
        if mw<300 and logp<1:
            _xp(f"✅ MW {mw}, LogP {logp} — small and hydrophilic. Direct renal clearance. May need frequent dosing.")
        elif logp>4:
            _xp(f"⚠️ LogP {logp} — hepatic metabolism required before excretion. Watch for active metabolites.")
        else:
            _xp("✅ Balanced lipophilicity — mixed renal/hepatic excretion. Common in approved oral drugs.")
        with st.expander("📋 Full descriptor table"):
            rows=[("MW",mw,"Da","≤500"),("LogP",logp,"","−1 to 5"),("HBD",hbd,"","≤5"),
                  ("HBA",hba,"","≤10"),("TPSA",tpsa,"Ų","≤140"),("Rot.Bonds",data["rotatable_bonds"],"","≤10"),
                  ("Rings",data["rings"],"",""),("Aromatic rings",ar,"","≤3"),("Fsp³",fsp3,"",">0.25"),
                  ("QED",data["drug_likeness"]["qed"],"","0–1"),("LogS",sol["log_s"],"",">−3"),
                  ("BBB",f"{bs}/4","","≥3=CNS"),("Synth.Score",f"{sc['score']}/10","","1=easy")]
            st.dataframe(pd.DataFrame(rows,columns=["Property","Value","Unit","Range"]),
                         use_container_width=True, hide_index=True)


# ═══ TAB 3 — Toxicity ════════════════════════════════════════════════════════
with tab_tox:
    st.subheader("☠️ Toxicity Assessment")
    _xp("Runs three screens: <b>PAINS</b> (false positives in assays), "
        "<b>hERG liability</b> (cardiac risk), and <b>structural toxicophores</b> "
        "(functional groups with known toxicity mechanisms).")
    t1,t2 = st.columns(2)
    with t1:
        st.markdown("#### PAINS Filter")
        st.markdown('<p class="cite">Baell & Holloway, J. Med. Chem. 2010 — New substructure filters for removal of pan assay interference compounds.</p>', unsafe_allow_html=True)
        pains = run_pains(mol)
        if pains:
            for p in pains: st.markdown(_b(f"PAINS: {p}","fail"),unsafe_allow_html=True)
            _xp("❌ PAINS pattern detected. Assay results may be artifactual (fluorescence, redox cycling, aggregation). Replace the flagged substructure before advancing.")
        else:
            st.success("✅ No PAINS alerts.")
            _xp("No pan-assay interference patterns. Results more likely reflect genuine target engagement.")
        st.markdown("#### hERG Liability (cardiac risk)")
        basic_n = Chem.MolFromSmarts("[NX3;H0,H1;!$(NC=O);!$(N~[#7,#8,S])]")
        herg_risk = mol.HasSubstructMatch(basic_n) and logp > 2 if basic_n else False
        if herg_risk:
            st.markdown(_b("Potential hERG liability","warn"),unsafe_allow_html=True)
            _xp(f"⚠️ Basic nitrogen + LogP {logp} > 2. hERG channel binding predicted. "
                "Run patch-clamp assay before advancing. Reduce basicity or lower LogP to mitigate.")
        else:
            st.success("✅ Low predicted hERG liability.")
            _xp("No basic N + high LogP combination detected.")
    with t2:
        st.markdown("#### Structural Toxicophores")
        tox_hits = detect_tox(mol)
        sev_icon = {"high":"🔴","moderate":"🟡","low":"🟢"}
        if not tox_hits:
            st.success(f"✅ No structural toxicophores in {compound_name}.")
        else:
            for h in tox_hits:
                st.markdown(f"**{sev_icon.get(h['severity'],'•')} {h['label']}** — `{h['severity'].upper()}`")
                _xp(h["explanation"])
        st.markdown("🔴 **High** — strong toxicity mechanism  \n🟡 **Moderate** — context-dependent  \n🟢 **Low** — usually manageable")


# ═══ TAB 4 — Covalent ════════════════════════════════════════════════════════
with tab_cov:
    st.subheader("Covalent Warhead Detection")
    _xp("Reactive groups that permanently bond to protein residues. Sometimes intentional (ibrutinib, aspirin). Non-selective reactivity causes toxicity.")
    hits = detect_warheads(mol)
    ri   = {"high":"🔴","moderate":"🟡","low":"🟢"}
    if not hits:
        st.success(f"✅ No covalent warheads in {compound_name}.")
    else:
        for h in hits:
            st.markdown(f"**{ri.get(h['reactivity'],'•')} {h['label']}** — `{h['reactivity'].upper()}`")
            _xp(h["mechanism"])


# ═══ TAB 5 — 3D Structure ════════════════════════════════════════════════════
with tab_3d:
    st.subheader("🧊 3D Structure Viewer")
    _xp("RDKit generates a 3D conformer using the ETKDG algorithm, then optimises it with the MMFF94 force field. "
        "The viewer uses py3Dmol (WebGL). Drag to rotate, scroll to zoom, right-click to change style.")
    if PY3DMOL_OK:
        style_choice = st.radio("Display style",
                                ["Stick + sphere","Stick only","Ball and stick","Spheres (CPK)","Surface"],
                                horizontal=True)
        render_3d(mol, style_choice)
        st.caption("⚠️ This is one low-energy conformer, not a crystal structure.")
    else:
        st.info("py3Dmol not installed. Add `py3Dmol` to requirements.txt.")


# ═══ TAB 6 — Similar Drugs ═══════════════════════════════════════════════════
with tab_sim:
    st.subheader("Similar Approved Drugs")
    _xp("Tanimoto similarity via Morgan fingerprints (radius=2, 2048 bits) against 30+ approved drugs. "
        "Tanimoto ≥0.7 = closely related scaffold; ≥0.4 = structurally related; <0.4 = dissimilar.")
    st.markdown('<p class="cite">Morgan, J. Chem. Doc. 1965 · Rogers & Hahn, J. Chem. Inf. Model. 2010</p>',
                unsafe_allow_html=True)

    similars = find_similar(mol, top=12)
    for i, item in enumerate(similars):
        s    = item["similarity"]
        tier = ("Same scaffold" if s>=0.85 else "Closely related" if s>=0.7
                else "Structurally related" if s>=0.4 else "Dissimilar")
        c1,c2,c3 = st.columns([2,4,1])
        c1.markdown(f"**{i+1}. {item['name']}**"); c1.caption(tier)
        c2.progress(s); c3.markdown(f"**{s:.3f}**")

    st.divider()
    st.markdown("#### Similarity Network")
    _xp("Interactive graph showing structural relationships. Blue = query molecule. "
        "Green nodes = similar (≥0.4), yellow = distant. Node size ∝ similarity score.")
    thresh = st.slider("Minimum similarity to display", 0.1, 0.8, 0.2, 0.05, key="sim_thresh")
    if PLOTLY_OK:
        fig = similarity_network(mol, compound_name, DRUG_DB, threshold=thresh)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No drugs with Tanimoto ≥ {thresh}. Try lowering the threshold.")
    else:
        st.info("Install plotly for the network graph: add `plotly` to requirements.txt.")


# ═══ TAB 7 — Drug Repurposing ════════════════════════════════════════════════
with tab_rep:
    st.subheader("🔄 Drug Repurposing Finder")
    _xp("Structural similarity to approved drugs suggests shared targets — the basis of drug repurposing programs. "
        "Aspirin was repurposed for antiplatelet therapy. Sildenafil was repurposed for PAH. "
        "This analysis highlights structurally related approved drugs and their known targets.")

    min_sim = st.slider("Minimum similarity for repurposing candidates", 0.1, 0.8, 0.25, 0.05, key="rep_thresh")
    candidates = [(item["name"], item["similarity"]) for item in find_similar(mol, top=30)
                  if item["similarity"] >= min_sim]

    if not candidates:
        st.info(f"No approved drugs with Tanimoto ≥ {min_sim}. Lower the threshold to see more candidates.")
    else:
        st.success(f"**{len(candidates)} repurposing candidate(s)** found at similarity ≥ {min_sim}")
        for drug_name, sim in candidates:
            info = DRUG_TARGETS.get(drug_name, {})
            targets     = info.get("targets", [])
            indications = info.get("indications", [])
            drug_class  = info.get("class", "approved drug")
            tier = "🟢 High" if sim>=0.7 else "🟡 Moderate" if sim>=0.4 else "🔵 Low"
            with st.expander(f"{tier} similarity — **{drug_name}** (Tanimoto {sim:.3f})"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Drug class:** {drug_class}")
                    if targets:
                        st.markdown(f"**Known targets:** {', '.join(targets)}")
                with c2:
                    if indications:
                        st.markdown(f"**Approved indications:** {', '.join(indications)}")
                    st.progress(sim, text=f"Structural similarity: {sim:.3f}")

                # Repurposing hypothesis
                hypo_targets = ", ".join(targets[:2]) if targets else "similar biological targets"
                hypo_indic   = ", ".join(indications[:2]) if indications else "related indications"
                st.info(
                    f"💡 **Repurposing hypothesis:** {compound_name} shares structural features with "
                    f"{drug_name} (a {drug_class}). If they share the same scaffold pharmacophore, "
                    f"{compound_name} may also modulate **{hypo_targets}**, suggesting potential activity in "
                    f"**{hypo_indic}**. Requires experimental validation."
                )

    if PLOTLY_OK and candidates:
        st.divider()
        st.markdown("#### Repurposing Landscape Network")
        fig = similarity_network(mol, compound_name, DRUG_DB, threshold=min_sim)
        if fig:
            st.plotly_chart(fig, use_container_width=True)


# ═══ TAB 8 — Stereo & Tautomers ══════════════════════════════════════════════
with tab_stereo:
    st.subheader("🧬 Stereochemistry & Tautomers")
    st.markdown('<p class="cite">R/S assignment: CIP rules (Cahn, Ingold & Prelog 1966) · '
                'Stereoisomer enumeration: RDKit EnumerateStereoisomers · '
                'Tautomers: RDKit MolStandardize (Sayle 2010)</p>', unsafe_allow_html=True)

    stereo_res = analyze_stereo(mol)

    # ── Stereochemistry overview ───────────────────────────────────────────────
    st.markdown("#### Stereochemistry")
    _xp("Chirality is critical in drug development. Thalidomide's R-enantiomer is sedative; "
        "its S-enantiomer is teratogenic. Undefined stereocenters are a regulatory red flag — "
        "the FDA requires specification of absolute configuration in INDs and NDAs.")

    if not stereo_res["is_chiral"]:
        st.success("✅ Achiral molecule — no stereocenters or stereospecific double bonds.")
        _xp("Identical to its mirror image (superimposable). "
            "No enantiomer separation needed; a single synthesis route yields one compound.")
    else:
        km1, km2, km3, km4 = st.columns(4)
        km1.metric("Total centers",    stereo_res["total"])
        km2.metric("Defined (R/S)",    len(stereo_res["defined"]),
                   help="Absolute configuration assigned")
        km3.metric("Undefined ⚠️",     len(stereo_res["undefined"]),
                   help="Configuration not yet specified — creates multiple isomers")
        km4.metric("Possible isomers", stereo_res["max_isomers"],
                   help="2ⁿ where n = undefined centers")

        # R/S table
        if stereo_res["defined"]:
            st.markdown("**Defined stereocenters (R/S):**")
            rows_rs = []
            for idx, config in stereo_res["defined"]:
                atom = mol.GetAtomWithIdx(idx)
                rows_rs.append({
                    "Atom index": idx,
                    "Element":    atom.GetSymbol(),
                    "Configuration": config,
                    "Meaning": ("Right-handed (clockwise CIP priority order)"
                                if config == "R" else
                                "Left-handed (counter-clockwise CIP priority order)"),
                })
            st.dataframe(pd.DataFrame(rows_rs), use_container_width=True, hide_index=True)

        if stereo_res["undefined"]:
            st.markdown(_b(f"⚠️ {len(stereo_res['undefined'])} undefined stereocenters","warn"),
                        unsafe_allow_html=True)
            _xp(f"**{stereo_res['max_isomers']} possible stereoisomers** arise from the undefined centers. "
                "Each may have a completely different pharmacological profile. "
                "Assign absolute configuration via X-ray crystallography or chiral HPLC-CD before advancing.")

        if stereo_res["db_stereo"]:
            st.markdown("**Double bond geometry (E/Z):**")
            for bond_idx, config in stereo_res["db_stereo"]:
                meaning = ("E = substituents on opposite sides (trans-like)"
                           if config == "E" else
                           "Z = substituents on same side (cis-like)")
                st.markdown(f"  - Bond {bond_idx}: **{config}** — {meaning}")

    # ── Enumerate all stereoisomers ────────────────────────────────────────────
    if stereo_res["is_chiral"]:
        st.markdown("#### All Stereoisomers")
        _xp("RDKit EnumerateStereoisomers generates every possible stereoisomer by permuting "
            "all undefined (and defined) chiral centers. Structures capped at 16.")

        all_isomers = enumerate_stereoisomers(mol, max_isomers=16)
        input_smi   = Chem.MolToSmiles(mol, isomericSmiles=True)

        if not all_isomers:
            st.info("Stereoisomer enumeration returned no results.")
        else:
            st.success(f"**{len(all_isomers)} stereoisomer(s) enumerated** "
                       f"(theoretical max: {stereo_res['max_isomers']})")

            cols_per_row = 3
            for row_start in range(0, len(all_isomers), cols_per_row):
                row_cols = st.columns(cols_per_row)
                for col_i, (smi, iso_mol) in enumerate(
                        all_isomers[row_start:row_start + cols_per_row]):
                    with row_cols[col_i]:
                        is_input = (smi == input_smi)
                        label = f"Isomer {row_start + col_i + 1}"
                        if is_input:
                            label += " ★ (your input)"
                        st.markdown(f"**{label}**")
                        try:
                            img_b = mol_to_png(iso_mol, w=260, h=190)
                            st.image(Image.open(io.BytesIO(img_b)), use_container_width=True)
                        except Exception:
                            pass
                        st.code(smi, language=None)
                        # Show R/S assignments for this isomer
                        iso_centers = Chem.FindMolChiralCenters(
                            iso_mol, includeUnassigned=True)
                        if iso_centers:
                            rs_str = ", ".join(
                                f"{sym}({c})" for _, c in iso_centers
                                for sym in [iso_mol.GetAtomWithIdx(_).GetSymbol()])
                            st.caption(f"Centers: {rs_str}")
                        if is_input:
                            st.success("★ This is your input structure")

    st.divider()

    # ── Tautomers ─────────────────────────────────────────────────────────────
    st.markdown("#### Tautomer Enumeration")
    _xp("Tautomers rapidly interconvert in solution by proton transfer. "
        "The biologically active tautomer may differ from the form you drew. "
        "Key example: keto vs. enol forms, or 1H vs. 3H imidazole tautomers.")

    tautomers, canon_smi = enumerate_tautomers(mol)

    if not tautomers:
        st.info("Only one reasonable tautomeric form found.")
    else:
        st.markdown(f"**{len(tautomers)} tautomer(s) found** · "
                    f"Canonical (most stable): `{canon_smi[:70] if canon_smi else '—'}`")

        t_cols = st.columns(min(len(tautomers), 3))
        for i, (smi, tmol) in enumerate(tautomers[:6]):
            with t_cols[i % 3]:
                is_canon = (smi == canon_smi)
                badge = " ★ canonical" if is_canon else ""
                st.markdown(f"**Tautomer {i+1}{badge}**")
                try:
                    t_img = mol_to_png(tmol, w=240, h=180)
                    st.image(Image.open(io.BytesIO(t_img)), use_container_width=True)
                except Exception:
                    pass
                st.code(smi[:60] + ("…" if len(smi) > 60 else ""), language=None)
                if is_canon:
                    st.caption("★ Most stable tautomer")


# ═══ TAB 9 — MSDS & Safety ═══════════════════════════════════════════════════
with tab_msds:
    st.subheader("🧪 MSDS / Safety Data Sheet")
    _xp("Structure-based safety profile computed by RDKit. "
        "If you have already clicked <b>Fetch database records</b> in the Databases tab, "
        "official GHS data from PubChem will also appear below. "
        "<b>Always consult the supplier SDS before handling any chemical.</b>")

    tox_hits_msds = detect_tox(mol)
    msds = compute_msds_profile(mol, data, tox_hits_msds)

    # ── Signal word + hazard banner ───────────────────────────────────────────
    signal_color = "#dc2626" if msds["signal"] == "Danger" else "#d97706"
    st.markdown(
        f'<div style="background:{signal_color};color:#fff;padding:10px 18px;'
        f'border-radius:8px;font-size:1.4rem;font-weight:800;margin-bottom:12px">'
        f'⚠️ {msds["signal"].upper()}</div>',
        unsafe_allow_html=True)
    st.caption(msds["note"])

    col_left, col_right = st.columns(2)

    with col_left:
        # Physical & chemical properties
        st.markdown("#### Physical & Chemical Properties")
        phys_rows = [
            ("Molecular formula",  msds["formula"]),
            ("Molecular weight",   f"{data['molecular_weight']} Da"),
            ("Calculated logP",    f"{data['logP']} (Wildman-Crippen)"),
            ("TPSA",               f"{data['tpsa']} Ų"),
            ("H-bond donors",      str(data["h_bond_donors"])),
            ("H-bond acceptors",   str(data["h_bond_acceptors"])),
            ("Rotatable bonds",    str(data["rotatable_bonds"])),
            ("Heavy atom count",   str(msds["heavy_atoms"])),
            ("Est. solubility",    f"LogS = {data['estimated_solubility']['log_s']} "
                                   f"({data['estimated_solubility']['category']})"),
            ("Volatility estimate", msds["volatility"]),
            ("Environmental risk", msds["env_risk"]),
        ]
        st.dataframe(pd.DataFrame(phys_rows, columns=["Property","Value"]),
                     use_container_width=True, hide_index=True)

        # GHS Hazard statements
        st.markdown("#### GHS Hazard Statements (H-codes)")
        _xp("H-codes are UN GHS hazard statements. Prefixed [structural estimate] = "
            "computed from RDKit substructure analysis, not from experimental data.")
        for h in msds["ghs_hazards"]:
            severity_color = ("#f87171" if "Toxic" in h or "fatal" in h.lower()
                              else "#facc15" if "Harmful" in h or "Suspected" in h
                              else "#94a3b8")
            st.markdown(
                f'<div style="border-left:3px solid {severity_color};'
                f'padding:4px 10px;margin:3px 0;font-size:0.87rem">{h}</div>',
                unsafe_allow_html=True)

    with col_right:
        # First aid
        st.markdown("#### First Aid Measures")
        for route, action in msds["first_aid"].items():
            with st.expander(f"🩺 {route}"):
                st.markdown(action)

        # PPE
        st.markdown("#### Personal Protective Equipment")
        for item in msds["ppe"]:
            st.markdown(f"🛡️ {item}")

        # Storage
        st.markdown("#### Storage & Handling")
        for tip in msds["storage"]:
            st.markdown(f"📦 {tip}")

        # Disposal
        st.markdown("#### Disposal")
        _xp("Dispose of contents and container in accordance with local/national regulations. "
            "Do not pour down the drain. Contact a licensed hazardous waste disposal company. "
            "Never dispose of unknown chemicals without expert guidance.")

    # ── Official PubChem GHS (shown if CID is available from Databases tab) ───
    st.divider()
    st.markdown("#### Official GHS Data from PubChem")

    pc_cached = st.session_state.get("_db_pc")
    if pc_cached and pc_cached != "__not_fetched__" and pc_cached is not None:
        cid_for_ghs = pc_cached.get("CID")
        if cid_for_ghs:
            if st.button("Load official GHS from PubChem", key="load_ghs"):
                if _allowed("ghs"):
                    with st.spinner("Fetching GHS data from PubChem…"):
                        ghs = pubchem_ghs(int(cid_for_ghs))
                    st.session_state["_ghs"] = ghs
                else:
                    st.warning("Rate limit reached. Wait 60 s.")

            ghs = st.session_state.get("_ghs")
            if ghs:
                g1, g2 = st.columns(2)
                with g1:
                    if ghs.get("signal"):
                        st.markdown(f"**Signal word:** `{ghs['signal']}`")
                    if ghs.get("hazards"):
                        st.markdown("**H-codes (official):**")
                        for h in ghs["hazards"]:
                            st.markdown(f"- {h}")
                    if ghs.get("phys"):
                        st.markdown("**Experimental properties:**")
                        for k, v in ghs["phys"].items():
                            st.markdown(f"- **{k.replace('_',' ').title()}:** {v}")
                with g2:
                    if ghs.get("precautions"):
                        st.markdown("**P-codes (precautionary statements):**")
                        for p in ghs["precautions"]:
                            st.markdown(f"- {p}")
            elif ghs is not None:
                st.info("No GHS data found in PubChem for this compound's CID.")
    else:
        st.info("Go to the **🔍 Databases** tab and click **Fetch database records** first — "
                "this unlocks the official PubChem GHS hazard classification.")


# ═══ TAB 9 — Compare ═════════════════════════════════════════════════════════
with tab_cmp:
    st.subheader("📊 Side-by-side Comparison")
    _xp("Add up to 3 additional compounds and compare all key properties in a colour-coded table. "
        "The current compound is always shown first. Green = best value, red = worst.")
    cmp_entries = [{"name": compound_name, "smiles": smiles_input.strip()}]
    cols = st.columns(3)
    for i, col in enumerate(cols):
        with col:
            n = col.text_input(f"Compound {i+2} name", key=f"cmp_name_{i}", placeholder=f"e.g. Naproxen")
            s = col.text_area(f"SMILES {i+2}", key=f"cmp_smi_{i}", height=70, placeholder="paste SMILES…")
            if s.strip():
                cmp_entries.append({"name": n or f"Compound {i+2}", "smiles": s.strip()})
    if st.button("Compare", type="primary"):
        rows = []
        for entry in cmp_entries:
            m = Chem.MolFromSmiles(entry["smiles"])
            if m is None:
                st.warning(f"Invalid SMILES for {entry['name']}"); continue
            d = predict_admet(entry["smiles"])
            if "error" in d: continue
            sc2 = synthetic_complexity(m)
            rows.append({
                "Compound": entry["name"], "MW (Da)": d["molecular_weight"],
                "LogP": d["logP"], "QED": d["drug_likeness"]["qed"],
                "TPSA": d["tpsa"], "HBD": d["h_bond_donors"], "HBA": d["h_bond_acceptors"],
                "Rot. Bonds": d["rotatable_bonds"],
                "Ro5 Pass": "✅" if d["lipinski"]["pass"] else "❌",
                "Veber Pass": "✅" if d["veber_rules"]["pass"] else "❌",
                "BBB": d["bbb_penetration"]["prediction"],
                "Solubility": d["estimated_solubility"]["category"],
                "Synth. Score": sc2["score"],
            })
        if rows:
            df_cmp = pd.DataFrame(rows).set_index("Compound")
            num_cols = ["MW (Da)","LogP","QED","TPSA","HBD","HBA","Rot. Bonds","Synth. Score"]
            def highlight_best(col):
                if col.name in ("QED",):       best, worst = col.max(), col.min()
                elif col.name in num_cols:      best, worst = col.min(), col.max()
                else:                           return [""]*len(col)
                return ["background-color:#14532d" if v==best
                        else "background-color:#7f1d1d" if v==worst
                        else "" for v in col]
            styled = df_cmp.style.apply(highlight_best, subset=num_cols, axis=0)
            st.dataframe(styled, use_container_width=True)
            st.download_button("⬇️ Download CSV", df_cmp.reset_index().to_csv(index=False),
                               "comparison.csv", "text/csv")


# ═══ TAB 10 — Batch ══════════════════════════════════════════════════════════
with tab_batch:
    st.subheader("📦 Batch Processing")
    _xp("Upload a CSV with a <b>smiles</b> column (and optional <b>name</b> column). "
        "All compounds are analysed in one go. This is how real medicinal chemists screen compound libraries.")
    sample_csv = "name,smiles\nAspirin,CC(=O)Oc1ccccc1C(=O)O\nIbuprofen,CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O\nMetformin,CN(C)C(=N)NC(=N)N\n"
    st.download_button("⬇️ Download sample CSV", sample_csv, "sample.csv", "text/csv")
    uploaded = st.file_uploader("Upload your CSV", type=["csv"])
    if uploaded:
        try:
            df_in = pd.read_csv(uploaded)
            st.write(f"Loaded {len(df_in)} rows. Preview:")
            st.dataframe(df_in.head(), use_container_width=True)
            if st.button("Run batch analysis", type="primary"):
                prog = st.progress(0, text="Analysing…")
                results = []
                total      = len(df_in)
                smiles_col = next((c for c in df_in.columns if c.lower() in ("smiles","smile","smi")), df_in.columns[0])
                name_col   = next((c for c in df_in.columns if c.lower() in ("name","compound","id")), None)
                for i, row in df_in.iterrows():
                    smi  = str(row[smiles_col]).strip()
                    name = str(row[name_col]) if name_col else f"Compound_{i+1}"
                    d    = predict_admet(smi)
                    if "error" in d:
                        results.append({"Name":name,"SMILES":smi,"Error":d["error"]}); continue
                    m   = Chem.MolFromSmiles(smi)
                    sc2 = synthetic_complexity(m) if m else {}
                    results.append({"Name":name,"SMILES":smi,
                        "MW":d["molecular_weight"],"LogP":d["logP"],"HBD":d["h_bond_donors"],
                        "HBA":d["h_bond_acceptors"],"TPSA":d["tpsa"],"RotBonds":d["rotatable_bonds"],
                        "QED":d["drug_likeness"]["qed"],"Ro5Pass":d["lipinski"]["pass"],
                        "Ro5Violations":d["lipinski"]["violations"],"VerberPass":d["veber_rules"]["pass"],
                        "BBB":d["bbb_penetration"]["prediction"],"LogS":d["estimated_solubility"]["log_s"],
                        "Solubility":d["estimated_solubility"]["category"],
                        "SynthScore":sc2.get("score",""),"SynthLabel":sc2.get("label","")})
                    prog.progress((i+1)/total, text=f"Analysed {i+1}/{total}")
                df_out = pd.DataFrame(results)
                st.success(f"Done! {len(df_out)} compounds analysed.")
                st.dataframe(df_out, use_container_width=True)
                st.download_button("⬇️ Download results CSV", df_out.to_csv(index=False),
                                   "batch_results.csv","text/csv")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")


# ═══ TAB 11 — Databases ══════════════════════════════════════════════════════
with tab_db:
    st.subheader("🔍 PubChem · ChEMBL · PubMed · Where to Buy")
    _xp("Click <b>Fetch database records</b> to query external databases. "
        "Rate-limited to protect the service. Results are cached for 5 minutes per compound.")

    # Clear cached results when SMILES changes
    if st.session_state.get("_db_smiles") != smiles_input.strip():
        for k in ("_db_pc","_db_chembl","_db_pubmed"):
            st.session_state.pop(k, None)
        st.session_state["_db_smiles"] = smiles_input.strip()

    if st.button("🔍 Fetch database records", type="primary"):
        if _allowed("db"):
            # ── Step 1: PubChem ───────────────────────────────────────────────
            pc = None
            pc_method = ""
            with st.spinner("Querying PubChem via SMILES (POST + GET fallback)…"):
                pc = pubchem_by_smiles(smiles_input.strip())
                pc_method = "SMILES"

            # Auto-fallback: try name search if SMILES failed
            if pc is None and compound_name != "Custom Compound":
                with st.spinner(f"SMILES lookup failed — retrying by name '{compound_name}'…"):
                    pc = pubchem_by_name(compound_name)
                    pc_method = f"name '{compound_name}'"

            st.session_state["_db_pc"]        = pc
            st.session_state["_db_pc_method"] = pc_method if pc else "all methods failed"

            # ── Step 2: ChEMBL ────────────────────────────────────────────────
            ik        = pc.get("InChIKey") if pc else None
            name_hint = pc.get("IUPACName", compound_name) if pc else compound_name
            with st.spinner("Querying ChEMBL (InChIKey + name fallback)…"):
                chembl = chembl_by_inchikey(ik, name_hint)
            st.session_state["_db_chembl"] = chembl

            # ── Step 3: PubMed ────────────────────────────────────────────────
            search_q = (pc.get("IUPACName", compound_name) if pc else compound_name)
            if search_q and search_q != "Custom Compound":
                with st.spinner(f"Searching PubMed for '{search_q[:40]}'…"):
                    articles = pubmed_search(search_q + " drug")
                st.session_state["_db_pubmed"] = articles
            else:
                st.session_state["_db_pubmed"] = []
        else:
            st.warning("⚠️ Rate limit: max 15 API calls per minute. Please wait a moment.")

    pc      = st.session_state.get("_db_pc",     "__not_fetched__")
    chembl  = st.session_state.get("_db_chembl", "__not_fetched__")
    articles = st.session_state.get("_db_pubmed", "__not_fetched__")

    if pc == "__not_fetched__":
        st.info("Click **Fetch database records** above to query PubChem, ChEMBL, and PubMed.")
    else:
        # ── PubChem ──────────────────────────────────────────────────────────
        st.markdown("#### PubChem")
        if pc:
            cid = pc.get("CID")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**CID:** [{cid}](https://pubchem.ncbi.nlm.nih.gov/compound/{cid})")
                st.markdown(f"**IUPAC name:** {pc.get('IUPACName','—')}")
                st.markdown(f"**Formula:** {pc.get('MolecularFormula','—')}  ·  **MW:** {pc.get('MolecularWeight','—')} Da")
                st.markdown(f"**InChIKey:** `{pc.get('InChIKey','—')}`")
            with c2:
                syns = pubchem_synonyms(cid)
                if syns:
                    st.markdown("**Synonyms / trade names:**")
                    for s in syns[:6]: st.markdown(f"  - {s}")
        else:
            method_tried = st.session_state.get("_db_pc_method", "unknown")
            st.warning(
                f"⚠️ PubChem returned no match ({method_tried}). "
                "This is normal for novel or custom molecules not yet registered in PubChem."
            )
            with st.expander("🔍 Diagnostic info"):
                st.markdown("**What was tried:**")
                st.markdown(
                    f"1. **SMILES POST** → `POST /compound/smiles/property/…/JSON`  \n"
                    f"   Body: `smiles={smiles_input.strip()}`\n\n"
                    f"2. **SMILES GET** → `GET /compound/smiles/{{url-encoded}}/property/…/JSON`\n\n"
                    f"3. **Name fallback** → `GET /compound/name/{compound_name}/property/…/JSON`"
                )
                st.markdown("**Common reasons for failure:**")
                st.markdown(
                    "- Compound not registered in PubChem (novel molecule ✓ expected)\n"
                    "- SMILES has unusual ring notation that PubChem can't parse\n"
                    "- Transient network timeout on Streamlit Cloud — click the button again\n"
                    "- Non-standard isotope labels or charges in SMILES"
                )
                st.markdown("**Try this:**")
                st.markdown(
                    "Type the drug name in the **sidebar name search** (e.g. 'aspirin') — "
                    "this uses a different API endpoint and is more reliable for known drugs."
                )

        st.divider()

        # ── ChEMBL ───────────────────────────────────────────────────────────
        st.markdown("#### ChEMBL")
        if chembl:
            cid2  = chembl.get("molecule_chembl_id","—")
            phase_map = {0:"Preclinical",1:"Phase I",2:"Phase II",3:"Phase III",4:"Approved"}
            phase = phase_map.get(chembl.get("max_phase") or 0,"—")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**ChEMBL ID:** [{cid2}](https://www.ebi.ac.uk/chembl/compound_report_card/{cid2}/)")
                st.markdown(f"**Preferred name:** {chembl.get('pref_name','—')}")
                st.markdown(f"**Max clinical phase:** {phase}")
                st.markdown(f"**Molecule type:** {chembl.get('molecule_type','—')}")
            with c2:
                props = chembl.get("molecule_properties") or {}
                st.markdown(f"**Oral:** {'✅ Yes' if chembl.get('oral') else 'No'}")
                st.markdown(f"**Black box warning:** {'⚠️ Yes' if chembl.get('black_box_warning') else 'No'}")
                st.markdown(f"**First approval:** {chembl.get('first_approval','—')}")
                if props.get("alogp"):
                    st.markdown(f"**ALogP:** {props['alogp']}")
        elif chembl is None and pc:
            st.info("No ChEMBL record found via InChIKey or name. "
                    "The compound may be a research tool not yet in ChEMBL, or still in preclinical development.")
        elif chembl is None and not pc:
            st.info("ChEMBL lookup attempted but PubChem also returned no data — "
                    "both InChIKey and name searches were tried. "
                    "If this is a known drug, use the sidebar name search first.")

        st.divider()

        # ── PubMed literature ────────────────────────────────────────────────
        if articles != "__not_fetched__":
            st.markdown("#### PubMed Literature")
            if articles:
                st.markdown(f"**{len(articles)} recent paper(s)** mentioning this compound:")
                for art in articles:
                    st.markdown(
                        f"**[{art['title'][:120]}{'...' if len(art['title'])>120 else ''}]({art['url']})**  \n"
                        f"<span style='color:#94a3b8;font-size:0.85rem'>"
                        f"{art['authors']} · {art['journal']} · {art['year']}</span>",
                        unsafe_allow_html=True)
                    st.markdown("---")
            else:
                st.info("No PubMed papers found. The compound may be very novel or have a different common name.")

    st.divider()

    # ── Where to Buy ─────────────────────────────────────────────────────────
    st.markdown("#### 🛒 Where to Buy")
    _xp("Search links for major lab chemical suppliers. For research use only. "
        "Always verify purity, safety data (SDS), and legal restrictions before ordering.")
    sn = urllib.parse.quote(compound_name if compound_name != "Custom Compound"
                            else smiles_input.strip()[:40])
    for sup_name, url, desc in [
        ("Sigma-Aldrich (Merck)",f"https://www.sigmaaldrich.com/US/en/search#q={sn}",
         "World's largest chemistry supplier — broad catalog, rigorous purity standards."),
        ("Cayman Chemical",     f"https://www.caymanchem.com/search?term={sn}",
         "Specialises in bioactive lipids, eicosanoids, and pharmacological standards."),
        ("MedChemExpress (MCE)",f"https://www.medchemexpress.com/search.html?q={sn}",
         "Drug-like inhibitors, approved drugs, and tool compounds for research."),
        ("Tocris Bioscience",   f"https://www.tocris.com/search#query={sn}",
         "High-purity pharmacological tools for receptor and ion channel biology."),
        ("Fisher Scientific",  f"https://www.fishersci.com/us/en/catalog/search/products?query={sn}",
         "Analytical-grade and reference standards; broad lab supply catalog."),
    ]:
        st.markdown(
            f'<div class="buy-card"><b><a href="{url}" target="_blank" style="color:#38bdf8">'
            f'{sup_name} ↗</a></b><br>'
            f'<span style="font-size:0.85rem;color:#94a3b8">{desc}</span></div>',
            unsafe_allow_html=True)
