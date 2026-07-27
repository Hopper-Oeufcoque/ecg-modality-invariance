"""PTB-XL data loading for the simulator-validation experiments.

Loads the 100 Hz subset, maps to the 5 superdiagnostic classes
(NORM, MI, STTC, CD, HYP) following the standard PTB-XL benchmark
(Wagner et al. 2020), and exposes a clean (signals, labels, ids) interface.

The 5 superdiagnostic classes are deliberately chosen because they separate
*lead-dependent* spatial pathologies (MI, STTC, HYP, CD) from the largely
rhythm-driven normal class — letting the simulator experiment show that
lead reduction hurts spatial classes more than rhythm, which is the central
sanity check for the forward-physics simulator.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

# The 5 PTB-XL superclasses (Wagner et al. 2020). The code->superclass map is
# read from scp_statements.csv (official diagnostic_class column) rather than
# hand-maintained, so it stays correct and complete (incl. STTC).
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]


def build_superclass_map(scp_csv: str | Path) -> dict[str, str]:
    """code -> diagnostic_class from scp_statements.csv."""
    df = pd.read_csv(scp_csv, index_col=0)
    m = {}
    for code, row in df.iterrows():
        dc = row.get("diagnostic_class")
        if isinstance(dc, str) and dc in SUPERCLASSES:
            m[code] = dc
    return m


def load_ptbxl_meta(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["ecg_id"] = df["ecg_id"].astype(int)
    return df


def label_superclass(scp_codes_str: str, smap: dict) -> str | None:
    """Map an scp_codes dict (as a string) to a single superclass.

    Returns None if ambiguous (multiple distinct superclasses present) so
    we keep the benchmark clean rather than guessing.
    """
    import ast
    try:
        codes = ast.literal_eval(scp_codes_str)
    except Exception:
        return None
    supers = set()
    for code in codes:
        if code in smap:
            supers.add(smap[code])
    if len(supers) == 1:
        return supers.pop()
    return None


def build_split(df: pd.DataFrame, data_root: str | Path,
                smap: dict,
                fs_col: str = "filename_lr",
                max_per_class: int | None = 1500,
                strat_fold_col: str = "strat_fold") -> dict:
    """Return {'train':..., 'val':..., 'test':...} each a list of records.

    Each record: dict(ecg ndarray [T,12], label str, ecg_id int).
    Uses PTB-XL strat_fold: 1-7 train, 8 val, 9-10 test (standard split).
    data_root is the PTB-XL root dir (filename_lr is relative to it, e.g.
    'records100/00000/00001_lr').
    """
    data_root = Path(data_root)
    df = df.copy()
    df["super"] = df["scp_codes"].apply(lambda s: label_superclass(s, smap))
    df = df[df["super"].notna()].reset_index(drop=True)

    # balance classes
    if max_per_class is not None:
        parts = []
        for cls in SUPERCLASSES:
            sub = df[df["super"] == cls]
            parts.append(sub.head(max_per_class))
        df = pd.concat(parts).sample(frac=1.0, random_state=0).reset_index(drop=True)

    splits = {"train": [], "val": [], "test": []}
    for _, row in df.iterrows():
        fold = int(row[strat_fold_col])
        if fold <= 7:
            split = "train"
        elif fold == 8:
            split = "val"
        else:
            split = "test"
        rec_path = str(data_root / row[fs_col])
        try:
            sig, _ = wfdb.rdsamp(rec_path)
        except Exception:
            continue
        # sig shape (T, 12), units mV, fs 100
        splits[split].append({
            "ecg": sig.astype(np.float32),
            "label": row["super"],
            "ecg_id": int(row["ecg_id"]),
        })
    return splits


def load_all(data_dir: str | Path = None, max_per_class: int | None = 1500):
    """Convenience: load PTB-XL with standard config."""
    if data_dir is None:
        data_dir = os.path.expanduser("~/data/ptbxl")
    data_dir = Path(data_dir)
    df = load_ptbxl_meta(data_dir / "ptbxl_database.csv")
    smap = build_superclass_map(data_dir / "scp_statements.csv")
    return build_split(df, data_root=data_dir, smap=smap,
                       max_per_class=max_per_class)
