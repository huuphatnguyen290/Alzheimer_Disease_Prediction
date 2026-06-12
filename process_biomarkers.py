"""
process_biomarkers.py
---------------------
1. Normalizes PTAU/ABETA in the CSF data (min-max → 0–1).
2. Finds RIDs in Plasma but not in CSF.
3. Normalizes pT217_AB42_F for those plasma-only rows separately (min-max → 0–1).
4. Renames plasma columns to match CSF column names:
      VISCODE2     → VISCODE   (plasma VISCODE is dropped)
      EXAMDATE     → DRAWDTE
      AB42_F       → ABETA
      pT217_F      → PTAU
      pT217_AB42_F → PTAU/ABETA
5. Drops the TAU column from the final output.
6. Adds a "Source" column ("CSF" or "Plasma") to every row.
7. Filters to VISCODE = "bl" rows only.
8. Matches DX from ADNI_Cognitive.csv in two passes:
   a) Exact match on RID + DRAWDTE == RID + VISDATE.
   b) For still-missing DX: match on RID + VISCODE (against cognitive
      VISCODE or VISCODE2), accepting DX only when |DRAWDTE − VISDATE| < 30 days.
9. Saves the result as ADNI_CSF_biomarkers_processed.csv.

Usage
-----
    python process_biomarkers.py               # files in current directory
    python process_biomarkers.py /path/to/data  # files in a custom directory
"""

import sys
import os
import numpy as np
import pandas as pd


def main():
    # ── Resolve file directory ──────────────────────────────────────────
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    csf_path    = os.path.join(data_dir, "ADNI_CSF_biomarkers.csv")
    plasma_path = os.path.join(data_dir, "ADNI_Plasma_biomarkers.csv")
    cog_path    = os.path.join(data_dir, "ADNI_Cognitive.csv")
    output_path = os.path.join(data_dir, "ADNI_CSF_biomarkers_processed.csv")

    # ── Load data ───────────────────────────────────────────────────────
    csf    = pd.read_csv(csf_path)
    plasma = pd.read_csv(plasma_path)
    cog    = pd.read_csv(cog_path)

    print(f"Loaded CSF   file: {len(csf)} rows, {csf['RID'].nunique()} unique RIDs")
    print(f"Loaded Plasma file: {len(plasma)} rows, {plasma['RID'].nunique()} unique RIDs")
    print(f"Loaded Cog   file: {len(cog)} rows, {cog['RID'].nunique()} unique RIDs")

    # ── Step 1: Normalize CSF PTAU/ABETA (min-max → 0–1) ──────────────
    col = "PTAU/ABETA"
    csf_min = csf[col].min()
    csf_max = csf[col].max()
    csf[col] = (csf[col] - csf_min) / (csf_max - csf_min)

    print(f"\nNormalized CSF '{col}' (min-max)")
    print(f"  Original range: {csf_min:.6f} – {csf_max:.6f}")
    print(f"  New range     : {csf[col].min():.6f} – {csf[col].max():.6f}")

    # ── Step 2: Find RIDs in Plasma but not in CSF ─────────────────────
    csf_rids       = set(csf["RID"].unique())
    plasma_only_df = plasma[~plasma["RID"].isin(csf_rids)].copy()

    print(f"\nRIDs in Plasma but not in CSF: {plasma_only_df['RID'].nunique()}")
    print(f"Rows to append              : {len(plasma_only_df)}")

    # ── Step 3: Normalize plasma pT217_AB42_F separately (min-max → 0–1)
    p_col   = "pT217_AB42_F"
    p_min   = plasma_only_df[p_col].min()
    p_max   = plasma_only_df[p_col].max()
    plasma_only_df[p_col] = (plasma_only_df[p_col] - p_min) / (p_max - p_min)

    print(f"\nNormalized Plasma '{p_col}' (min-max)")
    print(f"  Original range: {p_min:.6f} – {p_max:.6f}")
    print(f"  New range     : {plasma_only_df[p_col].min():.6f} – {plasma_only_df[p_col].max():.6f}")

    # ── Step 4: Build rows, renaming plasma cols → CSF col names ───────
    rows_to_add = pd.DataFrame({
        "RID"        : plasma_only_df["RID"].values,
        "VISCODE"    : plasma_only_df["VISCODE2"].values,    # VISCODE2 → VISCODE
        "DRAWDTE"    : plasma_only_df["EXAMDATE"].values,    # EXAMDATE → DRAWDTE
        "ABETA"      : plasma_only_df["AB42_F"].values,      # AB42_F   → ABETA
        "PTAU"       : plasma_only_df["pT217_F"].values,     # pT217_F  → PTAU
        "PTAU/ABETA" : plasma_only_df["pT217_AB42_F"].values,# normalized above
    })

    # ── Step 5: Tag sources ────────────────────────────────────────────
    csf["Source"]         = "CSF"
    rows_to_add["Source"] = "Plasma"

    # ── Step 6: Drop TAU column from CSF before concat ─────────────────
    csf = csf.drop(columns=["TAU"])
    print("\nDropped 'TAU' column")

    # ── Step 7: Append ───────────────────────────────────────────────
    result = pd.concat([csf, rows_to_add], ignore_index=True)

    # ── Step 8: Keep only VISCODE = "bl" ─────────────────────────────
    before = len(result)
    result = result[result["VISCODE"] == "bl"].reset_index(drop=True)
    print(f"\nFiltered to VISCODE='bl': {before} → {len(result)} rows")

    # ── Step 9a: DX pass 1 — exact match on RID + date ───────────────
    result["_date"] = pd.to_datetime(result["DRAWDTE"], format="mixed", dayfirst=False)

    cog["_date"] = pd.to_datetime(cog["VISDATE"], format="mixed", dayfirst=False)
    cog_date_dedup = cog[["RID", "_date", "DX"]].drop_duplicates(subset=["RID", "_date"])

    result = result.merge(cog_date_dedup, on=["RID", "_date"], how="left")

    pass1 = result["DX"].notna().sum()
    print(f"\nDX pass 1 (exact RID+date): {pass1} matched")

    # ── Step 9b: DX pass 2 — RID + VISCODE fallback (< 30 day check)─
    # Cognitive rows where VISCODE or VISCODE2 = "bl", one per RID
    cog_bl = cog[(cog["VISCODE"] == "bl") | (cog["VISCODE2"] == "bl")].copy()
    cog_bl = cog_bl[["RID", "_date", "DX"]].drop_duplicates(subset=["RID"])
    cog_bl = cog_bl.rename(columns={"_date": "_cog_date", "DX": "_DX_viscode"})

    # Merge cognitive bl data onto rows still missing DX
    result = result.merge(cog_bl, on="RID", how="left")

    # Accept the VISCODE-matched DX only when date gap < 30 days
    day_diff = (result["_date"] - result["_cog_date"]).dt.days.abs()
    fill_mask = result["DX"].isna() & result["_DX_viscode"].notna() & (day_diff < 30)

    result.loc[fill_mask, "DX"] = result.loc[fill_mask, "_DX_viscode"]

    pass2 = fill_mask.sum()
    skipped = (result["DX"].isna() & result["_DX_viscode"].notna()).sum()
    print(f"DX pass 2 (RID+VISCODE, <30 days): {pass2} matched, "
          f"{skipped} skipped (≥30 days)")
    print(f"Total DX filled: {result['DX'].notna().sum()} / {len(result)}")

    # Clean up temp columns
    result = result.drop(columns=["_date", "_cog_date", "_DX_viscode"])

    # ── Step 10: Reorder columns — DX between PTAU/ABETA and Source ──
    result = result[["RID", "VISCODE", "DRAWDTE", "ABETA", "PTAU", "PTAU/ABETA",
                      "DX", "Source"]]

    # ── Step 11: Save ─────────────────────────────────────────────────
    result.to_csv(output_path, index=False)

    print(f"\nSaved → {output_path}")
    print(f"  Total rows   : {len(result)}")
    print(f"  Columns      : {list(result.columns)}")


if __name__ == "__main__":
    main()
