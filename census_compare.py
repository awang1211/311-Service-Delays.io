"""
Compare neighborhood 311 gaps with Census ACS demographics (descriptive, not causal).

Steps:
  1. Download ACS 5-year tract data for NYC (Census API, no key needed for this volume)
  2. Map census tracts to 2020 NTAs (NYC DCP equivalency file on NYC Open Data)
  3. Aggregate to NTA: % renter, % Black, % Hispanic, % White, % Asian, median household income
  4. Correlate with the gaps and "no access" share; compare lowest vs highest income quarter

Requires: output/nta_summary.csv (from nyc311_delay_gap.py)
Optional: output/ml_nta_comparison.csv (from geoai_model.py)
Run:      python census_compare.py
Outputs:  output/acs_nta.csv, output/acs_summary.txt
"""

import io
import os
import re

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr

COUNTIES = "005,047,061,081,085"        # Bronx, Brooklyn, Manhattan, Queens, Staten Island
BORO_TO_COUNTY = {"1": "061", "2": "005", "3": "047", "4": "081", "5": "085"}
VARS = {
    "B19013_001E": "median_income",
    "B03002_001E": "pop",
    "B03002_003E": "white_nh",
    "B03002_004E": "black_nh",
    "B03002_006E": "asian_nh",
    "B03002_012E": "hispanic",
    "B25003_001E": "households",
    "B25003_003E": "renter_hh",
}
EQUIV_URLS = [
    "https://data.cityofnewyork.us/resource/hm78-6dwm.json?$limit=5000",   # tract -> NTA equivalency
    "https://data.cityofnewyork.us/resource/63ge-mke6.json?$limit=5000",   # 2020 census tracts (has NTA)
]
MIN_N = 30
lines = []


def say(msg=""):
    print(msg, flush=True)
    lines.append(str(msg))


# ---------------------------------------------------------------- 1. ACS tract data
def get_acs():
    key = os.environ.get("CENSUS_KEY", "")
    for year in (2024, 2023, 2022):
        url = f"https://api.census.gov/data/{year}/acs/acs5"
        frames = []
        try:
            for county in COUNTIES.split(","):          # one small request per county
                params = [("get", ",".join(VARS)), ("for", "tract:*"), ("in", "state:36"), ("in", f"county:{county}")]
                if key:
                    params.append(("key", key))
                r = requests.get(url, params=params, timeout=120)
                if r.status_code != 200 or not r.text.strip().startswith("["):
                    raise ValueError(f"HTTP {r.status_code}: {r.text.strip()[:300]}")
                rows = r.json()
                frames.append(pd.DataFrame(rows[1:], columns=rows[0]))
            df = pd.concat(frames, ignore_index=True).rename(columns=VARS)
            say(f"ACS {year} 5-year: {len(df):,} tracts")
            df["geoid"] = df["state"] + df["county"] + df["tract"]
            for c in VARS.values():
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[df["median_income"] < 0, "median_income"] = np.nan    # Census uses negative codes for missing
            return df, year
        except Exception as e:
            print(f"  ACS {year} failed: {e}")
    raise SystemExit("Could not download ACS data from api.census.gov (see the server message above)")


# ---------------------------------------------------------------- 2. tract -> NTA lookup
def find_col(df, pattern):
    for c in df.columns:
        vals = df[c].dropna().astype(str).head(200)
        if len(vals) and vals.str.fullmatch(pattern).mean() > 0.9:
            return c
    return None


def get_equiv():
    for url in EQUIV_URLS:
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            eq = pd.DataFrame(r.json())
        except Exception as e:
            print(f"  equivalency download failed ({url}): {e}")
            continue
        nta_col = find_col(eq, r"[A-Z]{2}\d{4}")                  # e.g. MN1101
        geoid_col = find_col(eq, r"36\d{9}")                       # 11-digit tract GEOID
        boroct_col = find_col(eq, r"[1-5]\d{6}")                   # borough code + 6-digit tract
        print(f"  {url.split('/')[-1][:9]} columns -> NTA: {nta_col}, GEOID: {geoid_col}, BoroCT: {boroct_col}")
        if nta_col is None or (geoid_col is None and boroct_col is None):
            continue
        out = pd.DataFrame({"nta_code": eq[nta_col].astype(str)})
        if geoid_col:
            out["geoid"] = eq[geoid_col].astype(str)
        else:
            b = eq[boroct_col].astype(str)
            out["geoid"] = "36" + b.str[0].map(BORO_TO_COUNTY) + b.str[1:]
        return out.drop_duplicates("geoid")
    raise SystemExit("Could not build the tract-to-NTA lookup. Print the equivalency columns and adjust find_col.")


acs, year = get_acs()
eq = get_equiv()
tr = acs.merge(eq, on="geoid", how="inner")
say(f"Tracts matched to an NTA: {len(tr):,} of {len(acs):,}")

# ---------------------------------------------------------------- 3. aggregate to NTA
tr["inc_x_hh"] = tr["median_income"] * tr["households"]
tr["hh_with_inc"] = np.where(tr["median_income"].notna(), tr["households"], 0)
g = tr.groupby("nta_code").agg(pop=("pop", "sum"), white_nh=("white_nh", "sum"), black_nh=("black_nh", "sum"),
                               asian_nh=("asian_nh", "sum"), hispanic=("hispanic", "sum"),
                               households=("households", "sum"), renter_hh=("renter_hh", "sum"),
                               inc_x_hh=("inc_x_hh", "sum"), hh_with_inc=("hh_with_inc", "sum")).reset_index()
g["pct_black"] = g["black_nh"] / g["pop"]
g["pct_hispanic"] = g["hispanic"] / g["pop"]
g["pct_white"] = g["white_nh"] / g["pop"]
g["pct_asian"] = g["asian_nh"] / g["pop"]
g["pct_renter"] = g["renter_hh"] / g["households"]
g["income"] = g["inc_x_hh"] / g["hh_with_inc"].replace(0, np.nan)   # household-weighted mean of tract medians

s = pd.read_csv("output/nta_summary.csv")
d = s.merge(g, on="nta_code", how="inner")
if os.path.exists("output/ml_nta_comparison.csv"):
    ml = pd.read_csv("output/ml_nta_comparison.csv")[["nta_code", "gap_ml"]]
    d = d.merge(ml, on="nta_code", how="left")
d = d[(d["n_requests"] >= MIN_N) & (d["households"] >= 500)].copy()
say(f"Neighborhoods compared: {len(d)}")

# ---------------------------------------------------------------- 4. correlations
outcomes = [("median_gap_days", "Extra wait (benchmark)"), ("gap_ml", "Extra wait (ML)"),
            ("no_access_share", "Share closed as 'no access'")]
outcomes = [o for o in outcomes if o[0] in d.columns]
demo = [("income", "Median household income"), ("pct_renter", "% renter households"),
        ("pct_black", "% Black"), ("pct_hispanic", "% Hispanic"), ("pct_white", "% White"),
        ("pct_asian", "% Asian")]

say("\n=== Rank correlation (Spearman rho) across neighborhoods ===")
say("Positive = higher values of the demographic go with longer extra waits / more 'no access'.")
say("Stars: * p<0.05, ** p<0.01 (descriptive only, not causal)")
table = []
for dcol, dname in demo:
    row = {"demographic": dname}
    for ocol, oname in outcomes:
        m = d[[dcol, ocol]].dropna()
        rho, p = spearmanr(m[dcol], m[ocol])
        star = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        row[oname] = f"{rho:+.2f}{star}"
    table.append(row)
say(pd.DataFrame(table).to_string(index=False))

say("\n=== Lowest vs highest income quarter of neighborhoods ===")
d["income_q"] = pd.qcut(d["income"], 4, labels=["Lowest", "Lower-middle", "Upper-middle", "Highest"])
cols = [c for c, _ in outcomes]
qt = d.groupby("income_q", observed=True)[cols + ["income"]].median()
qt["income"] = qt["income"].round(-3)
say(qt.round(3).to_string())

d.to_csv("output/acs_nta.csv", index=False)
with open("output/acs_summary.txt", "w") as f:
    f.write(f"ACS {year} 5-year estimates\n" + "\n".join(lines))
print("\nSaved output/acs_nta.csv and output/acs_summary.txt")
