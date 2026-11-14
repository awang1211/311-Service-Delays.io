"""
NYC 311 "Left Waiting" pipeline - GeoAI Hackathon (Team Green)
=================================================================
What this script does (baseline version):
  1. Downloads 311 requests for chosen complaint types + date range (NYC Open Data API)
  2. Downloads 2020 NTA (Neighborhood Tabulation Area) boundaries
  3. Cleans data and computes resolution time (days)
  4. Computes EXPECTED delay = citywide median for the same
     complaint type x agency x month  (location NOT used)
  5. gap = actual - expected, per request
  6. Spatial-joins requests to NTAs and summarizes per neighborhood
  7. (Optional) Getis-Ord Gi* hotspot test, if esda/libpysal are installed
  8. Writes CSV + GeoJSON (for QGIS / Kepler.gl) + an interactive HTML map (if folium installed)

Install:   pip install pandas geopandas requests folium esda libpysal
Run:       python nyc311_delay_gap.py
Re-run without re-downloading: data is cached in ./data/
"""

import io
import os
import sys
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import requests

# ----------------------------- SETTINGS -------------------------------------
# Complaint types: check exact spelling on NYC Open Data if a type returns 0 rows
COMPLAINT_TYPES = ["HEAT/HOT WATER", "PLUMBING", "Rodent"]
START_DATE = "2024-01-01"
END_DATE = "2024-12-31"

MIN_REQUESTS_PER_NTA = 30      # NTAs with fewer requests are flagged as unreliable
MAX_DAYS = 365                 # resolution times above this are treated as data errors
EXCLUDE_NO_ACCESS = True       # drop closures where the agency could not get in
FOCUS_NTAS = ["East Harlem (North)", "Flatbush"]  # case-study neighborhoods (exact NTA names)
DEDUP = True                   # drop repeat requests: same location + complaint type + same day
FDR_ALPHA = 0.05               # false discovery rate for hotspot significance

DATA_DIR = "data"
OUT_DIR = "output"

URL_311 = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"
NTA_URLS = [
    "https://data.cityofnewyork.us/api/v3/views/9nt8-h7nd/query.geojson?accessType=DOWNLOAD",
    "https://data.cityofnewyork.us/resource/9nt8-h7nd.geojson?$limit=5000",
]
PAGE_SIZE = 50000
# -----------------------------------------------------------------------------

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- 1. download
def download_311():
    cache = os.path.join(DATA_DIR, "raw_311.csv")
    if os.path.exists(cache):
        log(f"Using cached 311 data: {cache}")
        return pd.read_csv(cache, low_memory=False)

    types = ",".join(f"'{t}'" for t in COMPLAINT_TYPES)
    where = (
        f"complaint_type in({types}) "
        f"AND created_date between '{START_DATE}T00:00:00' and '{END_DATE}T23:59:59'"
    )
    cols = ("unique_key,created_date,closed_date,agency,complaint_type,descriptor,"
            "status,resolution_description,borough,incident_zip,latitude,longitude")

    frames, offset = [], 0
    while True:
        params = {"$select": cols, "$where": where, "$order": "unique_key",
                  "$limit": PAGE_SIZE, "$offset": offset}
        for attempt in range(3):
            try:
                r = requests.get(URL_311, params=params, timeout=300)
                r.raise_for_status()
                break
            except Exception as e:
                log(f"  retry {attempt + 1}/3 after error: {e}")
                time.sleep(5)
        else:
            sys.exit("Download failed 3 times. Check internet / try a smaller date range.")
        chunk = pd.read_csv(io.StringIO(r.text), low_memory=False)
        frames.append(chunk)
        log(f"  downloaded {offset + len(chunk):,} rows")
        if len(chunk) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(cache, index=False)
    log(f"Saved {len(df):,} rows -> {cache}")
    return df


def download_nta():
    cache = os.path.join(DATA_DIR, "nta2020.geojson")
    if not os.path.exists(cache):
        for url in NTA_URLS:
            try:
                r = requests.get(url, timeout=120)
                r.raise_for_status()
                with open(cache, "wb") as f:
                    f.write(r.content)
                log(f"Downloaded NTA boundaries -> {cache}")
                break
            except Exception as e:
                log(f"  NTA download failed from {url}: {e}")
        else:
            sys.exit("Could not download NTAs. Download '2020 Neighborhood Tabulation Areas' "
                     f"as GeoJSON from NYC Open Data and save it as {cache}")
    nta = gpd.read_file(cache)
    nta.columns = [c.lower() for c in nta.columns]
    # normalise field names (they differ slightly between download sources)
    rename = {}
    for c in nta.columns:
        if c in ("nta2020", "ntacode"):
            rename[c] = "nta_code"
        elif c == "ntaname":
            rename[c] = "nta_name"
        elif c == "boroname":
            rename[c] = "borough_name"
    nta = nta.rename(columns=rename)
    keep = [c for c in ["nta_code", "nta_name", "borough_name"] if c in nta.columns]
    return nta[keep + ["geometry"]].to_crs(epsg=4326)


# ---------------------------------------------------------------- 2. clean
def clean(df):
    n0 = len(df)
    report = {"raw rows": n0}

    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["closed_date"] = pd.to_datetime(df["closed_date"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    df = df.dropna(subset=["created_date", "closed_date"])
    report["after dropping open / no closed date"] = len(df)

    df = df.dropna(subset=["latitude", "longitude"])
    report["after dropping missing location"] = len(df)

    df["resolution_days"] = (df["closed_date"] - df["created_date"]).dt.total_seconds() / 86400
    df = df[(df["resolution_days"] >= 0) & (df["resolution_days"] <= MAX_DAYS)]
    report[f"after dropping negative or >{MAX_DAYS} days"] = len(df)

    # building proxy: 311 points are geocoded to the address, so identical coords ~ same building
    df["location_id"] = df["latitude"].round(5).astype(str) + "," + df["longitude"].round(5).astype(str)
    if DEDUP:
        df["day"] = df["created_date"].dt.date
        n_before = len(df)
        df = df.sort_values("created_date").drop_duplicates(
            subset=["location_id", "complaint_type", "day"], keep="first")
        report["duplicates removed (same building + type + day)"] = n_before - len(df)
        report["after removing duplicates"] = len(df)
    desc = df["resolution_description"].fillna("").str.lower()
    df["no_access"] = desc.str.contains("not able to gain access|could not gain access|"
                                        "unable to gain access|no access", regex=True)
    report["requests flagged 'no access'"] = int(df["no_access"].sum())
    df["month"] = df["created_date"].dt.month
    df_all = df.copy()                      # keeps no-access rows (for the no-access layer)
    if EXCLUDE_NO_ACCESS:
        df = df[~df["no_access"]]
        report["after excluding 'no access' (delay analysis)"] = len(df)
    report = pd.Series(report, name="rows")
    return df.copy(), df_all, report


# ---------------------------------------------------------------- 3. expected delay & gap
def add_gap(df):
    # Expected delay: citywide median for same complaint type x agency x month.
    # Location is deliberately NOT used, so neighborhood effects show up in the gap.
    keys = ["complaint_type", "agency", "month"]
    df["expected_days"] = df.groupby(keys)["resolution_days"].transform("median")
    df["gap_days"] = df["resolution_days"] - df["expected_days"]
    return df


# ---------------------------------------------------------------- 4. spatial join + summary
def to_nta(df, nta):
    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
                           crs="EPSG:4326")
    return gpd.sjoin(pts, nta, how="inner", predicate="within")


def summarize(df, df_all, nta):
    joined = to_nta(df, nta)
    log(f"{len(joined):,} requests matched to an NTA")
    joined_all = to_nta(df_all, nta)

    def agg(g):
        return pd.Series({
            "n_requests": len(g),
            "median_actual_days": g["resolution_days"].median(),
            "median_expected_days": g["expected_days"].median(),
            "mean_gap_days": g["gap_days"].mean(),
            "median_gap_days": g["gap_days"].median(),
            "share_waited_longer": (g["gap_days"] > 0).mean(),
        })

    by_nta = joined.groupby("nta_code").apply(agg).reset_index()
    by_nta_type = (joined.groupby(["nta_code", "complaint_type"]).apply(agg).reset_index())

    # no-access share: % of ALL closed requests in the NTA closed as "no access"
    na = joined_all.groupby("nta_code").agg(n_all=("no_access", "size"),
                                            no_access_share=("no_access", "mean")).reset_index()
    # concentration: share of NTA requests coming from its 10 busiest buildings
    def top10(g):
        c = g["location_id"].value_counts()
        return pd.Series({"top10_building_share": c.head(10).sum() / c.sum(),
                          "n_buildings": len(c)})
    conc = joined.groupby("nta_code").apply(top10).reset_index()

    out = (nta.merge(by_nta, on="nta_code", how="left")
              .merge(na, on="nta_code", how="left")
              .merge(conc, on="nta_code", how="left"))
    out["reliable"] = out["n_requests"].fillna(0) >= MIN_REQUESTS_PER_NTA
    return joined, joined_all, out, by_nta_type


# ---------------------------------------------------------------- 5. optional hotspots
def add_hotspots(gdf):
    try:
        from libpysal.weights import Queen
        from esda.getisord import G_Local
    except ImportError:
        log("esda/libpysal not installed -> skipping hotspot analysis (optional)")
        return gdf
    sub = gdf[gdf["reliable"]].copy().reset_index(drop=True)
    w = Queen.from_dataframe(sub, use_index=False, silence_warnings=True)
    w.transform = "r"
    g = G_Local(sub["median_gap_days"].values, w, star=True, permutations=999, seed=42)
    sub["gi_z"] = g.Zs
    sub["gi_p"] = g.p_sim
    # Benjamini-Hochberg FDR correction across all NTAs tested
    p = sub["gi_p"].values
    order = np.argsort(p)
    m = len(p)
    passed = p[order] <= FDR_ALPHA * (np.arange(1, m + 1) / m)
    cutoff = p[order][passed].max() if passed.any() else -1
    sub["sig_fdr"] = sub["gi_p"] <= cutoff
    log(f"Hotspots significant at p<0.05: {(sub['gi_p'] < 0.05).sum()}, after FDR: {sub['sig_fdr'].sum()}")
    sub["hotspot"] = np.select(
        [(sub["gi_z"] > 0) & sub["sig_fdr"], (sub["gi_z"] < 0) & sub["sig_fdr"]],
        ["Hot spot (longer waits)", "Cold spot (shorter waits)"], default="Not significant")
    gdf = gdf.merge(sub[["nta_code", "gi_z", "gi_p", "sig_fdr", "hotspot"]], on="nta_code", how="left")
    gdf["hotspot"] = gdf["hotspot"].fillna("Too few requests")
    log("Hotspot analysis done: " + gdf["hotspot"].value_counts().to_dict().__str__())
    return gdf


# ---------------------------------------------------------------- 6. map
def make_map(gdf):
    try:
        import folium
    except ImportError:
        log("folium not installed -> skipping HTML map (open the GeoJSON in QGIS/Kepler instead)")
        return
    m = folium.Map(location=[40.70, -73.94], zoom_start=10, tiles="OpenStreetMap")
    data = gdf[gdf["reliable"]].copy()
    for col in ["median_actual_days", "median_gap_days"]:
        data[col] = data[col].round(2)
    data["share_waited_longer"] = (data["share_waited_longer"] * 100).round(0)
    data["no_access_pct"] = (data["no_access_share"] * 100).round(1)
    data["top10_pct"] = (data["top10_building_share"] * 100).round(0)
    tooltip_fields = ["nta_name", "n_requests", "median_actual_days", "median_gap_days",
                      "share_waited_longer", "no_access_pct", "top10_pct"]
    aliases = ["Neighborhood", "Requests", "Median wait (days)", "Typical extra wait vs expected (days)",
               "% waited longer than expected", "% closed as 'no access'",
               "% of requests from 10 busiest buildings"]
    if "hotspot" in data.columns:
        tooltip_fields.append("hotspot")
        aliases.append("Hotspot")
    geo = data.drop(columns=[c for c in data.columns if data[c].dtype == "datetime64[ns]"])
    for col, name, unit, show in [("median_gap_days", "Extra wait vs expected (gap)", "days", True),
                                  ("no_access_pct", "Closed as 'no access'", "%", False),
                                  ("median_actual_days", "Actual median wait", "days", False)]:
        layer = folium.Choropleth(geo_data=geo.to_json(), data=geo, columns=["nta_code", col],
                                  key_on="feature.properties.nta_code", fill_color="YlOrRd",
                                  fill_opacity=0.75, line_opacity=0.3, name=name,
                                  legend_name=f"{name} ({unit})", show=show)
        layer.add_to(m)
        folium.GeoJsonTooltip(fields=tooltip_fields, aliases=aliases).add_to(layer.geojson)
    if "hotspot" in geo.columns:
        colors = {"Hot spot (longer waits)": "#b2182b", "Cold spot (shorter waits)": "#2166ac"}
        hs = geo[geo["hotspot"].isin(colors)]
        if len(hs):
            folium.GeoJson(hs.to_json(), name="Hotspots (FDR-corrected)", show=False,
                           style_function=lambda f: {"fillColor": colors[f["properties"]["hotspot"]],
                                                     "color": "black", "weight": 1.5, "fillOpacity": 0.6},
                           tooltip=folium.GeoJsonTooltip(fields=["nta_name", "hotspot"],
                                                         aliases=["Neighborhood", "Hotspot"])).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    path = os.path.join(OUT_DIR, "map_311_delay_gap.html")
    m.save(path)
    log(f"Interactive map -> {path}")


# ---------------------------------------------------------------- main
def main():
    raw = download_311()
    log(f"Complaint types found: {raw['complaint_type'].value_counts().to_dict()}")
    nta = download_nta()

    df, df_all, report = clean(raw)
    log("Cleaning report:\n" + report.to_string())
    df = add_gap(df)

    joined, joined_all, gdf, by_type = summarize(df, df_all, nta)
    gdf = add_hotspots(gdf)

    # outputs
    report.to_csv(os.path.join(OUT_DIR, "cleaning_report.csv"))
    gdf.drop(columns="geometry").to_csv(os.path.join(OUT_DIR, "nta_summary.csv"), index=False)
    gdf.to_file(os.path.join(OUT_DIR, "nta_summary.geojson"), driver="GeoJSON")
    by_type.to_csv(os.path.join(OUT_DIR, "nta_by_complaint_type.csv"), index=False)

    for name in FOCUS_NTAS:
        f = joined[joined["nta_name"] == name]
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        f.drop(columns="geometry").to_csv(os.path.join(OUT_DIR, f"case_{slug}_requests.csv"), index=False)
        bld = (f.groupby("location_id").agg(n_requests=("unique_key", "size"),
                                            median_gap_days=("gap_days", "median"),
                                            types=("complaint_type", lambda x: ", ".join(sorted(set(x)))))
                .sort_values("n_requests", ascending=False))
        bld.head(25).to_csv(os.path.join(OUT_DIR, f"case_{slug}_top_buildings.csv"))

    make_map(gdf)

    # quick findings in the console
    rel = gdf[gdf["reliable"]].sort_values("median_gap_days", ascending=False)
    cols = ["nta_name", "borough_name", "n_requests", "median_actual_days", "median_gap_days", "share_waited_longer"]
    cols = [c for c in cols if c in rel.columns]
    print("\nTop 10 neighborhoods waiting LONGER than expected:")
    print(rel[cols].head(10).round(2).to_string(index=False))
    print("\nTop 10 neighborhoods waiting SHORTER than expected:")
    print(rel[cols].tail(10).round(2).to_string(index=False))
    print("\nCase-study check:")
    for name in FOCUS_NTAS:
        row = gdf[gdf["nta_name"] == name]
        f = joined[joined["nta_name"] == name]
        if row.empty or f.empty:
            print(f"  {name}: not found - check exact NTA name in nta_summary.csv")
            continue
        r = row.iloc[0]
        f_all = joined_all[joined_all["nta_name"] == name]
        f_wo_top = f[~f["location_id"].isin(f["location_id"].value_counts().head(10).index)]
        print(f"\n  {name}: {int(r['n_requests'])} requests from {int(r['n_buildings'])} buildings")
        print(f"    median gap: {r['median_gap_days']:.2f} days | hotspot: {r.get('hotspot', 'n/a')}")
        print(f"    10 busiest buildings = {r['top10_building_share']*100:.0f}% of requests")
        print(f"    median gap WITHOUT those 10 buildings: {f_wo_top['gap_days'].median():.2f} days")
        print(f"    closed as 'no access': {f_all['no_access'].mean()*100:.1f}% (city: {joined_all['no_access'].mean()*100:.1f}%)")
        print("    by complaint type (median days):")
        print(f.groupby("complaint_type")[["resolution_days", "expected_days", "gap_days"]]
              .median().round(2).to_string().replace("\n", "\n      "))
    print("\nTop 10 neighborhoods by 'no access' share:")
    na_cols = [c for c in ["nta_name", "borough_name", "n_all", "no_access_share"] if c in rel.columns]
    print(rel.sort_values("no_access_share", ascending=False)[na_cols].head(10).round(3).to_string(index=False))
    log(f"Done. Outputs in ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
