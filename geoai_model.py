"""
GeoAI step: machine learning benchmark for "expected wait", then spatial analysis of the gaps.

Idea:
  - A gradient boosting model learns how long a request normally takes citywide,
    using only non-location features (complaint type, agency, month, weekday, hour,
    how busy the city was that day). Location is deliberately excluded.
  - Gap = actual wait - model's expected wait, summarized per neighborhood (NTA).
  - Compare with the simple median benchmark: if the same neighborhoods stand out,
    the finding does not depend on the method.

Requires: output of nyc311_delay_gap.py (data/raw_311.csv, data/nta2020.geojson)
Run:      python geoai_model.py
Outputs:  output/ml_nta_comparison.csv, output/ml_summary.txt
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

import nyc311_delay_gap as base

CAT = ["complaint_type", "agency"]
NUM = ["month", "weekday", "hour", "city_daily_volume", "type_daily_volume"]
CASE = "East Harlem (North)"
lines = []


def say(msg=""):
    print(msg, flush=True)
    lines.append(msg)


# ---------------------------------------------------------------- 1. same data + cleaning as the main analysis
raw = base.download_311()
nta = base.download_nta()
df, df_all, _ = base.clean(raw)
df = base.add_gap(df)                       # adds the median benchmark: expected_days, gap_days

# ---------------------------------------------------------------- 2. features (no location)
df["weekday"] = df["created_date"].dt.weekday
df["hour"] = df["created_date"].dt.hour
day = df["created_date"].dt.date
df["city_daily_volume"] = day.map(day.value_counts())
df["type_daily_volume"] = df.groupby([day, "complaint_type"])["unique_key"].transform("size")

X = df[CAT + NUM].copy()
for c in CAT:
    X[c] = X[c].astype("category").cat.codes
y = np.log1p(df["resolution_days"] * 24)    # log hours: wait times are heavily right-skewed

model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                      categorical_features=[0, 1], random_state=42)

# ---------------------------------------------------------------- 3. accuracy on held-out data vs median benchmark
X_tr, X_te, y_tr, y_te, i_tr, i_te = train_test_split(X, y, df.index, test_size=0.2, random_state=42)
model.fit(X_tr, y_tr)
pred_te_days = np.expm1(model.predict(X_te)) / 24
actual_te = df.loc[i_te, "resolution_days"]

train = df.loc[i_tr]
med = train.groupby(["complaint_type", "agency", "month"])["resolution_days"].median()
keys = list(zip(df.loc[i_te, "complaint_type"], df.loc[i_te, "agency"], df.loc[i_te, "month"]))
base_te = pd.Series([med.get(k, np.nan) for k in keys], index=i_te).fillna(train["resolution_days"].median())

mae_model = np.mean(np.abs(actual_te - pred_te_days))
mae_base = np.mean(np.abs(actual_te - base_te))
medae_model = np.median(np.abs(actual_te - pred_te_days))
medae_base = np.median(np.abs(actual_te - base_te))
say("=== Model accuracy on held-out 20% of requests ===")
say(f"Median absolute error  - ML model: {medae_model:.2f} days | median benchmark: {medae_base:.2f} days")
say(f"Mean absolute error    - ML model: {mae_model:.2f} days | median benchmark: {mae_base:.2f} days")

# which features matter (explainability)
sample = np.random.RandomState(0).choice(len(X_te), size=min(20000, len(X_te)), replace=False)
imp = permutation_importance(model, X_te.iloc[sample], y_te.iloc[sample], n_repeats=3, random_state=0)
say("\nFeature importance (drop in accuracy when the feature is shuffled):")
for name, v in sorted(zip(CAT + NUM, imp.importances_mean), key=lambda t: -t[1]):
    say(f"  {name:20s} {v:.3f}")

# ---------------------------------------------------------------- 4. expected wait for every request (out-of-fold)
oof = cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=42))
df["expected_ml_days"] = np.expm1(oof) / 24
df["gap_ml_days"] = df["resolution_days"] - df["expected_ml_days"]
# center so the citywide median gap is 0, like the median benchmark (log-scale models under-predict skewed waits)
shift = df["gap_ml_days"].median()
df["expected_ml_days"] += shift
df["gap_ml_days"] -= shift

# ---------------------------------------------------------------- 5. per-neighborhood comparison
joined = base.to_nta(df, nta)
agg = joined.groupby("nta_code").agg(n_requests=("unique_key", "size"),
                                     gap_median_benchmark=("gap_days", "median"),
                                     gap_ml=("gap_ml_days", "median"),
                                     share_slower_ml=("gap_ml_days", lambda s: (s > 0).mean())).reset_index()
names = nta[["nta_code", "nta_name", "borough_name"]]
comp = names.merge(agg, on="nta_code", how="inner")
rel = comp[comp["n_requests"] >= base.MIN_REQUESTS_PER_NTA].copy()

rho, _ = spearmanr(rel["gap_median_benchmark"], rel["gap_ml"])
big = rel[rel["n_requests"] >= 500]
top_b = set(big.nlargest(10, "gap_median_benchmark")["nta_name"])
top_m = set(big.nlargest(10, "gap_ml")["nta_name"])
say("\n=== Do both methods find the same neighborhoods? ===")
say(f"Rank correlation of neighborhood gaps (Spearman): {rho:.2f}  (1 = identical ranking)")
say(f"Overlap of top-10 slowest neighborhoods (500+ requests): {len(top_b & top_m)} of 10")
rel["rank_ml"] = rel["gap_ml"].rank(ascending=False)
rel["rank_benchmark"] = rel["gap_median_benchmark"].rank(ascending=False)
if CASE in set(rel["nta_name"]):
    r = rel[rel["nta_name"] == CASE].iloc[0]
    say(f"{CASE}: gap {r['gap_ml']:+.2f} days (ML) vs {r['gap_median_benchmark']:+.2f} days (benchmark); "
        f"rank {int(r['rank_ml'])} vs {int(r['rank_benchmark'])} of {len(rel)}")

say("\nTop 10 slowest neighborhoods by ML gap (500+ requests):")
cols = ["nta_name", "borough_name", "n_requests", "gap_ml", "gap_median_benchmark"]
say(big.sort_values("gap_ml", ascending=False)[cols].head(10).round(2).to_string(index=False))

# ---------------------------------------------------------------- 6. hot spots on the ML gap (same Gi* + FDR)
gdf = nta[["nta_code", "geometry"]].merge(comp, on="nta_code", how="left")
gdf["reliable"] = gdf["n_requests"].fillna(0) >= base.MIN_REQUESTS_PER_NTA
gdf["median_gap_days"] = gdf["gap_ml"]      # add_hotspots reads this column
gdf = base.add_hotspots(gdf)
hs = gdf[gdf["hotspot"].str.contains("spot", na=False)]
say("\nSignificant clusters on the ML gap (Gi*, FDR-corrected):")
say(hs[["nta_name", "borough_name", "hotspot"]].to_string(index=False) if len(hs) else "  none")

rel = rel.merge(gdf[["nta_code", "hotspot"]].rename(columns={"hotspot": "hotspot_ml"}), on="nta_code", how="left")
rel.sort_values("gap_ml", ascending=False).to_csv("output/ml_nta_comparison.csv", index=False)
with open("output/ml_summary.txt", "w") as f:
    f.write("\n".join(lines))
print("\nSaved output/ml_nta_comparison.csv and output/ml_summary.txt")
