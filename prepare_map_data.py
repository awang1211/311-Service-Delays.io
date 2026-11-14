"""
Join per-complaint-type results to NTA boundaries so the web map can filter by complaint type.

Input:  output/nta_summary.geojson, output/nta_by_complaint_type.csv
Output: output/nta_by_type.geojson
Run:    python prepare_map_data.py   (after nyc311_delay_gap.py)
"""
import geopandas as gpd
import pandas as pd

nta = gpd.read_file("output/nta_summary.geojson")[["nta_code", "nta_name", "borough_name", "geometry"]]
bt = pd.read_csv("output/nta_by_complaint_type.csv")
g = nta.merge(bt, on="nta_code")
# drop small samples and Rodent (most Rodent requests close the same day, so wait time is not meaningful)
g = g[(g["n_requests"] >= 30) & (g["complaint_type"] != "Rodent")]
g.to_file("output/nta_by_type.geojson", driver="GeoJSON")
print(len(g), "rows saved")
