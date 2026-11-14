"""
Build the one-page dashboard (index.html) from the map data and the HTML template.

Input:  map_data.geojson (written by make_web_map.py to output/; a copy is kept in the repo root)
        dashboard_template.html
Output: index.html  (published with GitHub Pages)
Run:    python build_dashboard.py [path/to/map_data.geojson]
"""
import sys

src = sys.argv[1] if len(sys.argv) > 1 else "map_data.geojson"
with open(src) as f:
    geojson = f.read()
with open("dashboard_template.html") as f:
    html = f.read()
with open("index.html", "w") as f:
    f.write(html.replace("__GEOJSON__", geojson))
print(f"Saved index.html ({len(geojson) / 1e6:.1f} MB of map data)")
