"""
Build the final interactive web map (single HTML file, no account needed).

Inputs  (from nyc311_delay_gap.py + the nta_by_type step):
  output/nta_summary.geojson
  output/nta_by_type.geojson
Output:
  output/lat_long_wait_map.html

Run:  python make_web_map.py
"""

import json
import geopandas as gpd

SUMMARY = "output/nta_summary.geojson"
BY_TYPE = "output/nta_by_type.geojson"
OUT = "output/lat_long_wait_map.html"
MIN_N = 30          # neighborhoods with fewer requests are shown grey
CASE_STUDY = "East Harlem (North)"

# ---------------------------------------------------------------- 1. load + merge
nta = gpd.read_file(SUMMARY).to_crs(epsg=4326)
by_type = gpd.read_file(BY_TYPE).drop(columns="geometry")

def type_cols(name, prefix):
    t = by_type[by_type["complaint_type"] == name][["nta_code", "n_requests", "median_gap_days",
                                                      "median_actual_days"]]
    return t.rename(columns={"n_requests": f"{prefix}_n", "median_gap_days": f"{prefix}_gap",
                             "median_actual_days": f"{prefix}_actual"})

nta = nta.merge(type_cols("HEAT/HOT WATER", "heat"), on="nta_code", how="left")
nta = nta.merge(type_cols("PLUMBING", "plumb"), on="nta_code", how="left")

# ---------------------------------------------------------------- 2. keep only what the map needs
keep = ["nta_code", "nta_name", "borough_name", "n_requests", "median_actual_days",
        "median_gap_days", "share_waited_longer", "no_access_share", "top10_building_share",
        "hotspot", "heat_n", "heat_gap", "plumb_n", "plumb_gap", "geometry"]
keep = [c for c in keep if c in nta.columns]
nta = nta[keep].copy()
nta["geometry"] = nta.geometry.simplify(0.0001, preserve_topology=True)   # smaller file
for c in nta.columns:
    if nta[c].dtype.kind == "f":
        nta[c] = nta[c].round(3)
if "hotspot" not in nta.columns:
    nta["hotspot"] = "n/a"
nta["hotspot"] = nta["hotspot"].fillna("n/a")
geojson = nta.to_json()
with open("output/map_data.geojson", "w") as f:   # also used by build_dashboard.py
    f.write(geojson)

# ---------------------------------------------------------------- 3. HTML template (Leaflet)
html = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lat/Long Wait - NYC 311 Service Gaps</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body, #map { height: 100%; margin: 0; font-family: -apple-system, Segoe UI, Roboto, sans-serif; }
  .panel { position: absolute; top: 12px; left: 12px; z-index: 1000; background: white; padding: 14px 16px;
           border-radius: 8px; box-shadow: 0 1px 6px rgba(0,0,0,.25); width: 300px; font-size: 13px; }
  .panel h1 { font-size: 17px; margin: 0 0 4px; }
  .panel p { margin: 0 0 10px; color: #555; line-height: 1.35; }
  .panel label { display: block; font-weight: 600; margin: 8px 0 3px; }
  .panel select { width: 100%; padding: 4px; font-size: 13px; }
  .legend { position: absolute; bottom: 24px; left: 12px; z-index: 1000; background: white; padding: 10px 12px;
            border-radius: 8px; box-shadow: 0 1px 6px rgba(0,0,0,.25); font-size: 12px; width: 300px; }
  .bar { height: 12px; border-radius: 3px; margin: 6px 0 2px; }
  .ticks { display: flex; justify-content: space-between; color: #444; }
  .note { color: #777; margin-top: 6px; }
  .leaflet-popup-content table td { padding: 1px 6px 1px 0; }
</style></head>
<body>
<div id="map"></div>
<div class="panel">
  <h1>Lat/Long Wait</h1>
  <p>Where do NYC 311 requests wait longer than similar requests citywide?
     Each request is compared with the citywide median for the same complaint type, agency and month (2024).</p>
  <label for="metric">Show</label>
  <select id="metric">
    <option value="median_gap_days">Extra wait vs similar requests - all types</option>
    <option value="heat_gap">Extra wait vs similar requests - Heat/Hot Water</option>
    <option value="plumb_gap">Extra wait vs similar requests - Plumbing</option>
    <option value="no_access_share">% closed as "no access"</option>
    <option value="median_actual_days">Median actual wait (days)</option>
  </select>
  <label for="borough">Borough</label>
  <select id="borough">
    <option value="All">All boroughs</option>
    <option>Manhattan</option><option>Bronx</option><option>Brooklyn</option>
    <option>Queens</option><option>Staten Island</option>
  </select>
  <label for="hot">Statistical clusters</label>
  <select id="hot">
    <option value="on">Outline significant hot spots (FDR-corrected)</option>
    <option value="off">Hide outlines</option>
  </select>
</div>
<div class="legend" id="legend"></div>
<script>
const DATA = __GEOJSON__;
const MIN_N = __MIN_N__;
const CASE = "__CASE__";

const map = L.map('map', {zoomControl: false}).setView([40.72, -73.94], 11);
L.control.zoom({position: 'topright'}).addTo(map);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom: 18, attribution: '&copy; OpenStreetMap contributors | Data: NYC Open Data 311, NYC DCP NTAs'}).addTo(map);

const METRICS = {
  median_gap_days: {type: 'div', lo: -0.6, hi: 0.6, unit: 'days', n: 'n_requests',
                    label: 'Extra wait vs similar requests (days, median)'},
  heat_gap:        {type: 'div', lo: -0.6, hi: 0.6, unit: 'days', n: 'heat_n',
                    label: 'Heat/Hot Water: extra wait vs similar requests (days)'},
  plumb_gap:       {type: 'div', lo: -3, hi: 3, unit: 'days', n: 'plumb_n',
                    label: 'Plumbing: extra wait vs similar requests (days)'},
  no_access_share: {type: 'seq', lo: 0.1, hi: 0.5, unit: '%', n: 'n_requests',
                    label: 'Share of requests closed as "no access"'},
  median_actual_days: {type: 'seq', lo: 1, hi: 3, unit: 'days', n: 'n_requests',
                    label: 'Median actual wait (days)'}
};

function lerp(a, b, t) { return Math.round(a + (b - a) * t); }
function mix(c1, c2, t) { return `rgb(${lerp(c1[0],c2[0],t)},${lerp(c1[1],c2[1],t)},${lerp(c1[2],c2[2],t)})`; }
const BLUE = [33,102,172], WHITE = [247,247,247], RED = [178,24,43], PALE = [254,240,217], DARK = [179,0,0];

function color(v, m) {
  if (v === null || v === undefined || isNaN(v)) return '#cccccc';
  const t = Math.max(0, Math.min(1, (v - m.lo) / (m.hi - m.lo)));
  if (m.type === 'div') return t < 0.5 ? mix(BLUE, WHITE, t * 2) : mix(WHITE, RED, (t - 0.5) * 2);
  return mix(PALE, DARK, t);
}
function fmt(v, unit) {
  if (v === null || v === undefined || isNaN(v)) return 'n/a';
  if (unit === '%') return (v * 100).toFixed(1) + '%';
  return (v > 0 ? '+' : '') + Number(v).toFixed(2);
}

let layer;
function draw() {
  const key = document.getElementById('metric').value;
  const boro = document.getElementById('borough').value;
  const hot = document.getElementById('hot').value === 'on';
  const m = METRICS[key];
  if (layer) map.removeLayer(layer);
  layer = L.geoJSON(DATA, {
    filter: f => boro === 'All' || f.properties.borough_name === boro,
    style: f => {
      const p = f.properties, ok = (p[m.n] || 0) >= MIN_N;
      const isHot = hot && p.hotspot === 'Hot spot (longer waits)';
      const isCase = p.nta_name === CASE;
      return {fillColor: ok ? color(p[key], m) : '#d9d9d9', fillOpacity: 0.8,
              color: isHot ? '#000' : (isCase ? '#6a00a8' : '#888'),
              weight: isHot ? 3 : (isCase ? 3 : 0.5), dashArray: isCase && !isHot ? '6 4' : null};
    },
    onEachFeature: (f, l) => {
      const p = f.properties;
      l.bindPopup(`<b>${p.nta_name}</b> (${p.borough_name})<table>
        <tr><td>Requests (deduplicated)</td><td>${p.n_requests ?? 'n/a'}</td></tr>
        <tr><td>Median actual wait</td><td>${p.median_actual_days ?? 'n/a'} days</td></tr>
        <tr><td>Extra wait vs similar, all</td><td>${fmt(p.median_gap_days, 'days')} days</td></tr>
        <tr><td>Extra wait, Heat/Hot Water</td><td>${fmt(p.heat_gap, 'days')} days</td></tr>
        <tr><td>Extra wait, Plumbing</td><td>${fmt(p.plumb_gap, 'days')} days</td></tr>
        <tr><td>% slower than citywide norm</td><td>${fmt(p.share_waited_longer, '%')}</td></tr>
        <tr><td>% closed as "no access"</td><td>${fmt(p.no_access_share, '%')}</td></tr>
        <tr><td>Cluster test (Gi*, FDR)</td><td>${p.hotspot}</td></tr></table>`);
      l.on('mouseover', () => l.setStyle({weight: 3, color: '#333'}));
      l.on('mouseout', () => layer.resetStyle(l));
    }
  }).addTo(map);
  if (boro !== 'All' && layer.getLayers().length) map.fitBounds(layer.getBounds());
  if (boro === 'All') map.setView([40.72, -73.94], 11);
  legend(m);
}

function legend(m) {
  const bar = m.type === 'div'
    ? `linear-gradient(to right, ${mix(BLUE,BLUE,0)}, ${mix(WHITE,WHITE,0)}, ${mix(RED,RED,0)})`
    : `linear-gradient(to right, ${mix(PALE,PALE,0)}, ${mix(DARK,DARK,0)})`;
  const lo = m.unit === '%' ? Math.round(m.lo*100)+'%' : m.lo, hi = m.unit === '%' ? Math.round(m.hi*100)+'%+' : m.hi + '+';
  const mid = m.type === 'div' ? '<span>0 = citywide norm</span>' : '';
  document.getElementById('legend').innerHTML = `<b>${m.label}</b>
    <div class="bar" style="background:${bar}"></div>
    <div class="ticks"><span>${m.type === 'div' ? 'faster ' + lo : lo}</span>${mid}<span>${m.type === 'div' ? 'slower ' + hi : hi}</span></div>
    <div class="note">Grey = fewer than ${MIN_N} requests. Black outline = significant hot spot.
    Dashed purple = case study (${CASE}). Not causal: patterns overlap with, but are not explained by, demographics.</div>`;
}

['metric', 'borough', 'hot'].forEach(id => document.getElementById(id).addEventListener('change', draw));
draw();
</script></body></html>"""

html = (html.replace("__GEOJSON__", geojson)
            .replace("__MIN_N__", str(MIN_N))
            .replace("__CASE__", CASE_STUDY))
with open(OUT, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Saved {OUT} ({len(html) / 1e6:.1f} MB)")
