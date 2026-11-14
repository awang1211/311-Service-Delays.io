# Lat/Long Wait: Mapping Where NYC 311 Service Falls Behind

**NYC GeoAI Hackathon 2026 - GeoAI for Social Good track**
Team: Anruo Wang, LuClandereine Leger

**Live site:** https://lleger0.github.io/lleger0.311-Service-Delays.io/

## The problem

New Yorkers rely on 311 to report problems that affect safety and quality of life, such as no heat, plumbing failures and rodents. Raw average wait times are misleading: heat complaints need an inspection and are slower by nature, so a neighborhood that files many of them will look slow even if it is treated fairly.

We ask: **which neighborhoods wait longer than similar requests citywide, and where do those gaps cluster?**

**Intended users:** 311 and agency managers (HPD, DOHMH), City Council members and community boards.

## Data

| Dataset | Source |
| --- | --- |
| 311 Service Requests (2024; Heat/Hot Water, Plumbing, Rodent) | [NYC Open Data](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9) |
| 2020 Neighborhood Tabulation Areas (NTAs) | [NYC Open Data](https://data.cityofnewyork.us/d/9nt8-h7nd) |
| American Community Survey 2020-2024 5-year (income, race/ethnicity, renter share), census tracts | [U.S. Census API](https://www.census.gov/data/developers/data-sets/acs-5year.html) |
| 2020 Census Tract to NTA equivalency | [NYC Open Data](https://data.cityofnewyork.us/d/hm78-6dwm) |

Preparation (see `output/cleaning_report.csv` after running):

| Step | Rows |
| --- | --- |
| Raw requests | 370,380 |
| After dropping open requests, missing locations and invalid times | 366,586 |
| Duplicates removed (same building + complaint type + same day) | 106,750 |
| Unique requests | 259,836 |
| Closed as "no access" (excluded from wait-time analysis, mapped separately) | 62,911 (24.2%) |

## Method

1. **Actual wait:** closed date minus created date, per request.
2. **Expected wait:** citywide median wait for the same complaint type, agency and month. Location is deliberately excluded, so the benchmark is a citywide norm.
3. **Gap:** actual minus expected, summarized per NTA (median). Positive = slower than similar requests citywide.
4. **Clusters:** Getis-Ord Gi* hot spot analysis (Queen contiguity, 9,999 permutations) with Benjamini-Hochberg false discovery rate (FDR) correction across all NTAs tested.
5. **Robustness checks:** deduplication, minimum 30 requests per NTA, and a concentration check (share of requests from the 10 busiest buildings, and the gap with those buildings removed).

All GIS operations (spatial join, spatial weights, hot spot analysis, choropleth mapping) are done in Python with GeoPandas and PySAL, so the analysis is fully reproducible.

## Key findings (2024)

- **29% of requests are duplicates:** the same building reports the same problem several times on the same day.
- **About 1 in 4 unique requests (24.2%) is closed as "no access".** Closed does not mean resolved. The share is highest in lower Manhattan (Chinatown-Two Bridges: 51%).
- **Upper Manhattan is the only significant slow cluster** after FDR correction (Harlem South and Harlem North).
- **East Harlem (North)** has the largest extra wait among high-volume neighborhoods: +0.59 days (median), with 65% of requests slower than the citywide norm. The gap holds (+0.61 days) after removing its 10 busiest buildings. Plumbing requests wait +2.35 days longer than similar requests citywide.
- Neighborhoods that wait less than similar requests are concentrated in the Bronx.
- Renter-heavy neighborhoods tend to wait longer (Spearman rho +0.47 with the ML benchmark, +0.13 with the simple benchmark). "No access" closures are more common where a larger share of residents is Black (+0.21). Links with income and other groups depend on the method. All are descriptive, not causal.

## Limitations

- **Not causal.** Overlap between slow areas and demographics does not mean demographics cause delays (ecological fallacy).
- **Closed is not fixed.** Closure dates measure when a request was closed, not when the problem was resolved.
- **Rodent requests** mostly close the same day, so their wait time is not meaningful; the map shows Heat/Hot Water and Plumbing.
- **Building identity** is approximated by identical coordinates (311 locations are geocoded to the address).
- The benchmark does not know building age, landlord or other building characteristics.

## How to run

```bash
pip install -r requirements.txt
python nyc311_delay_gap.py     # download, clean, compute gaps, hot spots -> output/
python prepare_map_data.py     # per-complaint-type layer -> output/nta_by_type.geojson
python make_web_map.py         # interactive map -> output/lat_long_wait_map.html
python geoai_model.py          # ML benchmark + comparison -> output/ml_summary.txt
CENSUS_KEY=... python census_compare.py   # ACS demographics comparison -> output/acs_summary.txt (free key: api.census.gov/data/key_signup.html)
```

## Next steps

- Add construction and building-safety complaints (Department of Buildings).
- Add building characteristics (age, size, owner) to the benchmark.
- Refresh automatically each month so agencies can track change over time.
