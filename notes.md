## Data source
- Source: Zindi Urban Heat Island Challenge
- Downloaded: 2026-01-28
- Files received:
  - Data.zip (raw competition data, stored locally)
  - Sample notebooks (reference only)
  - SampleSubmission.csv
- Note: Raw data is excluded from version control.

## Conceptual understanding
- CSV provides point locations and UHI class labels
- Features are derived from spatial datasets (satellite imagery, building footprints)
- For each location, map-based values (e.g. vegetation, urban density) are extracted
- These values serve as input features for a classification model

## Project direction
- This project follows a competition-style workflow.
- Data sources, targets, and feature definitions are treated as given.
- Focus is on machine learning evaluation and baseline modeling.

## Baseline progression

### Building-based features
- Started with building count within fixed radius (100m).
- Extended to multi-scale radii (50m / 100m / 200m), which significantly improved results.
- Added total building footprint area per radius as an additional proxy for urban density.
- Feature scaling (StandardScaler) was required for stable convergence.

Result:
- Accuracy improved from ~0.37 (single radius) to ~0.51 (multi-scale + area).

### Road-based features (OSM)
- Added OpenStreetMap road length per radius (50m / 100m / 200m) as proxy for surface sealing and urban intensity.
- Roads were fetched once via OSMnx and stored locally (GeoPackage).
- Road length is currently approximated by summing full segment lengths intersecting each buffer (MVP, no clipping).

Result:
- Accuracy improved further to ~0.53.
- Macro F1-score increased, indicating more balanced class performance.

### Notes
- Medium UHI class remains difficult to separate, likely representing a transition zone.
- Further improvements likely require non-building features (e.g. vegetation, surface properties).
