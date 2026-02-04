# Urban Heat Island Detection for Climate-Aware Urban Planning

## Context
This project is based on the Zindi "Urban Heat Island Challenge".
The goal is to identify urban heat island patterns using geospatial and environmental data.

The project is not optimized for leaderboard ranking.
The focus is on methodological clarity, interpretability, and relevance for urban climate adaptation.

## Problem Statement
Urban heat islands increase health risks, energy consumption, and reduce quality of life.
Identifying heat-prone areas can support targeted urban planning measures such as
tree planting, shading, and surface de-sealing.

## Data
The dataset is provided by the competition organizers and includes
geospatial and environmental features related to urban areas.
Raw data is stored unchanged in `data/raw/`.

## Approach
- Exploratory data analysis to understand spatial and feature patterns
- Feature preprocessing and basic feature engineering
- Baseline supervised classification model
- Evaluation with appropriate metrics
- Visualization of predicted heat island areas

## Results
This project prioritizes explainable results over maximum accuracy.
Visualizations are used to relate predictions to urban structures.

## Limitations
- No hyperparameter optimization for competition ranking
- Limited domain-specific calibration
- Results are indicative, not operational

## Possible Extensions
- Integration of vegetation and water proximity features
- Temporal analysis with additional climate data
- Comparison with urban cooling measures

## Motivation
This project is part of a personal portfolio focused on
machine learning applications for climate adaptation in cities.

## Baseline results (Brazil, building features)

A logistic regression baseline using multi-scale building density (50m / 100m / 200m) 
and building footprint area achieves:

- Accuracy: ~0.51
- Macro F1-score: ~0.45

Low-UHI areas are identified reliably, while Medium-UHI remains challenging, 
indicating that additional environmental features (e.g. vegetation or surface properties) 
are required.

## Data
Raw training data is not included in this repository.

Expected structure:
data/
├─ raw/
│  ├─ Sample_Brazil_uhi_data.csv
│  └─ Brazil Building Footprints/
│     └─ *.shp