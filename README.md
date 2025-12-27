<div align="center">

# NYC Taxi Trip Duration Prediction
### Ensemble Learning for Urban Mobility


<p align="center">
  <strong>Predicting taxi trip duration using ensemble methods on 2.8M+ NYC taxi trips</strong>
</p>

[View Notebook](NYC_Taxi_Final_Project.ipynb) • [Results](#model-comparison) • [Methodology](#methodology)



## Project Overview

This project applies **ensemble learning methods** to predict NYC Yellow Taxi trip durations. Using January 2024 data enriched with weather information, we compare multiple machine learning approaches and analyze their generalization capabilities through learning curves.

### Objectives

- Predict trip duration accurately for better ETA estimation
- Compare ensemble methods (Bagging, Boosting, Voting)
- Analyze overfitting behavior using learning curves
- Integrate external weather data for improved predictions

---

## Dataset

### Primary Source
**NYC Yellow Taxi Trip Records** - January 2024
- **Source:** [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- **Raw size:** ~3 million trips
- **After cleaning:** 2,859,135 trips

### Weather Integration
**Meteostat API** - NYC Central Park station
- Temperature (avg, min, max)
- Precipitation
- Wind speed
- Atmospheric pressure

### Features Used

| Feature | Description |
|---------|-------------|
| `trip_distance` | Distance in miles |
| `passenger_count` | Number of passengers |
| `pickup_hour` | Hour of pickup (0-23) |
| `pickup_dayofweek` | Day of week (0-6) |
| `is_weekend` | Weekend indicator |
| `PULocationID` | Pickup zone ID |
| `DOLocationID` | Dropoff zone ID |
| `tavg` | Average temperature (°C) |
| `prcp` | Precipitation (mm) |
| `wspd` | Wind speed (km/h) |

---

## Methodology

### 1. Data Preprocessing
- Outlier removal (trips < 1 min or > 180 min)
- Missing value handling
- Feature engineering (temporal features, speed calculation)

### 2. Exploratory Data Analysis
- Distribution analysis of target variable
- Correlation heatmap
- Borough and hourly patterns

### 3. Model Training with GridSearchCV
All models optimized using 3-fold cross-validation on a sample of 100k observations.

**Models implemented:**
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor
- Bagging Regressor
- Voting Regressor

### 4. Dimensionality Reduction
PCA comparison to evaluate if reduced dimensions maintain predictive power.

### 5. Overfitting Analysis
Learning curves on Random Forest and XGBoost to verify generalization.

---

## Model Comparison

### Evaluation Metrics

to complete

### Key Findings

to complete
---

## XGBoost: Algorithm Choice Justification

XGBoost was selected as an advanced algorithm outside the course scope. It extends gradient boosting with:

- **Regularization:** L1 and L2 penalties to prevent overfitting
- **Sparsity-aware splits:** Efficient handling of missing values
- **Cache optimization:** Better computational performance

**Reference:**  
Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.  
https://doi.org/10.1145/2939672.2939785

---

## Project Structure
```
Machine-learning-project/
├── NYC_Taxi_Final_Project.ipynb    # Main notebook
├── README.md                        # Documentation
├── data/
│   └── (auto-downloaded from NYC TLC)
└── figures/
    └── (generated visualizations)
```

---



[![GitHub](https://img.shields.io/badge/GitHub-HugoNcy-181717?style=flat-square&logo=github)](https://github.com/HugoNcy)

---

<div align="center">

*ESILV - Machine Learning Project 2024-2025*

</div>
