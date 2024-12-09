# STAT 486: Final Project

Author: Madison Wozniak

## Project Description

This project explores income inequality among women-headed households in California using data-driven insights from machine learning models. The primary goal is to predict median income using features related to food affordability, family demographics, and regional data. Additionally, the project includes anomaly detection to identify unique cases and SHAP analysis for feature interpretability.

**Key objectives:**

- Understand the relationship between income, affordability, and demographics.

- Build and evaluate machine learning models to predict median income.

- Interpret model results using SHAP values to identify key features.

- Detect anomalies in the data to highlight outliers and unique subgroups.

- The final report, along with the supporting code is included in this repo and provides insights into income disparities and offers actionable recommendations for future work and policy interventions.

## Overview of Results

**Key Findings:**

1. XGBoost as the Best Model:

    XGBoost outperformed other models, achieving the lowest rMSE score and demonstrating strong generalization capabilities.
    Hyperparameter tuning improved performance further.

2. Feature Interpretability with SHAP:

    Key features influencing income include affordability_per_person and affordability_ratio.
    SHAP visualizations provided insights into individual predictions and global feature importance.

3. Anomaly Detection:

    Identified income outliers and subgroups with unique income patterns, offering actionable insights into economic disparities.

4. Dimension Reduction:

    PCA revealed that most of the dataset's variance could be explained with a reduced set of components, aiding in model simplification and visualization.

Link to data source: https://catalog.data.gov/dataset/food-affordability-fc448