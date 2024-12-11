# STAT 486: Final Project

Author: Madison Wozniak

## Project Description

This project explores fod affordability inequalities among women-headed households in California using data-driven insights from machine learning models. The primary goal is to predict affordability ratios using features related to annual income, family demographics, and regional data. Additionally, the project includes anomaly detection to identify unique cases and SHAP analysis for feature interpretability.

**Key objectives:**

- Understand the relationship between affordability, income, and demographics.

- Build and evaluate machine learning models to predict affordability.

- Interpret model results using SHAP values to identify key features.

- Detect anomalies in the data to highlight outliers and unique subgroups.

- The final report, along with the supporting code is included in this repo and provides insights into affordabiility disparities and offers actionable recommendations for future work and policy interventions.

## Overview of Results

**Key Findings:**

1. XGBoost as the Best Model:

    XGBoost outperformed other models, achieving the lowest rMSE score and demonstrating strong generalization capabilities.
    Hyperparameter tuning improved performance further.

2. Feature Interpretability with SHAP:

    Key features influencing affordability include median_income and different geographic locations.

    SHAP visualizations provided insights into individual predictions and global feature importance.

3. Anomaly Detection:

    Identified income outliers and subgroups with unique affordability patterns, offering actionable insights into economic disparities.

4. Dimension Reduction:

    PCA revealed that most of the dataset's variance could be explained with a reduced set of components, aiding in model simplification and visualization.

## Navigating this Repo

        ├── code/                        # Python scripts for data processing, modeling, and analysis
        │   ├── EDA.ipynb                # Code for exploratory data analysis
        │   ├── working_income.ipynb     # feature engineering, model selection
        │   ├── selected_model.ipynb     # XGBoost, SHAP, Cluster analysis, dimension reduction 
        ├── plots/                       # Saved visualizations (e.g., SHAP, PCA, UMAP)
        ├── reports/                     # Reports from entire project process
        |   |── final_report.md          # Full report in Markdown format
        |   |──final_report.pdf          # Full report pdf
        |   |── Midway Report.docx              
        |   |──Project Proposal.docx 
        └── README.md                    # Project overview (this file)


Dataset not included for privacy purposes but can be found here: 

Link to data source: https://catalog.data.gov/dataset/food-affordability-fc448