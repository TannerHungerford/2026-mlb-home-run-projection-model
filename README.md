# MLB 2026 Home Run Projection Model

This project builds a machine learning model to project the **2026 MLB home run leaderboard** using historical performance data and Statcast power metrics.

The model pulls historical batting data from **FanGraphs** and advanced contact-quality metrics from **Statcast** using the `pybaseball` Python library. It then trains an ensemble of machine learning models to estimate each player's future home run rate.

Key features of the projection system include:

• Historical performance metrics (HR/PA, fly ball rate, HR/FB)  
• Statcast power indicators (exit velocity, barrel rate, launch angle, hard-hit rate)  
• Expected statistics (xSLG, xwOBA)  
• Automatic ballpark adjustments using Statcast park factors  
• Aging curve adjustments to account for player development and decline  
• Playing-time projections based on recent plate appearance trends  
• Ensemble machine learning models (Random Forest, Gradient Boosting, Extra Trees, Ridge Regression)

The model predicts a player's **home run rate per plate appearance**, adjusts it for park effects and aging, and multiplies by projected plate appearances to estimate total home runs.

Final projection formula:

Projected HR = HR_rate_final × Projected_PA

Where:

HR_rate_final = (Model Prediction × Aging Adjustment × Park Factor)

The output dataset ranks players by projected home run totals for the 2026 season.

Output file:

`projected_2026_hr_dataset.csv`
