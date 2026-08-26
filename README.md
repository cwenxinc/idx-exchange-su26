# Single-Family Residence Sales Price Prediction
This project develops automated valuation models for single-family homes in California, with the goal of generalizing to both on-market and off-market properties. 

The models are trained on monthly sales data from the CRMLS and evaluated using a chronological split scheme, with the most recent months reserved for validation and testing. Given the skewed distribution of sales prices, model performance is evaluated using multiple metrics, with median absolute percentage error (MdAPE) as the primary metric. The model with the lowest MdAPE is selected for deployment.

## Directory Structure
```
appendix
├── merge.py                       - Merges historical monthly sales
└── utilities.py                   - Helper functions for applying transformations learned from the training data
01_exploration.ipynb               - Explores variable distributions to inform preprocessing
02_data_cleaning.ipynb.            - Cleans the sales data
03_data_transformation.ipynb       - Engineers features, performs chronological splits, and applies training-based transformations
04_modeling.ipynb                  - Trains prediction models
README.md
```
