# California Single-Family Home Valuation
This project develops automated valuation models (AVMs) for single-family homes in California, with the goal of generalizing to both on-market and off-market properties.

The models are trained on monthly CRMLS sales records and evaluated using a chronological split, with January 2025–April 2026 used for training, May 2026 for validation, and June 2026 for testing. Given the right-skewed distribution of sales prices, model performance is evaluated primarily using Median Absolute Percentage Error (MdAPE), with Mean Absolute Percentage Error (MAPE), R<sup>2</sup>, and other metrics provided for reference.
