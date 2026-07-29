import numpy as np
import pandas as pd

# aggregate all sales in 2024
months_2024 = [f"{m:02d}" for m in range(1, 13)]
files_2024 = [f"raw-data/CRMLSSold2024{m}.csv" for m in months_2024]
sold_2024 = pd.concat(pd.read_csv(f) for f in files_2024)

# aggregate all sales in 2025
months_2025 = [f"{m:02d}" for m in range(1, 13)]
files_2025 = [f"raw-data/CRMLSSold2025{m}.csv" for m in months_2025]
sold_2025 = pd.concat(pd.read_csv(f) for f in files_2025)

# aggregate all sales available in 2026 (till June 2026)
months_2026 = [f"{m:02d}" for m in range(1, 7)]
files_2026 = [f"raw-data/CRMLSSold2026{m}.csv" for m in months_2026]
sold_2026 = pd.concat(pd.read_csv(f) for f in files_2026)

# aggregate sales across all years into a single dataset
sold = pd.concat([sold_2024, sold_2025, sold_2026])
sold.shape # (663761, 84)
sold.to_csv('raw-data/sold_raw.csv', index=False)