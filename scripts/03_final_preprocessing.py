
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\pc\OneDrive\Documents\PM25-BiLSTM-Forecasting"
)

DATA_PATH = PROJECT_DIR / "data" / "raw" / "combined_gee_pm25.csv"

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "CENCA1_pm25_preprocessed.csv"

# Check that the raw dataset exists
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

print("Raw dataset found:", DATA_PATH)

# ============================================================
# LOAD AND CLEAN THE DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

df_clean = df.copy()

df_clean["date"] = pd.to_datetime(df_clean["date"])

# Replace negative PM2.5 values with NaN
df_clean.loc[df_clean["pm25"] < 0, "pm25"] = np.nan

# ============================================================
# 50. PREPARE THE SELECTED STATION
# ============================================================

selected_station = "CENCA1"

features = [
    "temperature_celsius",
    "pressure_mb",
    "wind_u",
    "wind_v",
    "NO2",
    "CO",
    "O3",
    "AOD"
]

target = "pm25"

selected_columns = ["date", target] + features

df_selected = df_clean[
    df_clean["sensor_name"] == selected_station
][selected_columns].copy()

df_selected = df_selected.sort_values("date")

print("\n" + "=" * 60)
print("SELECTED STATION:", selected_station)
print("=" * 60)

print("\nDATASET SHAPE:")
print(df_selected.shape)

print("\nFIRST 5 ROWS:")
print(df_selected.head())

print("\nMISSING VALUES:")
print(df_selected.isna().sum())

print("\nDATE RANGE:")
print(df_selected["date"].min(), "to", df_selected["date"].max())

print("\nDUPLICATED DATES:")
print(df_selected["date"].duplicated().sum())

# ============================================================
# 51. CONSECUTIVE-DAY ANALYSIS
# ============================================================

dates = df_selected["date"].sort_values().reset_index(drop=True)

# Calculate the number of days between observations
gaps = dates.diff().dt.days.dropna()

# Identify continuous segments of consecutive days
segment_id = dates.diff().dt.days.ne(1).cumsum()
segment_lengths = dates.groupby(segment_id).size()

# Calculate possible 3-day input sequences
sequence_length = 3

possible_sequences = sum(
    max(0, n - sequence_length)
    for n in segment_lengths
)

print("\n" + "=" * 60)
print("CONSECUTIVE-DAY ANALYSIS – CENCA1")
print("=" * 60)

print("\nGAP STATISTICS:")
print("Median gap:", gaps.median(), "days")
print("Maximum gap:", gaps.max(), "days")
print(
    "Consecutive-day percentage:",
    round((gaps == 1).mean() * 100, 2),
    "%"
)

print("\nCONTINUOUS SEGMENTS:")
print("Number of segments:", len(segment_lengths))
print("Longest segment:", segment_lengths.max(), "days")
print(
    "Average segment length:",
    round(segment_lengths.mean(), 2),
    "days"
)

print("\nSEQUENCE ANALYSIS:")
print("Input sequence length:", sequence_length, "days")
print(
    "Possible 3-day input sequences with next-day target:",
    possible_sequences
)

# ============================================================
# 52. FINAL DATASET PREPARATION AND EXPORT
# ============================================================

# Sort observations chronologically
df_final = df_selected.sort_values("date").reset_index(drop=True)

# Verify data quality
print("\n" + "=" * 60)
print("FINAL DATASET CHECK")
print("=" * 60)

print("\nDataset shape:")
print(df_final.shape)

print("\nMissing values:")
print(df_final.isna().sum())

print("\nDuplicated dates:")
print(df_final["date"].duplicated().sum())

print("\nNegative PM2.5 values:")
print((df_final["pm25"] < 0).sum())

print("\nDate order:")
print(df_final["date"].is_monotonic_increasing)

# Save the preprocessed dataset
df_final.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nDataset successfully saved!")
print("Saved location:", OUTPUT_PATH)

# ============================================================
# 53. VERIFY THE EXPORTED DATASET
# ============================================================

# Reload the saved CSV file
df_check = pd.read_csv(OUTPUT_PATH)

# Convert date column back to datetime
df_check["date"] = pd.to_datetime(df_check["date"])

print("\n" + "=" * 60)
print("EXPORTED DATASET VERIFICATION")
print("=" * 60)

print("\nDataset shape:")
print(df_check.shape)

print("\nColumn names:")
print(df_check.columns.tolist())

print("\nFirst 5 rows:")
print(df_check.head())

print("\nMissing values:")
print(df_check.isna().sum().sum())

print("\nDate range:")
print(df_check["date"].min(), "to", df_check["date"].max())

print("\nDataset verification completed!")

# ============================================================
# FINAL COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("FINAL PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nFinal dataset:", OUTPUT_PATH)
print("Final shape:", df_check.shape)
