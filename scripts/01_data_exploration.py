
# ============================================================
# PM2.5 FORECASTING USING BiLSTM
# PHASE 1 - DATA EXPLORATION
# Sections 1 to 24
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# Visualization settings
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

# ============================================================
# 0. PROJECT PATHS
# ============================================================

# Locate the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dataset path
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "combined_gee_pm25.csv"

# Directory for saving figures
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. LOAD THE DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

# Create a cleaned copy of the dataset
df_clean = df.copy()

# Convert date column to datetime
df_clean["date"] = pd.to_datetime(df_clean["date"])

# Replace negative PM2.5 values with NaN
df_clean.loc[df_clean["pm25"] < 0, "pm25"] = np.nan

# Display the first 5 rows
print("FIRST 5 ROWS:")
print(df.head())

# Display dataset dimensions
print("\nDATASET SHAPE:")
print(df.shape)

# Display column names
print("\nCOLUMN NAMES:")
print(df.columns.tolist())

# Display data types
print("\nDATA TYPES:")
print(df.dtypes)

# Display dataset information
print("\nDATASET INFORMATION:")
df.info()

# Display basic statistics
print("\nDESCRIPTIVE STATISTICS:")
print(df.describe())


# ============================================================
# 2. MISSING VALUES AND DUPLICATES
# ============================================================

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nMISSING VALUES PERCENTAGE:")
print((df.isnull().sum() / len(df)) * 100)

print("\nEXACT DUPLICATES:")
print(df.duplicated().sum())

print("\nDUPLICATED STATION-DATE COMBINATIONS:")
print(df.duplicated(subset=["date", "sensor_name"]).sum())

print("\nROWS WITH MISSING VALUES:")
print(df[df.isnull().any(axis=1)].head(10))


# ============================================================
# 3. EXAMINE THE REPEATED OBSERVATIONS
# ============================================================

duplicates = df[
    df.duplicated(
        subset=["date", "sensor_name"],
        keep=False
    )
].sort_values(["sensor_name", "date"])

print("\nDUPLICATED OBSERVATIONS:")
print(duplicates.to_string(index=False))


# ============================================================
# 4. DESCRIPTIVE STATISTICS
# ============================================================

numeric_cols = df.select_dtypes(include=np.number).columns

df_stats = df_clean.copy()

print("\nDESCRIPTIVE STATISTICS:")
print(df_stats[numeric_cols].describe().T)

print("\nSKEWNESS:")
print(df_stats[numeric_cols].skew())

print("\nKURTOSIS:")
print(df_stats[numeric_cols].kurtosis())


# ============================================================
# 5. DATA VISUALIZATION
# ============================================================

features = [
    "temperature_celsius",
    "pressure_mb",
    "wind_u",
    "wind_v",
    "NO2",
    "CO",
    "O3",
    "AOD",
    "pm25"
]

df_plot = df_clean.copy()

# 5.1 Histograms
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(features):
    sns.histplot(
        data=df_plot,
        x=col,
        bins=30,
        kde=True,
        ax=axes[i]
    )
    axes[i].set_title(f"Distribution of {col}")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_feature_histograms.png", dpi=300)
plt.show()

# 5.2 Boxplots
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(features):
    sns.boxplot(
        data=df_plot,
        x=col,
        ax=axes[i]
    )
    axes[i].set_title(f"Boxplot of {col}")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "02_feature_boxplots.png", dpi=300)
plt.show()


# ============================================================
# 6. CHECK INVALID PM2.5 VALUES
# ============================================================

print("\nNUMBER OF -999 VALUES:")
print((df["pm25"] == -999).sum())

print("\nNUMBER OF 1000 VALUES:")
print((df["pm25"] == 1000).sum())

print("\nPM2.5 MINIMUM IN df_plot:")
print(df_plot["pm25"].min())

print("\nPM2.5 MAXIMUM IN df_plot:")
print(df_plot["pm25"].max())


# ============================================================
# 7. CHECK NEGATIVE PM2.5 VALUES
# ============================================================

print("\nNUMBER OF NEGATIVE PM2.5 VALUES:")
print((df["pm25"] < 0).sum())

print("\nNEGATIVE PM2.5 OBSERVATIONS:")
print(
    df.loc[
        df["pm25"] < 0,
        ["date", "id", "sensor_name", "pm25"]
    ]
    .sort_values("pm25")
    .to_string(index=False)
)

print("\nHIGHEST PM2.5 VALUES:")
print(
    df.nlargest(10, "pm25")[
        ["date", "id", "sensor_name", "pm25"]
    ].to_string(index=False)
)


# ============================================================
# 8. ANALYZE VERY HIGH PM2.5 VALUES
# ============================================================

print("\nFREQUENCY OF NEGATIVE VALUES:")
print(
    df.loc[df["pm25"] < 0, "pm25"]
    .value_counts()
    .head(15)
)

print("\nNEGATIVE VALUES BY DATE:")
print(
    df.loc[df["pm25"] < 0]
    .groupby("date")
    .size()
    .sort_values(ascending=False)
    .head(15)
)

print("\nNEGATIVE VALUES BY STATION:")
print(
    df.loc[df["pm25"] < 0]
    .groupby("sensor_name")
    .size()
    .sort_values(ascending=False)
    .head(15)
)

print("\nVALUES GREATER THAN 300:")
print(
    df.loc[
        df["pm25"] > 300,
        ["date", "sensor_name", "pm25"]
    ]
    .sort_values("pm25", ascending=False)
    .to_string(index=False)
)


# ============================================================
# 9. CLEANING SUMMARY
# ============================================================

negative_count = (df["pm25"] < 0).sum()

print("\nDATA CLEANING SUMMARY")
print("Original rows:", len(df))
print("Negative PM2.5 values replaced:", negative_count)
print("Missing PM2.5 after cleaning:", df_clean["pm25"].isna().sum())
print("Minimum valid PM2.5:", df_clean["pm25"].min())
print("Maximum valid PM2.5:", df_clean["pm25"].max())

print("\nORIGINAL NUMBER OF ROWS:")
print(len(df))

print("\nNUMBER OF MISSING PM2.5 VALUES:")
print(df_clean["pm25"].isnull().sum())

print("\nMINIMUM PM2.5 VALUE AFTER CLEANING:")
print(df_clean["pm25"].min())

print("\nMAXIMUM PM2.5 VALUE AFTER CLEANING:")
print(df_clean["pm25"].max())


# ============================================================
# 10. CHECK DUPLICATED DATES IN CANDIDATE STATIONS
# ============================================================

candidate_stations = ["CENMO1", "CENCA1", "MOMPEAN"]

candidate_duplicates = df_clean[
    df_clean["sensor_name"].isin(candidate_stations)
    & df_clean.duplicated(
        subset=["sensor_name", "date"],
        keep=False
    )
].sort_values(["sensor_name", "date"])

print("\nDUPLICATES IN CANDIDATE STATIONS")

if candidate_duplicates.empty:
    print("No duplicated station-date observations found.")
else:
    print(
        candidate_duplicates[
            [
                "date", "id", "sensor_name",
                "latitude", "longitude", "pm25"
            ]
        ].to_string(index=False)
    )


# ============================================================
# 11. EXAMINE MISSING VALUES AFTER CLEANING
# ============================================================

print("\nMISSING VALUES AFTER CLEANING:")
print(df_clean.isnull().sum())

print("\nMISSING VALUES PERCENTAGE:")
print(
    (df_clean.isnull().sum() / len(df_clean) * 100).round(2)
)

print("\nTOTAL NUMBER OF MISSING VALUES:")
print(df_clean.isnull().sum().sum())


# ============================================================
# 12. EXAMINE MISSING VALUES OVER TIME
# ============================================================

missing_by_date = (
    df_clean.groupby("date")[["NO2", "CO", "O3", "pm25"]]
    .apply(lambda x: x.isnull().sum())
)

print("\nDATES WITH THE MOST MISSING VALUES:")
print(
    missing_by_date.sum(axis=1)
    .sort_values(ascending=False)
    .head(15)
)

print("\nMISSING VALUES BY MONTH:")
print(
    df_clean.set_index("date")[["NO2", "CO", "O3", "pm25"]]
    .isnull()
    .resample("ME")
    .sum()
    .sum(axis=1)
    .sort_values(ascending=False)
    .head(15)
)


# ============================================================
# 13. CHECK MISSING VALUE PATTERNS
# ============================================================

missing_patterns = (
    df_clean[["NO2", "CO", "O3"]]
    .isnull()
    .value_counts()
)

print("\nMISSING VALUE PATTERNS:")
print(missing_patterns)

all_three_missing = df_clean[
    df_clean[["NO2", "CO", "O3"]].isnull().all(axis=1)
]

print("\nROWS WHERE NO2, CO AND O3 ARE ALL MISSING:")
print(len(all_three_missing))

at_least_one_missing = df_clean[
    df_clean[["NO2", "CO", "O3"]].isnull().any(axis=1)
]

print("\nROWS WHERE AT LEAST ONE VARIABLE IS MISSING:")
print(len(at_least_one_missing))


# ============================================================
# 14. EXAMINE THE STATIONS AFFECTED BY MISSING VALUES
# ============================================================

missing_by_station = (
    df_clean.groupby("sensor_name")[["NO2", "CO", "O3"]]
    .apply(lambda x: x.isnull().sum())
)

missing_by_station["total_missing"] = missing_by_station.sum(axis=1)

missing_by_station = missing_by_station.sort_values(
    by="total_missing",
    ascending=False
)

print("\nSTATIONS WITH THE MOST MISSING VALUES:")
print(missing_by_station.head(15))

print("\nTOTAL NUMBER OF STATIONS:")
print(df_clean["sensor_name"].nunique())


# ============================================================
# 15. CALCULATE MISSING VALUE PERCENTAGE BY STATION
# ============================================================

station_counts = df_clean.groupby("sensor_name").size()

missing_station_analysis = (
    df_clean.groupby("sensor_name")[["NO2", "CO", "O3"]]
    .apply(lambda x: x.isnull().sum())
)

missing_station_analysis["total_observations"] = station_counts

missing_station_analysis["missing_percentage"] = (
    missing_station_analysis[["NO2", "CO", "O3"]].sum(axis=1)
    / (station_counts * 3) * 100
)

missing_station_analysis = missing_station_analysis.sort_values(
    "missing_percentage",
    ascending=False
)

print("\nMISSING DATA PERCENTAGE BY STATION:")
print(missing_station_analysis.head(15).round(2))


# ============================================================
# 16. ANALYZE MISSING PM2.5 VALUES BY STATION
# ============================================================

pm25_station_analysis = df_clean.groupby("sensor_name").agg(
    total_observations=("pm25", "size"),
    missing_pm25=("pm25", lambda x: x.isnull().sum())
)

pm25_station_analysis["missing_percentage"] = (
    pm25_station_analysis["missing_pm25"]
    / pm25_station_analysis["total_observations"] * 100
)

pm25_station_analysis = pm25_station_analysis.sort_values(
    "missing_percentage",
    ascending=False
)

print("\nPM2.5 MISSING VALUES BY STATION:")
print(pm25_station_analysis.head(15).round(2))

print("\nTOTAL STATIONS WITH MISSING PM2.5:")
print((pm25_station_analysis["missing_pm25"] > 0).sum())

print("\nTOTAL STATIONS WITHOUT MISSING PM2.5:")
print((pm25_station_analysis["missing_pm25"] == 0).sum())


# ============================================================
# 17. ANNUAL PM2.5 EVOLUTION
# ============================================================

annual_pm25 = (
    df_clean.groupby(df_clean["date"].dt.year)["pm25"]
    .mean()
)

print("\nAVERAGE PM2.5 BY YEAR:")
print(annual_pm25.round(2))

plt.figure(figsize=(12, 5))

sns.lineplot(
    x=annual_pm25.index,
    y=annual_pm25.values,
    marker="o"
)

plt.title("Annual Average PM2.5 Concentration")
plt.xlabel("Year")
plt.ylabel("Average PM2.5")
plt.xticks(annual_pm25.index, rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "03_annual_pm25_evolution.png", dpi=300)
plt.show()


# ============================================================
# 18. MONTHLY PM2.5 EVOLUTION
# ============================================================

monthly_pm25 = (
    df_clean.groupby(df_clean["date"].dt.month)["pm25"]
    .mean()
)

print("\nAVERAGE PM2.5 BY MONTH:")
print(monthly_pm25.round(2))

plt.figure(figsize=(10, 5))

sns.lineplot(
    x=monthly_pm25.index,
    y=monthly_pm25.values,
    marker="o"
)

plt.title("Average PM2.5 Concentration by Month")
plt.xlabel("Month")
plt.ylabel("Average PM2.5")
plt.xticks(range(1, 13))
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "04_monthly_pm25_evolution.png", dpi=300)
plt.show()


# ============================================================
# 19. CORRELATION MATRIX
# ============================================================

correlation_cols = [
    "temperature_celsius",
    "pressure_mb",
    "wind_u",
    "wind_v",
    "NO2",
    "CO",
    "O3",
    "AOD",
    "pm25"
]

correlation_matrix = df_clean[correlation_cols].corr()

print("\nCORRELATION MATRIX:")
print(correlation_matrix.round(2))

plt.figure(figsize=(11, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Matrix of Environmental Variables")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "05_correlation_matrix.png", dpi=300)
plt.show()


# ============================================================
# 20. COMPARE STATIONS
# ============================================================

station_pm25 = (
    df_clean.groupby("sensor_name")["pm25"]
    .mean()
    .sort_values(ascending=False)
)

print("\nTOP 15 STATIONS WITH HIGHEST AVERAGE PM2.5:")
print(station_pm25.head(15).round(2))

print("\nBOTTOM 15 STATIONS WITH LOWEST AVERAGE PM2.5:")
print(station_pm25.tail(15).round(2))

plt.figure(figsize=(12, 6))

sns.barplot(
    x=station_pm25.head(15).values,
    y=station_pm25.head(15).index
)

plt.title("Top 15 Stations by Average PM2.5")
plt.xlabel("Average PM2.5")
plt.ylabel("Station")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "06_top_15_stations_pm25.png", dpi=300)
plt.show()


# ============================================================
# 21. EXAMINE EXTREME STATIONS
# ============================================================

stations_to_check = [
    "Gibraltar",
    "064F",
    "mimiz",
    "testcompnex"
]

for station in stations_to_check:

    station_data = df_clean[
        df_clean["sensor_name"] == station
    ]["pm25"]

    print("\nStation:", station)
    print("Number of observations:", station_data.count())
    print("Missing values:", station_data.isnull().sum())
    print("Minimum:", station_data.min())
    print("Median:", station_data.median())
    print("Mean:", station_data.mean())
    print("Maximum:", station_data.max())


# ============================================================
# 22. EXAMINE THE NUMBER OF OBSERVATIONS PER STATION
# ============================================================

station_counts = (
    df_clean.groupby("sensor_name")["pm25"]
    .count()
    .sort_values(ascending=True)
)

print("\n15 STATIONS WITH THE FEWEST OBSERVATIONS:")
print(station_counts.head(15))

print("\n15 STATIONS WITH THE MOST OBSERVATIONS:")
print(station_counts.tail(15))

print("\nGENERAL STATISTICS:")
print(station_counts.describe())


# ============================================================
# 23. CHECK TEMPORAL COVERAGE
# ============================================================

station_periods = (
    df_clean.groupby("sensor_name")
    .agg(
        first_date=("date", "min"),
        last_date=("date", "max"),
        total_observations=("pm25", "count"),
        distinct_days=("date", "nunique")
    )
)

station_periods["period_days"] = (
    station_periods["last_date"] - station_periods["first_date"]
).dt.days + 1

station_periods = station_periods.sort_values(
    "distinct_days",
    ascending=True
)

print("\n15 STATIONS WITH THE SHORTEST TEMPORAL COVERAGE:")
print(station_periods.head(15))

print("\n15 STATIONS WITH THE LONGEST TEMPORAL COVERAGE:")
print(station_periods.tail(15))


# ============================================================
# 24. ANALYZE TEMPORAL GAPS BETWEEN OBSERVATIONS
# ============================================================

# Keep valid PM2.5 observations
df_time = df_clean.dropna(subset=["pm25"]).copy()

# Keep only one record per station and date for this analysis
df_time = df_time.drop_duplicates(
    subset=["sensor_name", "date"]
)

# Sort observations chronologically
df_time = df_time.sort_values(["sensor_name", "date"])

# Calculate the number of days between consecutive observations
df_time["gap_days"] = (
    df_time.groupby("sensor_name")["date"]
    .diff()
    .dt.days
)

# Calculate temporal gap statistics for each station
gap_stats = (
    df_time.groupby("sensor_name")["gap_days"]
    .agg(
        median_gap="median",
        max_gap="max",
        number_of_gaps="count",
        consecutive_days=lambda x: (x == 1).sum()
    )
)

gap_stats["consecutive_percentage"] = (
    gap_stats["consecutive_days"]
    / gap_stats["number_of_gaps"] * 100
)

print("\n15 STATIONS WITH THE HIGHEST CONSECUTIVE-DAY PERCENTAGE:")
print(
    gap_stats.sort_values(
        "consecutive_percentage",
        ascending=False
    ).head(15).round(2)
)

print("\n15 STATIONS WITH THE LARGEST TEMPORAL GAPS:")
print(
    gap_stats.sort_values(
        "max_gap",
        ascending=False
    ).head(15).round(2)
)


# ============================================================
# END OF PHASE 1 - GENERAL DATA EXPLORATION
# ============================================================

print("\n" + "=" * 60)
print("PHASE 1 - DATA EXPLORATION COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nFigures saved in:")
print(FIGURES_DIR)
