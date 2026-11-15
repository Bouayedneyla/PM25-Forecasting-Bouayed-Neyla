
# ============================================================
# PM2.5 FORECASTING USING BiLSTM
# PHASE 1 - STATION ANALYSIS
# Sections 25 to 49
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
# 0. PROJECT PATHS AND DATA LOADING
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "combined_gee_pm25.csv"

FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load the original dataset
df = pd.read_csv(DATA_PATH)

# Create a cleaned copy
df_clean = df.copy()

# Convert date column to datetime
df_clean["date"] = pd.to_datetime(df_clean["date"])

# Replace negative PM2.5 values with NaN
df_clean.loc[df_clean["pm25"] < 0, "pm25"] = np.nan

# Keep valid PM2.5 observations
df_time = df_clean.dropna(subset=["pm25"]).copy()

# Keep only one record per station and date for gap analysis
df_time = df_time.drop_duplicates(
    subset=["sensor_name", "date"]
)

# Sort observations chronologically
df_time = df_time.sort_values(["sensor_name", "date"])

# Calculate temporal gaps
df_time["gap_days"] = (
    df_time.groupby("sensor_name")["date"]
    .diff()
    .dt.days
)

# ============================================================
# 25. DISPLAY THE LARGEST TEMPORAL GAPS
# ============================================================

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

print("\nLARGEST TEMPORAL GAPS BY STATION:")

print(
    gap_stats.sort_values(
        "max_gap",
        ascending=False
    )[
        [
            "max_gap",
            "median_gap",
            "number_of_gaps",
            "consecutive_percentage"
        ]
    ].head(15).to_string()
)


# ============================================================
# 26. EXAMINE TEMPORAL GAPS IN CENMO1
# ============================================================

station_name = "CENMO1"

station_data = (
    df_time[df_time["sensor_name"] == station_name]
    .sort_values("date")
    .copy()
)

station_data["gap_days"] = (
    station_data["date"].diff().dt.days
)

print("\nLARGEST GAPS FOR", station_name)

print(
    station_data.nlargest(10, "gap_days")[
        ["date", "gap_days", "pm25"]
    ].to_string(index=False)
)

print("\nGAP STATISTICS:")
print(station_data["gap_days"].describe())


# ============================================================
# 27. CENMO1 YEARLY ANALYSIS
# ============================================================

cenmo1 = df_clean[
    df_clean["sensor_name"] == "CENMO1"
].copy()

cenmo1["year"] = cenmo1["date"].dt.year

yearly_counts = (
    cenmo1.groupby("year")["pm25"]
    .agg(["count", "mean", "median"])
)

print("\nCENMO1 - YEARLY ANALYSIS:")
print(yearly_counts.round(2))

ax = yearly_counts["count"].plot(
    kind="bar",
    figsize=(10, 5),
    title="Number of Valid PM2.5 Observations per Year - CENMO1"
)

ax.set_xlabel("Year")
ax.set_ylabel("Number of observations")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "07_cenmo1_yearly_observations.png", dpi=300)
plt.show()


# ============================================================
# 28. CENCA1 YEARLY ANALYSIS
# ============================================================

cenca1 = df_clean[
    df_clean["sensor_name"] == "CENCA1"
].copy()

cenca1["year"] = cenca1["date"].dt.year

yearly_counts_cenca1 = (
    cenca1.groupby("year")["pm25"]
    .agg(["count", "mean", "median"])
)

print("\nCENCA1 - YEARLY ANALYSIS:")
print(yearly_counts_cenca1.round(2))

ax = yearly_counts_cenca1["count"].plot(
    kind="bar",
    figsize=(10, 5),
    title="Number of Valid PM2.5 Observations per Year - CENCA1"
)

ax.set_xlabel("Year")
ax.set_ylabel("Number of observations")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "08_cenca1_yearly_observations.png", dpi=300)
plt.show()


# ============================================================
# 29. EXAMINE EXTREME CENCA1 VALUES IN 2024
# ============================================================

cenca1_2024 = cenca1[
    (cenca1["year"] == 2024)
    & (cenca1["pm25"].notna())
].copy()

print("\nCENCA1 - HIGHEST PM2.5 VALUES IN 2024:")

print(
    cenca1_2024.nlargest(15, "pm25")[
        ["date", "pm25"]
    ].to_string(index=False)
)

print("\nCENCA1 - DESCRIPTIVE STATISTICS FOR 2024:")
print(cenca1_2024["pm25"].describe())


# ============================================================
# 30. COMPARE CENMO1 AND CENCA1 IN 2024
# ============================================================

comparison_2024 = df_clean[
    (df_clean["sensor_name"].isin(["CENMO1", "CENCA1"]))
    & (df_clean["date"].dt.year == 2024)
].copy()

# 30.1 Boxplots
plt.figure(figsize=(9, 6))

sns.boxplot(
    data=comparison_2024,
    x="sensor_name",
    y="pm25"
)

plt.title("PM2.5 Distribution in 2024: CENMO1 vs CENCA1")
plt.xlabel("Station")
plt.ylabel("PM2.5")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "09_cenmo1_cenca1_boxplot_2024.png", dpi=300)
plt.show()

# 30.2 Descriptive statistics
print("\nDESCRIPTIVE STATISTICS - PM2.5 IN 2024:")

print(
    comparison_2024.groupby("sensor_name")["pm25"]
    .describe()
    .round(2)
    .to_string()
)

# 30.3 Quartiles and IQR
print("\nQUARTILES AND IQR:")

for station_name in ["CENMO1", "CENCA1"]:

    values = comparison_2024[
        comparison_2024["sensor_name"] == station_name
    ]["pm25"].dropna()

    Q1 = values.quantile(0.25)
    Q2 = values.quantile(0.50)
    Q3 = values.quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = values[
        (values < lower_limit)
        | (values > upper_limit)
    ]

    print(f"\nStation: {station_name}")
    print("Q1:", round(Q1, 2))
    print("Median:", round(Q2, 2))
    print("Q3:", round(Q3, 2))
    print("IQR:", round(IQR, 2))
    print("Lower limit:", round(lower_limit, 2))
    print("Upper limit:", round(upper_limit, 2))
    print("Number of outliers:", len(outliers))
    print("Maximum:", round(values.max(), 2))


# ============================================================
# 31. CENMO1 IQR OUTLIERS IN 2024
# ============================================================

station_name = "CENMO1"

values = comparison_2024[
    comparison_2024["sensor_name"] == station_name
][["date", "pm25"]].dropna()

Q1 = values["pm25"].quantile(0.25)
Q3 = values["pm25"].quantile(0.75)
IQR = Q3 - Q1

upper_limit = Q3 + 1.5 * IQR
lower_limit = Q1 - 1.5 * IQR

outliers = values[
    (values["pm25"] < lower_limit)
    | (values["pm25"] > upper_limit)
].sort_values("pm25", ascending=False)

print("\nCENMO1 - IQR OUTLIERS IN 2024:")
print(outliers.to_string(index=False))

print("\nNumber of outliers:", len(outliers))
print("Upper limit:", round(upper_limit, 2))
print("Maximum outlier:", round(outliers["pm25"].max(), 2))


# ============================================================
# 32. MOMPEAN YEARLY ANALYSIS
# ============================================================

mompean = df_clean[
    df_clean["sensor_name"] == "MOMPEAN"
].copy()

mompean["year"] = mompean["date"].dt.year

yearly_counts_mompean = (
    mompean.groupby("year")["pm25"]
    .agg(["count", "mean", "median"])
)

print("\nMOMPEAN - YEARLY ANALYSIS:")
print(yearly_counts_mompean.round(2).to_string())

ax = yearly_counts_mompean["count"].plot(
    kind="bar",
    figsize=(10, 5),
    title="Number of Valid PM2.5 Observations per Year - MOMPEAN"
)

ax.set_xlabel("Year")
ax.set_ylabel("Number of observations")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "10_mompean_yearly_observations.png", dpi=300)
plt.show()


# ============================================================
# 33. MOMPEAN IQR OUTLIERS IN 2024
# ============================================================

station_name = "MOMPEAN"

values = (
    df_clean[
        (df_clean["sensor_name"] == station_name)
        & (df_clean["date"].dt.year == 2024)
    ][["date", "pm25"]]
    .dropna()
)

Q1 = values["pm25"].quantile(0.25)
Q3 = values["pm25"].quantile(0.75)
IQR = Q3 - Q1

upper_limit = Q3 + 1.5 * IQR
lower_limit = Q1 - 1.5 * IQR

outliers = values[
    (values["pm25"] < lower_limit)
    | (values["pm25"] > upper_limit)
].sort_values("pm25", ascending=False)

print("\nMOMPEAN - IQR OUTLIERS IN 2024:")
print(outliers.to_string(index=False))

print("\nNumber of outliers:", len(outliers))
print("Upper limit:", round(upper_limit, 2))
print("Maximum value:", round(values["pm25"].max(), 2))


# ============================================================
# 34. EXAMINE TEMPORAL GAPS IN MOMPEAN
# ============================================================

station_name = "MOMPEAN"

station_data = (
    df_time[df_time["sensor_name"] == station_name]
    .sort_values("date")
    .copy()
)

station_data["gap_days"] = (
    station_data["date"].diff().dt.days
)

print("\nMOMPEAN - TEMPORAL GAP STATISTICS:")
print(station_data["gap_days"].describe())

print("\n10 LARGEST TEMPORAL GAPS:")
print(
    station_data.nlargest(10, "gap_days")[
        ["date", "gap_days", "pm25"]
    ].to_string(index=False)
)

print("\nNUMBER OF CONSECUTIVE-DAY OBSERVATIONS:")
print((station_data["gap_days"] == 1).sum())

print("\nNUMBER OF GAPS LONGER THAN 7 DAYS:")
print((station_data["gap_days"] > 7).sum())


# ============================================================
# 35. COMPARE THE THREE CANDIDATE STATIONS
# ============================================================

stations = ["CENMO1", "CENCA1", "MOMPEAN"]

comparison = []

for station_name in stations:

    # All observations of the station
    data = df_clean[
        df_clean["sensor_name"] == station_name
    ].copy()

    # Valid PM2.5 observations
    valid_data = (
        df_time[df_time["sensor_name"] == station_name]
        .sort_values("date")
        .copy()
    )

    total = len(data)
    valid = data["pm25"].count()
    missing = data["pm25"].isnull().sum()

    # Temporal gaps
    gaps = valid_data["date"].diff().dt.days.dropna()

    comparison.append({
        "station": station_name,
        "total_rows": total,
        "valid_pm25": valid,
        "missing_pm25": missing,
        "missing_percentage": (
            missing / total * 100 if total > 0 else 0
        ),
        "distinct_days": valid_data["date"].nunique(),
        "median_gap": gaps.median(),
        "max_gap": gaps.max(),
        "consecutive_percentage": (
            (gaps == 1).sum() / len(gaps) * 100
            if len(gaps) > 0 else 0
        )
    })

comparison_df = pd.DataFrame(comparison)

print("\nCOMPARISON OF CANDIDATE STATIONS:")
print(comparison_df.round(2).to_string(index=False))


# ============================================================
# 36. CHECK MISSING CALENDAR DAYS FOR CENMO1
# ============================================================

station_name = "CENMO1"

data = (
    df_time[df_time["sensor_name"] == station_name]
    .sort_values("date")
    .copy()
)

full_dates = pd.date_range(
    start=data["date"].min(),
    end=data["date"].max(),
    freq="D"
)

missing_dates = full_dates.difference(data["date"])

print("\nCENMO1 TEMPORAL COVERAGE")
print("First date:", data["date"].min())
print("Last date:", data["date"].max())
print("Total calendar days:", len(full_dates))
print("Observed days:", data["date"].nunique())
print("Missing calendar days:", len(missing_dates))

print("\nFIRST 20 MISSING DATES:")
print(missing_dates[:20])


# ============================================================
# 37. ANALYZE THE DISTRIBUTION OF TEMPORAL GAPS IN CENMO1
# ============================================================

gaps = (
    data["date"]
    .sort_values()
    .diff()
    .dt.days
    .dropna()
)

gap_categories = {
    "1 day": (gaps == 1).sum(),
    "2 to 3 days": gaps.between(2, 3).sum(),
    "4 to 7 days": gaps.between(4, 7).sum(),
    "8 to 30 days": gaps.between(8, 30).sum(),
    "31 to 90 days": gaps.between(31, 90).sum(),
    "More than 90 days": (gaps > 90).sum()
}

print("\nCENMO1 - GAP DISTRIBUTION")

for category, count in gap_categories.items():

    percentage = count / len(gaps) * 100

    print(
        f"{category}: {count} gaps "
        f"({percentage:.2f}%)"
    )


# ============================================================
# 38. EXAMINE CENMO1 PREDICTOR AVAILABILITY
# ============================================================

station = df_clean[
    df_clean["sensor_name"] == "CENMO1"
].copy()

predictors = [
    "temperature_celsius",
    "pressure_mb",
    "wind_u",
    "wind_v",
    "NO2",
    "CO",
    "O3",
    "AOD"
]

print("\nCENMO1 - MISSING VALUES IN PREDICTORS")

missing_count = station[predictors].isnull().sum()

missing_percentage = (
    station[predictors].isnull().mean() * 100
)

predictor_analysis = pd.DataFrame({
    "missing_count": missing_count,
    "missing_percentage": missing_percentage
})

print(predictor_analysis.round(2).to_string())

print("\nTOTAL OBSERVATIONS:", len(station))


# ============================================================
# 39. CENMO1 PM2.5 OUTLIERS OVER THE ENTIRE PERIOD
# ============================================================

station = df_clean[
    df_clean["sensor_name"] == "CENMO1"
][["date", "pm25"]].dropna()

Q1 = station["pm25"].quantile(0.25)
Q3 = station["pm25"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = station[
    (station["pm25"] < lower_limit)
    | (station["pm25"] > upper_limit)
].sort_values("pm25", ascending=False)

print("\nCENMO1 - IQR ANALYSIS OVER THE ENTIRE PERIOD")
print("Q1:", round(Q1, 2))
print("Q3:", round(Q3, 2))
print("IQR:", round(IQR, 2))
print("Lower limit:", round(lower_limit, 2))
print("Upper limit:", round(upper_limit, 2))

print("\nNumber of outliers:", len(outliers))
print("Maximum value:", round(station["pm25"].max(), 2))

print("\nTOP 20 HIGHEST OUTLIERS:")
print(outliers.head(20).to_string(index=False))


# ============================================================
# 40. FINAL DESCRIPTIVE STATISTICS FOR CENMO1
# ============================================================

station = df_clean[
    df_clean["sensor_name"] == "CENMO1"
].copy()

variables = [
    "pm25",
    "temperature_celsius",
    "pressure_mb",
    "wind_u",
    "wind_v",
    "NO2",
    "CO",
    "O3",
    "AOD"
]

print("\nCENMO1 - FINAL DESCRIPTIVE STATISTICS")

print(
    station[variables]
    .describe()
    .T
    .round(2)
    .to_string()
)


# ============================================================
# 41. CHECK NO2 AND CO VALUES IN CENMO1
# ============================================================

station = df_clean[
    df_clean["sensor_name"] == "CENMO1"
].copy()

print("\nNO2 STATISTICS:")
print(station["NO2"].describe().to_string())

print("\nNO2 UNIQUE VALUES:")
print(station["NO2"].round(8).nunique())

print("\nCO STATISTICS:")
print(station["CO"].describe().to_string())

print("\nCO UNIQUE VALUES:")
print(station["CO"].round(8).nunique())

print("\nNUMBER OF ZERO VALUES:")
print("NO2:", (station["NO2"] == 0).sum())
print("CO:", (station["CO"] == 0).sum())


# ============================================================
# 42. CENMO1 CORRELATION ANALYSIS
# ============================================================

station = df_clean[
    df_clean["sensor_name"] == "CENMO1"
].copy()

variables = [
    "pm25",
    "temperature_celsius",
    "pressure_mb",
    "wind_u",
    "wind_v",
    "NO2",
    "CO",
    "O3",
    "AOD"
]

correlation_matrix = station[variables].corr()

print("\nCENMO1 - CORRELATION WITH PM2.5:")

print(
    correlation_matrix["pm25"]
    .sort_values(ascending=False)
    .round(4)
    .to_string()
)


# ============================================================
# 43. CENMO1 PM2.5 TEMPORAL STATISTICS
# ============================================================

station = (
    df_clean[df_clean["sensor_name"] == "CENMO1"]
    .sort_values("date")
    .copy()
)

print("\nCENMO1 - PM2.5 TEMPORAL STATISTICS")

print("\nYEARLY STATISTICS:")

station["year"] = station["date"].dt.year

print(
    station.groupby("year")["pm25"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .round(2)
    .to_string()
)

print("\nMONTHLY STATISTICS:")

station["month"] = station["date"].dt.month

print(
    station.groupby("month")["pm25"]
    .agg(["count", "mean", "median"])
    .round(2)
    .to_string()
)


# ============================================================
# 44. CENMO1 YEARLY TEMPORAL COVERAGE
# ============================================================

station = (
    df_time[df_time["sensor_name"] == "CENMO1"]
    .sort_values("date")
    .copy()
)

full_dates = pd.date_range(
    start=station["date"].min(),
    end=station["date"].max(),
    freq="D"
)

observed_dates = pd.DatetimeIndex(station["date"].unique())

calendar = pd.DataFrame({"date": full_dates})

calendar["observed"] = calendar["date"].isin(observed_dates)
calendar["year"] = calendar["date"].dt.year

yearly_coverage = calendar.groupby("year")["observed"].agg(
    total_days="count",
    observed_days="sum"
)

yearly_coverage["missing_days"] = (
    yearly_coverage["total_days"]
    - yearly_coverage["observed_days"]
)

yearly_coverage["missing_percentage"] = (
    yearly_coverage["missing_days"]
    / yearly_coverage["total_days"] * 100
)

print("\nCENMO1 - YEARLY TEMPORAL COVERAGE")
print(yearly_coverage.round(2).to_string())


# ============================================================
# 45. IDENTIFY CONTINUOUS PERIODS IN CENMO1
# ============================================================

station = (
    df_clean[df_clean["sensor_name"] == "CENMO1"]
    .dropna(subset=["pm25"])
    .sort_values("date")
    .drop_duplicates(subset=["date"])
    .copy()
)

# Calculate the number of days between observations
station["gap"] = station["date"].diff().dt.days

# Start a new segment whenever the gap is greater than one day
station["segment"] = (station["gap"] > 1).cumsum()

# Calculate the length of each continuous segment
segments = (
    station.groupby("segment")
    .agg(
        start_date=("date", "min"),
        end_date=("date", "max"),
        observations=("date", "count")
    )
)

segments["duration_days"] = (
    segments["end_date"] - segments["start_date"]
).dt.days + 1

print("\nCENMO1 - CONTINUOUS PERIODS")

print("\nNUMBER OF CONTINUOUS SEGMENTS:")
print(len(segments))

print("\nSEGMENT STATISTICS:")
print(segments["observations"].describe().round(2))

print("\n10 LONGEST CONTINUOUS SEGMENTS:")
print(
    segments.sort_values(
        "observations",
        ascending=False
    ).head(10).to_string()
)

print("\nSEGMENTS WITH AT LEAST 30 CONSECUTIVE OBSERVATIONS:")
print((segments["observations"] >= 30).sum())


# ============================================================
# 46. EVALUATE POSSIBLE SEQUENCE LENGTHS FOR CENMO1
# ============================================================

sequence_lengths = [3, 5, 7, 10, 14]

results = []

for length in sequence_lengths:

    possible_sequences = (
        segments["observations"]
        .apply(lambda x: max(0, x - length))
        .sum()
    )

    results.append({
        "sequence_length": length,
        "possible_sequences": possible_sequences
    })

sequence_results = pd.DataFrame(results)

print("\nCENMO1 - POSSIBLE SEQUENCES")
print(sequence_results.to_string(index=False))


# ============================================================
# 47. COMPARE CANDIDATE STATIONS
# ============================================================

candidate_stations = ["CENMO1", "CENCA1", "MOMPEAN"]

station_comparison = (
    df_clean[df_clean["sensor_name"].isin(candidate_stations)]
    .groupby("sensor_name")
    .agg(
        observations=("pm25", "count"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        mean_pm25=("pm25", "mean"),
        std_pm25=("pm25", "std")
    )
)

station_comparison["calendar_days"] = (
    station_comparison["last_date"]
    - station_comparison["first_date"]
).dt.days + 1

station_comparison["coverage_percentage"] = (
    station_comparison["observations"]
    / station_comparison["calendar_days"] * 100
)

print("\nCANDIDATE STATIONS COMPARISON:")
print(station_comparison.to_string())


# ============================================================
# 48. TEMPORAL CONTINUITY COMPARISON
# ============================================================

candidate_stations = ["CENMO1", "CENCA1", "MOMPEAN"]

continuity_results = []

for station_name in candidate_stations:

    station_data = df_clean[
        (df_clean["sensor_name"] == station_name)
        & (df_clean["pm25"].notna())
    ].sort_values("date")

    dates = (
        station_data["date"]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Calculate gaps between observations
    gaps = dates.diff().dt.days.dropna()

    # Identify consecutive-day segments
    segment_id = dates.diff().dt.days.ne(1).cumsum()
    segment_lengths = dates.groupby(segment_id).size()

    # Calculate possible sequence windows
    windows_3 = sum(max(0, n - 3) for n in segment_lengths)
    windows_5 = sum(max(0, n - 5) for n in segment_lengths)
    windows_7 = sum(max(0, n - 7) for n in segment_lengths)

    continuity_results.append({
        "station": station_name,
        "observations": len(dates),
        "median_gap": gaps.median(),
        "max_gap": gaps.max(),
        "consecutive_percentage": (
            (gaps == 1).mean() * 100 if len(gaps) > 0 else 0
        ),
        "continuous_segments": len(segment_lengths),
        "longest_segment": (
            segment_lengths.max() if len(segment_lengths) > 0 else 0
        ),
        "windows_3": windows_3,
        "windows_5": windows_5,
        "windows_7": windows_7
    })

continuity_comparison = pd.DataFrame(continuity_results)

print("\nTEMPORAL CONTINUITY COMPARISON:")
print(continuity_comparison.round(2).to_string(index=False))


# ============================================================
# 49. PREDICTOR DATA AVAILABILITY
# ============================================================

candidate_stations = ["CENMO1", "CENCA1", "MOMPEAN"]

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

required_columns = ["pm25"] + features

availability_results = []

for station_name in candidate_stations:

    station_data = df_clean[
        df_clean["sensor_name"] == station_name
    ].sort_values("date").copy()

    missing_values = station_data[required_columns].isna().sum()

    complete_data = station_data.dropna(
        subset=required_columns
    )

    dates = (
        complete_data["date"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    segment_id = dates.diff().dt.days.ne(1).cumsum()
    segment_lengths = dates.groupby(segment_id).size()

    windows_7 = sum(
        max(0, n - 7) for n in segment_lengths
    )

    availability_results.append({
        "station": station_name,
        "total_rows": len(station_data),
        "missing_pm25": missing_values["pm25"],
        "missing_predictors": missing_values[features].sum(),
        "complete_rows": len(complete_data),
        "complete_percentage": (
            len(complete_data) / len(station_data) * 100
            if len(station_data) > 0 else 0
        ),
        "complete_7_day_windows": windows_7
    })

availability_comparison = pd.DataFrame(availability_results)

print("\nPREDICTOR DATA AVAILABILITY:")
print(availability_comparison.round(2).to_string(index=False))


# ============================================================
# END OF STATION ANALYSIS - SECTIONS 25 TO 49
# ============================================================

print("\n" + "=" * 60)
print("STATION ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nFigures saved in:")
print(FIGURES_DIR)

