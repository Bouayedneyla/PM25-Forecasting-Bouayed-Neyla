
import pandas as pd
import numpy as np

# 1. Load the dataset
df = pd.read_csv("combined_gee_pm25.csv")
# Create a cleaned copy of the dataset

df_clean = df.copy()
df_clean["date"] = pd.to_datetime(df_clean["date"])

# Replace negative PM2.5 values with NaN
df_clean.loc[df_clean["pm25"] < 0, "pm25"] = np.nan

# 2. Display the first 5 rows
print("FIRST 5 ROWS:")
print(df.head())

# 3. Display dataset dimensions
print("\nDATASET SHAPE:")
print(df.shape)

# 4. Display column names
print("\nCOLUMN NAMES:")
print(df.columns.tolist())

# 5. Display data types
print("\nDATA TYPES:")
print(df.dtypes)

# 6. Display dataset information
print("\nDATASET INFORMATION:")
df.info()

# 7. Display basic statistics
print("\nDESCRIPTIVE STATISTICS:")
print(df.describe())


#%% 2 missing values and duplicate

# 1. Count missing values
print("MISSING VALUES:")
print(df.isnull().sum())

# 2. Calculate missing value percentages
print("\nMISSING VALUES PERCENTAGE:")
print((df.isnull().sum() / len(df)) * 100)

# 3. Count exact duplicate rows
print("\nEXACT DUPLICATES:")
print(df.duplicated().sum())

# 4. Check duplicated station-date combinations
print("\nDUPLICATED STATION-DATE COMBINATIONS:")
print(df.duplicated(subset=["date", "sensor_name"]).sum())

# 5. Display some rows containing missing values
print("\nROWS WITH MISSING VALUES:")
print(df[df.isnull().any(axis=1)].head(10))


#%% 3 Examiner les deux observations répétées

# Find duplicated station-date combinations
duplicates = df[
    df.duplicated(
        subset=["date", "sensor_name"],
        keep=False
    )
].sort_values(["sensor_name", "date"])

print("DUPLICATED OBSERVATIONS:")
print(duplicates.to_string(index=False))

#%% 4 Descriptive Statistics

# Select numerical columns
numeric_cols = df.select_dtypes(include=np.number).columns

# Create a temporary copy for statistical analysis
df_stats = df_clean.copy()

# 1. Basic descriptive statistics
print("DESCRIPTIVE STATISTICS:")
print(df_stats[numeric_cols].describe().T)

# 2. Skewness
print("\nSKEWNESS:")
print(df_stats[numeric_cols].skew())

# 3. Kurtosis
print("\nKURTOSIS:")
print(df_stats[numeric_cols].kurtosis())

#%% 5 Data Visualization

import matplotlib.pyplot as plt
import seaborn as sns

# Variables to visualize
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

# Temporary copy for visualization
df_plot = df_clean.copy()

# 1. Histograms
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
plt.show()

# 2. Boxplots
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
plt.show()

#%% 6 vérifier que les valeurs -999 ont bien été exclues de tes graphiques PM2.5.

# Check invalid PM2.5 values
print("Number of -999 values:", (df["pm25"] == -999).sum())

print("Number of 1000 values:", (df["pm25"] == 1000).sum())

# Check the temporary plotting dataframe
print("\nPM2.5 minimum in df_plot:")
print(df_plot["pm25"].min())

print("\nPM2.5 maximum in df_plot:")
print(df_plot["pm25"].max())

#%% 7 Vérifions les valeurs négatives

# 1. Count all negative PM2.5 values
print("NUMBER OF NEGATIVE PM2.5 VALUES:")
print((df["pm25"] < 0).sum())

# 2. Display all negative values
print("\nNEGATIVE PM2.5 OBSERVATIONS:")
print(
    df.loc[df["pm25"] < 0,
           ["date", "id", "sensor_name", "pm25"]]
    .sort_values("pm25")
    .to_string(index=False)
)

# 3. Display the two highest values
print("\nHIGHEST PM2.5 VALUES:")
print(
    df.nlargest(10, "pm25")[
        ["date", "id", "sensor_name", "pm25"]
    ].to_string(index=False)
)

#%% 8 Analyse des valeurs très élevées

# 1. Most frequent negative PM2.5 values
print("FREQUENCY OF NEGATIVE VALUES:")
print(
    df.loc[df["pm25"] < 0, "pm25"]
    .value_counts()
    .head(15)
)

# 2. Number of negative values by date
print("\nNEGATIVE VALUES BY DATE:")
print(
    df.loc[df["pm25"] < 0]
    .groupby("date")
    .size()
    .sort_values(ascending=False)
    .head(15)
)

# 3. Number of negative values by station
print("\nNEGATIVE VALUES BY STATION:")
print(
    df.loc[df["pm25"] < 0]
    .groupby("sensor_name")
    .size()
    .sort_values(ascending=False)
    .head(15)
)

# 4. Highest positive PM2.5 values
print("\nVALUES GREATER THAN 300:")
print(
    df.loc[df["pm25"] > 300,
           ["date", "sensor_name", "pm25"]]
    .sort_values("pm25", ascending=False)
    .to_string(index=False)
)

#%% 9 créer une copie nettoyée

# Create a cleaned copy of the original dataset
df_clean = df.copy()

# Convert date column to datetime
df_clean["date"] = pd.to_datetime(df_clean["date"])

# Replace physically invalid negative PM2.5 values
negative_count = (df_clean["pm25"] < 0).sum()

df_clean.loc[df_clean["pm25"] < 0, "pm25"] = np.nan

print("DATA CLEANING SUMMARY")
print("Original rows:", len(df))
print("Negative PM2.5 values replaced:", negative_count)
print("Missing PM2.5 after cleaning:", df_clean["pm25"].isna().sum())
print("Minimum valid PM2.5:", df_clean["pm25"].min())
print("Maximum valid PM2.5:", df_clean["pm25"].max())

# Check the cleaning results
print("Original number of rows:", len(df))
print("Number of missing PM2.5 values:", df_clean["pm25"].isnull().sum())

print("\nMinimum PM2.5 value after cleaning:")
print(df_clean["pm25"].min())

print("\nMaximum PM2.5 value after cleaning:")
print(df_clean["pm25"].max())
#%% 10 

# Check duplicated dates for the candidate stations

candidate_stations = ["CENMO1", "CENCA1", "MOMPEAN"]

candidate_duplicates = df_clean[
    df_clean["sensor_name"].isin(candidate_stations)
    & df_clean.duplicated(
        subset=["sensor_name", "date"],
        keep=False
    )
].sort_values(["sensor_name", "date"])

print("DUPLICATES IN CANDIDATE STATIONS")

if candidate_duplicates.empty:
    print("No duplicated station-date observations found.")
else:
    print(candidate_duplicates[
        ["date", "id", "sensor_name", "latitude",
         "longitude", "pm25"]
    ].to_string(index=False))

#%% 11 examiner les valeurs manquantes
# Check missing values after cleaning
print("MISSING VALUES AFTER CLEANING:")
print(df_clean.isnull().sum())

print("\nMISSING VALUES PERCENTAGE:")
print((df_clean.isnull().sum() / len(df_clean) * 100).round(2))

print("\nTOTAL NUMBER OF MISSING VALUES:")
print(df_clean.isnull().sum().sum())
#%% 12 examiner les valeurs manquantes dans le temps

# Convert date column to datetime format
df_clean["date"] = pd.to_datetime(df_clean["date"])

# Count missing values by date
missing_by_date = (
    df_clean.groupby("date")[["NO2", "CO", "O3", "pm25"]]
    .apply(lambda x: x.isnull().sum())
)

# Display dates with the most missing values
print("DATES WITH THE MOST MISSING VALUES:")
print(
    missing_by_date.sum(axis=1)
    .sort_values(ascending=False)
    .head(15)
)

# Display total missing values by month
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
#%% 13 vérifier si les valeurs manquantes de NO2, CO et O3 apparaissent souvent sur les mêmes lignes.

# Check missing values combinations
missing_patterns = (
    df_clean[["NO2", "CO", "O3"]]
    .isnull()
    .value_counts()
)

print("MISSING VALUE PATTERNS:")
print(missing_patterns)

# Check rows where all three variables are missing
all_three_missing = df_clean[
    df_clean[["NO2", "CO", "O3"]].isnull().all(axis=1)
]

print("\nRows where NO2, CO and O3 are all missing:")
print(len(all_three_missing))

# Check rows where at least one of the three is missing
at_least_one_missing = df_clean[
    df_clean[["NO2", "CO", "O3"]].isnull().any(axis=1)
]

print("\nRows where at least one variable is missing:")
print(len(at_least_one_missing))
#%% 14 examiner les stations concernées
     
# Count missing values for each station
missing_by_station = (
    df_clean.groupby("sensor_name")[["NO2", "CO", "O3"]]
    .apply(lambda x: x.isnull().sum())
)

# Calculate total missing values per station
missing_by_station["total_missing"] = missing_by_station.sum(axis=1)

# Sort stations by missing values
missing_by_station = missing_by_station.sort_values(
    by="total_missing",
    ascending=False
)

print("STATIONS WITH THE MOST MISSING VALUES:")
print(missing_by_station.head(15))

# Display the total number of stations
print("\nTotal number of stations:")
print(df_clean["sensor_name"].nunique())
#%% 15 calculer le pourcentage par station

# Calculate the total number of observations per station
station_counts = df_clean.groupby("sensor_name").size()

# Calculate missing values and percentages
missing_station_analysis = (
    df_clean.groupby("sensor_name")[["NO2", "CO", "O3"]]
    .apply(lambda x: x.isnull().sum())
)

missing_station_analysis["total_observations"] = station_counts

missing_station_analysis["missing_percentage"] = (
    missing_station_analysis["total_missing"]
    if "total_missing" in missing_station_analysis.columns
    else missing_station_analysis[["NO2", "CO", "O3"]].sum(axis=1)
) / (station_counts * 3) * 100

# Sort by missing percentage
missing_station_analysis = missing_station_analysis.sort_values(
    "missing_percentage",
    ascending=False
)

print("MISSING DATA PERCENTAGE BY STATION:")
print(missing_station_analysis.head(15).round(2))
#%% 16 Vérification de la variable cible PM2.5
# Analyze missing PM2.5 values by station

pm25_station_analysis = df_clean.groupby("sensor_name").agg(
    total_observations=("pm25", "size"),
    missing_pm25=("pm25", lambda x: x.isnull().sum())
)

pm25_station_analysis["missing_percentage"] = (
    pm25_station_analysis["missing_pm25"]
    / pm25_station_analysis["total_observations"] * 100
)

# Sort stations by missing percentage
pm25_station_analysis = pm25_station_analysis.sort_values(
    "missing_percentage",
    ascending=False
)

print("PM2.5 MISSING VALUES BY STATION:")
print(pm25_station_analysis.head(15).round(2))

print("\nTOTAL STATIONS WITH MISSING PM2.5:")
print((pm25_station_analysis["missing_pm25"] > 0).sum())

print("\nTOTAL STATIONS WITHOUT MISSING PM2.5:")
print((pm25_station_analysis["missing_pm25"] == 0).sum())
#%% 17 Évolution annuelle de PM2.5
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the average PM2.5 concentration for each year
annual_pm25 = (
    df_clean.groupby(df_clean["date"].dt.year)["pm25"]
    .mean()
)

print("AVERAGE PM2.5 BY YEAR:")
print(annual_pm25.round(2))

# Plot annual PM2.5 evolution
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
plt.show()
#%% 18 analyse : évolution mensuelle
# Calculate average PM2.5 for each month
monthly_pm25 = (
    df_clean.groupby(df_clean["date"].dt.month)["pm25"]
    .mean()
)

print("AVERAGE PM2.5 BY MONTH:")
print(monthly_pm25.round(2))

# Plot monthly evolution
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
plt.show()
#%% 19 matrice de corrélation
# Select numerical variables
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

# Calculate correlation matrix
correlation_matrix = df_clean[correlation_cols].corr()

print("CORRELATION MATRIX:")
print(correlation_matrix.round(2))

# Plot correlation heatmap
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
plt.show()
#%% 20 comparer les stations
# Calculate average PM2.5 for each station
station_pm25 = (
    df_clean.groupby("sensor_name")["pm25"]
    .mean()
    .sort_values(ascending=False)
)

print("TOP 15 STATIONS WITH HIGHEST AVERAGE PM2.5:")
print(station_pm25.head(15).round(2))

print("\nBOTTOM 15 STATIONS WITH LOWEST AVERAGE PM2.5:")
print(station_pm25.tail(15).round(2))


# Plot the 15 stations with highest average PM2.5
plt.figure(figsize=(12, 6))

sns.barplot(
    x=station_pm25.head(15).values,
    y=station_pm25.head(15).index
)

plt.title("Top 15 Stations by Average PM2.5")
plt.xlabel("Average PM2.5")
plt.ylabel("Station")
plt.tight_layout()
plt.show()
#%% 21 vérifier les stations extrêmes
stations_to_check = [
    "Gibraltar",
    "064F",
    "mimiz",
    "testcompnex"
]

for station in stations_to_check:
    station_data = df_clean[df_clean["sensor_name"] == station]["pm25"]

    print("\nStation:", station)
    print("Number of observations:", station_data.count())
    print("Missing values:", station_data.isnull().sum())
    print("Minimum:", station_data.min())
    print("Median:", station_data.median())
    print("Mean:", station_data.mean())
    print("Maximum:", station_data.max())
#%% 22 examiner le nombre d'observations par station
# Count valid PM2.5 observations for each station
station_counts = (
    df_clean.groupby("sensor_name")["pm25"]
    .count()
    .sort_values(ascending=True)
)

print("15 STATIONS WITH THE FEWEST OBSERVATIONS:")
print(station_counts.head(15))

print("\n15 STATIONS WITH THE MOST OBSERVATIONS:")
print(station_counts.tail(15))

print("\nGENERAL STATISTICS:")
print(station_counts.describe())
#%% 23 vérifier la couverture temporelle
# Analyze temporal coverage for each station

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
    "distinct_days", ascending=True
)

print("15 STATIONS WITH THE SHORTEST TEMPORAL COVERAGE:")
print(station_periods.head(15))

print("\n15 STATIONS WITH THE LONGEST TEMPORAL COVERAGE:")
print(station_periods.tail(15))
#%% 24 mesurer les interruptions temporelles
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

# Display stations with the most consecutive observations
print("15 STATIONS WITH THE HIGHEST CONSECUTIVE-DAY PERCENTAGE:")
print(
    gap_stats.sort_values(
        "consecutive_percentage", ascending=False
    ).head(15).round(2)
)

# Display stations with the largest interruptions
print("\n15 STATIONS WITH THE LARGEST TEMPORAL GAPS:")
print(
    gap_stats.sort_values(
        "max_gap", ascending=False
    ).head(15).round(2)
)
#%% 25 Afficher les valeurs complètes
# Display the largest temporal gaps clearly

print(
    gap_stats.sort_values(
        "max_gap", ascending=False
    )[["max_gap", "median_gap", "number_of_gaps",
       "consecutive_percentage"]].head(15).to_string()
)
#%% 26 examiner les interruptions d'une station
# Select CENMO1
station_name = "CENMO1"

station_data = (
    df_time[df_time["sensor_name"] == station_name]
    .sort_values("date")
    .copy()
)

# Calculate gaps between consecutive observations
station_data["gap_days"] = (
    station_data["date"].diff().dt.days
)

# Display the largest interruptions
print("Largest gaps for", station_name)

print(
    station_data.nlargest(10, "gap_days")[
        ["date", "gap_days", "pm25"]
    ].to_string(index=False)
)

# Display general gap statistics
print("\nGap statistics:")
print(station_data["gap_days"].describe())
#%% 27 examiner les données de CENMO1
# Analyze CENMO1 observations by year

cenmo1 = df_clean[
    df_clean["sensor_name"] == "CENMO1"
].copy()

cenmo1["year"] = cenmo1["date"].dt.year

yearly_counts = (
    cenmo1.groupby("year")["pm25"]
    .agg(["count", "mean", "median"])
)

print("CENMO1 - YEARLY ANALYSIS:")
print(yearly_counts.round(2))

# Plot the number of observations per year
yearly_counts["count"].plot(
    kind="bar",
    figsize=(10, 5),
    title="Number of Valid PM2.5 Observations per Year - CENMO1"
)

plt.xlabel("Year")
plt.ylabel("Number of observations")
plt.tight_layout()
plt.show()
#%% 28 examiner une deuxième station :CENCA1
# Analyze CENCA1 observations by year

cenca1 = df_clean[
    df_clean["sensor_name"] == "CENCA1"
].copy()

cenca1["year"] = cenca1["date"].dt.year

yearly_counts_cenca1 = (
    cenca1.groupby("year")["pm25"]
    .agg(["count", "mean", "median"])
)

print("CENCA1 - YEARLY ANALYSIS:")
print(yearly_counts_cenca1.round(2))

# Plot the number of observations per year
yearly_counts_cenca1["count"].plot(
    kind="bar",
    figsize=(10, 5),
    title="Number of Valid PM2.5 Observations per Year - CENCA1"
)

plt.xlabel("Year")
plt.ylabel("Number of observations")
plt.tight_layout()
plt.show()
#%% 29 examiner les valeurs extrêmes de CENCA1
# Examine CENCA1 PM2.5 values in 2024

cenca1_2024 = cenca1[
    (cenca1["year"] == 2024) &
    (cenca1["pm25"].notna())
].copy()

print("CENCA1 - HIGHEST PM2.5 VALUES IN 2024:")

print(
    cenca1_2024.nlargest(15, "pm25")[
        ["date", "pm25"]
    ].to_string(index=False)
)

print("\nCENCA1 - DESCRIPTIVE STATISTICS FOR 2024:")
print(cenca1_2024["pm25"].describe())
#%% 30 visualiser la distribution de CENCA1
# Compare PM2.5 distributions in 2024

comparison_2024 = df_clean[
    (df_clean["sensor_name"].isin(["CENMO1", "CENCA1"])) &
    (df_clean["date"].dt.year == 2024)
].copy()


# 1. Display boxplots
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
plt.show()


# 2. Display descriptive statistics
print("\nDESCRIPTIVE STATISTICS - PM2.5 IN 2024:")

print(
    comparison_2024.groupby("sensor_name")["pm25"]
    .describe()
    .round(2)
    .to_string()
)


# 3. Display quartiles and IQR
print("\nQUARTILES AND IQR:")

for station in ["CENMO1", "CENCA1"]:

    values = comparison_2024[
        comparison_2024["sensor_name"] == station
    ]["pm25"].dropna()

    Q1 = values.quantile(0.25)
    Q2 = values.quantile(0.50)
    Q3 = values.quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = values[
        (values < lower_limit) |
        (values > upper_limit)
    ]

    print(f"\nStation: {station}")
    print("Q1:", round(Q1, 2))
    print("Median:", round(Q2, 2))
    print("Q3:", round(Q3, 2))
    print("IQR:", round(IQR, 2))
    print("Lower limit:", round(lower_limit, 2))
    print("Upper limit:", round(upper_limit, 2))
    print("Number of outliers:", len(outliers))
    print("Maximum:", round(values.max(), 2))
#%% 31 examiner les valeurs atypiques de CENMO1
# Display IQR outliers for CENMO1 in 2024

station = "CENMO1"

values = comparison_2024[
    comparison_2024["sensor_name"] == station
][["date", "pm25"]].dropna()

Q1 = values["pm25"].quantile(0.25)
Q3 = values["pm25"].quantile(0.75)
IQR = Q3 - Q1

upper_limit = Q3 + 1.5 * IQR
lower_limit = Q1 - 1.5 * IQR

outliers = values[
    (values["pm25"] < lower_limit) |
    (values["pm25"] > upper_limit)
].sort_values("pm25", ascending=False)

print("CENMO1 - IQR OUTLIERS IN 2024:")
print(outliers.to_string(index=False))

print("\nNumber of outliers:", len(outliers))
print("Upper limit:", round(upper_limit, 2))
print("Maximum outlier:", round(outliers["pm25"].max(), 2))
#%% 32 examiner MOMPEAN
# Analyze MOMPEAN observations by year

mompean = df_clean[
    df_clean["sensor_name"] == "MOMPEAN"
].copy()

mompean["year"] = mompean["date"].dt.year

yearly_counts_mompean = (
    mompean.groupby("year")["pm25"]
    .agg(["count", "mean", "median"])
)

print("MOMPEAN - YEARLY ANALYSIS:")
print(yearly_counts_mompean.round(2).to_string())

# Plot the number of observations per year
yearly_counts_mompean["count"].plot(
    kind="bar",
    figsize=(10, 5),
    title="Number of Valid PM2.5 Observations per Year - MOMPEAN"
)

plt.xlabel("Year")
plt.ylabel("Number of observations")
plt.tight_layout()
plt.show()
#%% 33 examiner les valeurs atypiques de MOMPEAN
# Display IQR outliers for MOMPEAN in 2024

station = "MOMPEAN"

values = (
    df_clean[
        (df_clean["sensor_name"] == station) &
        (df_clean["date"].dt.year == 2024)
    ][["date", "pm25"]]
    .dropna()
)

Q1 = values["pm25"].quantile(0.25)
Q3 = values["pm25"].quantile(0.75)
IQR = Q3 - Q1

upper_limit = Q3 + 1.5 * IQR
lower_limit = Q1 - 1.5 * IQR

outliers = values[
    (values["pm25"] < lower_limit) |
    (values["pm25"] > upper_limit)
].sort_values("pm25", ascending=False)

print("MOMPEAN - IQR OUTLIERS IN 2024:")
print(outliers.to_string(index=False))

print("\nNumber of outliers:", len(outliers))
print("Upper limit:", round(upper_limit, 2))
print("Maximum value:", round(values["pm25"].max(), 2))
#%% 34 vérifier les intervalles temporels de MOMPEAN
# Analyze temporal gaps for MOMPEAN

station_name = "MOMPEAN"

station_data = (
    df_time[df_time["sensor_name"] == station_name]
    .sort_values("date")
    .copy()
)

station_data["gap_days"] = (
    station_data["date"].diff().dt.days
)

print("MOMPEAN - TEMPORAL GAP STATISTICS:")
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
#%% 35 comparaison des trois stations
# Compare the three candidate stations

stations = ["CENMO1", "CENCA1", "MOMPEAN"]

comparison = []

for station in stations:

    # All observations of the station
    data = df_clean[
        df_clean["sensor_name"] == station
    ].copy()

    # Valid PM2.5 observations
    valid_data = (
        df_time[df_time["sensor_name"] == station]
        .sort_values("date")
        .copy()
    )

    total = len(data)
    valid = data["pm25"].count()
    missing = data["pm25"].isnull().sum()

    # Temporal gaps
    gaps = valid_data["date"].diff().dt.days.dropna()

    comparison.append({
        "station": station,
        "total_rows": total,
        "valid_pm25": valid,
        "missing_pm25": missing,
        "missing_percentage": missing / total * 100,
        "distinct_days": valid_data["date"].nunique(),
        "median_gap": gaps.median(),
        "max_gap": gaps.max(),
        "consecutive_percentage": (
            (gaps == 1).sum() / len(gaps) * 100
        )
    })

comparison_df = pd.DataFrame(comparison)

print("COMPARISON OF CANDIDATE STATIONS:")
print(comparison_df.round(2).to_string(index=False))
#%% 36 examiner la régularité des dates de CENMO1
# Check missing calendar days for CENMO1

station = "CENMO1"

data = (
    df_time[df_time["sensor_name"] == station]
    .sort_values("date")
    .copy()
)

full_dates = pd.date_range(
    start=data["date"].min(),
    end=data["date"].max(),
    freq="D"
)

missing_dates = full_dates.difference(data["date"])

print("CENMO1 TEMPORAL COVERAGE")
print("First date:", data["date"].min())
print("Last date:", data["date"].max())
print("Total calendar days:", len(full_dates))
print("Observed days:", data["date"].nunique())
print("Missing calendar days:", len(missing_dates))

print("\nFIRST 20 MISSING DATES:")
print(missing_dates[:20])
#%% 37 étudier la distribution des interruptions
# Analyze the distribution of temporal gaps in CENMO1

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

print("CENMO1 - GAP DISTRIBUTION")

for category, count in gap_categories.items():
    percentage = count / len(gaps) * 100

    print(
        f"{category}: {count} gaps "
        f"({percentage:.2f}%)"
    )
#%% 38 examiner les variables explicatives de CENMO1
# Analyze missing values in CENMO1 predictors

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

print("CENMO1 - MISSING VALUES IN PREDICTORS")

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
#%% 39 analyser les valeurs extrêmes de PM2.5 sur toute la période
# Analyze PM2.5 outliers in CENMO1 over the entire period

station = df_clean[
    df_clean["sensor_name"] == "CENMO1"
][["date", "pm25"]].dropna()

Q1 = station["pm25"].quantile(0.25)
Q3 = station["pm25"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = station[
    (station["pm25"] < lower_limit) |
    (station["pm25"] > upper_limit)
].sort_values("pm25", ascending=False)

print("CENMO1 - IQR ANALYSIS OVER THE ENTIRE PERIOD")
print("Q1:", round(Q1, 2))
print("Q3:", round(Q3, 2))
print("IQR:", round(IQR, 2))
print("Lower limit:", round(lower_limit, 2))
print("Upper limit:", round(upper_limit, 2))

print("\nNumber of outliers:", len(outliers))
print("Maximum value:", round(station["pm25"].max(), 2))

print("\nTOP 20 HIGHEST OUTLIERS:")
print(outliers.head(20).to_string(index=False))
#%% 40 vérifier les statistiques finales de CENMO1
# Final descriptive statistics for CENMO1

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

print("CENMO1 - FINAL DESCRIPTIVE STATISTICS")

print(
    station[variables]
    .describe()
    .T
    .round(2)
    .to_string()
)
#%% 41 vérification importante : NO2 et CO
# Check NO2 and CO with higher precision

station = df_clean[
    df_clean["sensor_name"] == "CENMO1"
].copy()

print("NO2 STATISTICS:")
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
#%% 42 vérifier les corrélations propres à CENMO1
# Correlation analysis for CENMO1

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

print("CENMO1 - CORRELATION WITH PM2.5:")

print(
    correlation_matrix["pm25"]
    .sort_values(ascending=False)
    .round(4)
    .to_string()
)
#%% 43 vérifier les valeurs de PM2.5 dans le temps
# PM2.5 temporal analysis for CENMO1

station = (
    df_clean[df_clean["sensor_name"] == "CENMO1"]
    .sort_values("date")
    .copy()
)

print("CENMO1 - PM2.5 TEMPORAL STATISTICS")

print("\nYearly statistics:")

station["year"] = station["date"].dt.year

print(
    station.groupby("year")["pm25"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .round(2)
    .to_string()
)

print("\nMonthly statistics:")

station["month"] = station["date"].dt.month

print(
    station.groupby("month")["pm25"]
    .agg(["count", "mean", "median"])
    .round(2)
    .to_string()
)
#%% 44 examiner les valeurs manquantes dans les dates de CENMO1
# Missing calendar days per year for CENMO1

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
    yearly_coverage["total_days"] -
    yearly_coverage["observed_days"]
)

yearly_coverage["missing_percentage"] = (
    yearly_coverage["missing_days"] /
    yearly_coverage["total_days"] * 100
)

print("CENMO1 - YEARLY TEMPORAL COVERAGE")
print(yearly_coverage.round(2).to_string())
#%% 45 Identifier les périodes continues de CENMO1
# Identify continuous daily periods for CENMO1

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

print("CENMO1 - CONTINUOUS PERIODS")

print("\nNumber of continuous segments:")
print(len(segments))

print("\nSegment statistics:")
print(segments["observations"].describe().round(2))

print("\n10 longest continuous segments:")
print(
    segments.sort_values(
        "observations", ascending=False
    ).head(10).to_string()
)

print("\nSegments with at least 30 consecutive observations:")
print((segments["observations"] >= 30).sum())
#%% 46 déterminer les longueurs de séquences possibles
# Evaluate possible sequence lengths for CENMO1

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

print("CENMO1 - POSSIBLE SEQUENCES")
print(sequence_results.to_string(index=False))
#%% 47 Comparison of Candidate Stations
# Compare the three candidate stations

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

print("CANDIDATE STATIONS COMPARISON:")
print(station_comparison.to_string())
#%% 48 – Temporal Continuity Comparison
# 48. TEMPORAL CONTINUITY COMPARISON

candidate_stations = ["CENMO1", "CENCA1", "MOMPEAN"]

continuity_results = []

for station in candidate_stations:

    station_data = df_clean[
        (df_clean["sensor_name"] == station)
        & (df_clean["pm25"].notna())
    ].sort_values("date")

    dates = station_data["date"].drop_duplicates().reset_index(drop=True)

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
        "station": station,
        "observations": len(dates),
        "median_gap": gaps.median(),
        "max_gap": gaps.max(),
        "consecutive_percentage": (gaps == 1).mean() * 100,
        "continuous_segments": len(segment_lengths),
        "longest_segment": segment_lengths.max(),
        "windows_3": windows_3,
        "windows_5": windows_5,
        "windows_7": windows_7
    })

continuity_comparison = pd.DataFrame(continuity_results)

print("TEMPORAL CONTINUITY COMPARISON:")
print(continuity_comparison.round(2).to_string(index=False))
#%% 49. PREDICTOR DATA AVAILABILITY

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

for station in candidate_stations:

    station_data = df_clean[
        df_clean["sensor_name"] == station
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
        "station": station,
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

print("PREDICTOR DATA AVAILABILITY:")
print(availability_comparison.round(2).to_string(index=False))

#%% 50. PREPARE THE SELECTED STATION

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

print("SELECTED STATION:", selected_station)

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

#%% 51 vérifier les intervalles temporels
#  CONSECUTIVE-DAY ANALYSIS

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

print("CONSECUTIVE-DAY ANALYSIS – CENCA1")

print("\nGAP STATISTICS:")
print("Median gap:", gaps.median(), "days")
print("Maximum gap:", gaps.max(), "days")
print("Consecutive-day percentage:", round((gaps == 1).mean() * 100, 2), "%")

print("\nCONTINUOUS SEGMENTS:")
print("Number of segments:", len(segment_lengths))
print("Longest segment:", segment_lengths.max(), "days")
print("Average segment length:", round(segment_lengths.mean(), 2), "days")

print("\nSEQUENCE ANALYSIS:")
print("Input sequence length:", sequence_length, "days")
print("Possible 3-day input sequences with next-day target:", possible_sequences)
#%% 52. FINAL DATASET PREPARATION AND EXPORT

# Sort observations chronologically
df_final = df_selected.sort_values("date").reset_index(drop=True)

# Verify data quality
print("FINAL DATASET CHECK")

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
    r"C:\Users\pc\Downloads\dataset projet\CENCA1_pm25_preprocessed.csv",
    index=False
)

print("\nDataset successfully saved!")

#%% # 53. VERIFY THE EXPORTED DATASET

# Reload the saved CSV file
df_check = pd.read_csv("CENCA1_pm25_preprocessed.csv")

# Convert date column back to datetime
df_check["date"] = pd.to_datetime(df_check["date"])

print("EXPORTED DATASET VERIFICATION")

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

