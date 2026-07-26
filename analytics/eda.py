import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

plt.style.use("ggplot")

pd.set_option("display.max_columns", None)

print("=" * 60)
print("LOADING TITANIC DATASET")
print("=" * 60)

df = sns.load_dataset("titanic")

os.makedirs("analytics/data", exist_ok=True)

df.to_csv("analytics/data/titanic.csv", index=False)

print("Dataset Loaded Successfully")
print("Offline copy saved as analytics/data/titanic.csv")

print("\nSHAPE")
print(df.shape)

print("\nINFO")
df.info()

print("\nDESCRIBE")
print(df.describe(include="all"))
print("\nMISSING VALUE PERCENTAGES")

missing = (
    df.isnull()
      .mean()
      .mul(100)
      .round(2)
)

missing = missing[missing > 0]

print(missing)

clean_df = df.copy()

# Age (~20%) → Median imputation
clean_df["age"] = clean_df["age"].fillna(clean_df["age"].median())

# Embarked (<5%) → Drop rows
clean_df = clean_df.dropna(subset=["embarked"])

# Embark Town (<5%) → Drop rows
clean_df = clean_df.dropna(subset=["embark_town"])

# Deck (~77%) → Drop column
clean_df = clean_df.drop(columns=["deck"])

print("\nMissing values after cleaning")
print(clean_df.isnull().sum())
clean_df.to_csv(
    "analytics/data/titanic_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved.")

print("\nAGE DISTRIBUTION")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(clean_df["age"], bins=30, kde=True, ax=axes[0])
axes[0].set_title("Age Histogram")

sns.boxplot(x=clean_df["age"], ax=axes[1])
axes[1].set_title("Age Box Plot")

plt.tight_layout()
plt.savefig("analytics/outputs/age_analysis.png")
plt.show()

print("\nFARE DISTRIBUTION")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(clean_df["fare"], bins=30, kde=True, ax=axes[0])
axes[0].set_title("Fare Histogram")

sns.boxplot(x=clean_df["fare"], ax=axes[1])
axes[1].set_title("Fare Box Plot")

plt.tight_layout()
plt.savefig("analytics/outputs/fare_analysis.png")
plt.show()
def outlier_count(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return ((df[column] < lower) | (df[column] > upper)).sum()


age_outliers = outlier_count(clean_df, "age")
fare_outliers = outlier_count(clean_df, "fare")

print("\nOUTLIERS")
print(f"Age Outliers  : {age_outliers}")
print(f"Fare Outliers : {fare_outliers}")

fare_mean = clean_df["fare"].mean()
fare_median = clean_df["fare"].median()
fare_mode = clean_df["fare"].mode()[0]

print("\nFARE STATISTICS")
print(f"Mean   : {fare_mean:.2f}")
print(f"Median : {fare_median:.2f}")
print(f"Mode   : {fare_mode:.2f}")

print("\nSKEWNESS ANALYSIS")

if fare_mean > fare_median > fare_mode:
    print("Fare is Right-Skewed.")
elif fare_mean < fare_median < fare_mode:
    print("Fare is Left-Skewed.")
else:
    print("Fare distribution is approximately symmetric.")

print("\n" + "=" * 60)
print("SURVIVAL RATE BY SEX")
print("=" * 60)

survival_by_sex = (
    clean_df.groupby("sex")["survived"]
    .mean()
    .mul(100)
    .round(2)
)

print(survival_by_sex)

print("\n" + "=" * 60)
print("SURVIVAL RATE BY PASSENGER CLASS")
print("=" * 60)

survival_by_class = (
    clean_df.groupby("pclass")["survived"]
    .mean()
    .mul(100)
    .round(2)
)

print(survival_by_class)

print("\n" + "=" * 60)
print("SURVIVAL RATE BY SEX AND PASSENGER CLASS")
print("=" * 60)

survival_by_both = (
    clean_df.groupby(["sex", "pclass"])["survived"]
    .mean()
    .mul(100)
    .round(2)
)

print(survival_by_both)

corr_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

corr = clean_df[corr_columns].corr()

print("\nCorrelation Matrix")
print(corr)

plt.figure(figsize=(8, 6))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("analytics/outputs/correlation_heatmap.png")

plt.show()

corr_abs = corr.abs()

mask = np.triu(np.ones_like(corr_abs, dtype=bool))

pairs = (
    corr_abs.where(~mask)
    .stack()
    .sort_values(ascending=False)
)

print("\nTwo Strongest Correlations")
print(pairs.head(2))

print("\nCHART 1 : SURVIVAL BY SEX")

plt.figure(figsize=(6,5))

sns.barplot(
    data=clean_df,
    x="sex",
    y="survived",
    estimator=np.mean
)

plt.title("Survival Rate by Sex")
plt.ylabel("Average Survival Rate")

plt.tight_layout()
plt.savefig("analytics/outputs/chart1_survival_by_sex.png")
plt.show()

print("\nCHART 2 : SURVIVAL BY CLASS")

plt.figure(figsize=(6,5))

sns.barplot(
    data=clean_df,
    x="pclass",
    y="survived",
    estimator=np.mean
)

plt.title("Survival Rate by Passenger Class")
plt.ylabel("Average Survival Rate")

plt.tight_layout()
plt.savefig("analytics/outputs/chart2_survival_by_class.png")
plt.show()

print("\nCHART 3 : FARE VS SURVIVAL")

plt.figure(figsize=(8,5))

sns.boxplot(
    data=clean_df,
    x="survived",
    y="fare"
)

plt.title("Fare Distribution by Survival")

plt.tight_layout()
plt.savefig("analytics/outputs/chart3_fare_vs_survival.png")
plt.show()

print("\nCHART 4 : AGE BY SURVIVAL")

plt.figure(figsize=(8,5))

sns.boxplot(
    data=clean_df,
    x="survived",
    y="age"
)

plt.title("Age Distribution by Survival")

plt.tight_layout()
plt.savefig("analytics/outputs/chart4_age_vs_survival.png")
plt.show()

print("\nSTANDARDIZATION CHECK")

scaler = StandardScaler()

scaled = scaler.fit_transform(
    clean_df[["age", "fare"]]
)

scaled_df = pd.DataFrame(
    scaled,
    columns=["age_z", "fare_z"]
)

print("\nMeans")
print(scaled_df.mean())

print("\nStandard Deviations")
print(scaled_df.std())

scaled_df.to_csv(
    "analytics/outputs/standardized_features.csv",
    index=False
)