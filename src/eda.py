"""
Exploratory Data Analysis Script
Run after data_acquisition.py to explore raw data

Developer: Eng. Mahmoud Khalid Alkodousy
"""

import matplotlib.pyplot as plt
import seaborn as sns
from data_acquisition import load_data
from functions import plot_barh, plot_hist, plot_grouped_bar

# Load data
telemetry_df, errors_df, maint_df, failures_df, machines_df = load_data()

# Basic info
print("\n===== Dataset Info =====")
print(f"Telemetry: {telemetry_df.shape}")
print(f"Errors: {errors_df.shape}")
print(f"Maintenance: {maint_df.shape}")
print(f"Failures: {failures_df.shape}")
print(f"Machines: {machines_df.shape}")

# Check missing values
print("\n===== Missing Values =====")
print(f"Telemetry: {telemetry_df.isnull().sum().sum()}")
print(f"Errors: {errors_df.isnull().sum().sum()}")
print(f"Maintenance: {maint_df.isnull().sum().sum()}")
print(f"Failures: {failures_df.isnull().sum().sum()}")
print(f"Machines: {machines_df.isnull().sum().sum()}")

# Check duplicates
print("\n===== Duplicates =====")
print(f"Telemetry: {telemetry_df.duplicated().sum()}")
print(f"Errors: {errors_df.duplicated().sum()}")
print(f"Maintenance: {maint_df.duplicated().sum()}")
print(f"Failures: {failures_df.duplicated().sum()}")
print(f"Machines: {machines_df.duplicated().sum()}")

# Statistics
print("\n===== Statistics =====")
print(telemetry_df.describe())

# Visualizations
print("\n===== Creating Visualizations =====")

# Sensor distributions
for name in ["volt", "rotate", "pressure", "vibration"]:
    plot_hist(telemetry_df, feature_name=name, log=False, bins=1000)
    plt.savefig(f"data/graphs/{name}_distribution.png")
    plt.close()

# Error types
plot_barh(errors_df, "errorID", title="Error Types Frequency", xlabel="Number of Errors")
plt.savefig("data/graphs/error_types_frequency.png")
plt.close()

# Failure types
plot_barh(failures_df, "failure", title="Failure Type Frequency", xlabel="Number of Failures")
plt.savefig("data/graphs/failure_types_frequency.png")
plt.close()

# Errors per machine
plot_grouped_bar(
    errors_df, "machineID", "errorID", "errorValues",
    "Type of Errors per Machine", "Machine ID", "Number of Errors"
)
plt.savefig("data/graphs/errors_per_machine.png")
plt.close()

# Components replaced per machine
plot_grouped_bar(
    maint_df, "machineID", "comp", "num_comp",
    "Components Replaced per Machine", "Machine ID", "Components Replaced"
)
plt.savefig("data/graphs/components_replaced_per_machine.png")
plt.close()

# Errors per day
errors_df["date"] = errors_df.datetime.dt.date
errors_df.groupby("date").size().hist(bins=20, figsize=(12, 6))
plt.title("Distribution of Errors per Day")
plt.xlabel("Errors per Day")
plt.ylabel("Frequency")
plt.savefig("data/graphs/errors_per_day.png")
plt.close()

# Correlation heatmap
features = ["volt", "rotate", "pressure", "vibration"]
corr = telemetry_df[features].corr()
plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Sensor Correlation Matrix")
plt.savefig("data/graphs/features_correlation.png")
plt.close()

print("\n✅ EDA Complete! Check data/graphs/ folder for visualizations")
