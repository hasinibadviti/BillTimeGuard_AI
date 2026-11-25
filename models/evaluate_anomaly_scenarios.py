
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow.keras.models import load_model

print("=" * 70)
print("BILLTIMEGUARD AI - INDIVIDUAL ANOMALY SCENARIO EVALUATION")
print("=" * 70)

# 1. Load original test records
data = np.load(
    "dataset/processed/legal_model_ready/legal_model_data.npz",
    allow_pickle=False
)

test_ids = data["test_entry_ids"]

original_df = pd.read_csv(
    "dataset/processed/legal_billing_features.csv"
)

test_df = original_df[
    original_df["EntryID"].isin(test_ids)
].copy()

test_df = test_df.reset_index(drop=True)
test_df["Position"] = test_df["Position"].fillna("Unknown")

# 2. Load preprocessing objects
preprocessor = joblib.load(
    "dataset/processed/legal_model_ready/legal_preprocessor.joblib"
)

scaler = preprocessor["scaler"]
encoder = preprocessor["encoder"]

numeric_features = preprocessor["numeric_features"]
categorical_features = preprocessor["categorical_features"]

# 3. Load trained Autoencoder
model = load_model(
    "models/saved/best_legal_autoencoder.keras"
)

# 4. Load anomaly threshold
threshold_data = pd.read_csv(
    "reports/legal_anomaly_threshold.csv"
)

threshold = threshold_data["AnomalyThreshold"].iloc[0]

print("\n1. MODEL LOADED")
print("Test records:", len(test_df))
print("Threshold:", threshold)


# 5. Transform records
def transform_records(df):

    numeric_data = scaler.transform(
        df[numeric_features]
    )

    categorical_data = encoder.transform(
        df[categorical_features]
    )

    return np.concatenate(
        [numeric_data, categorical_data],
        axis=1
    ).astype("float32")


# 6. Calculate anomaly scores
def calculate_errors(df):

    X = transform_records(df)

    reconstructed = model.predict(
        X,
        verbose=0
    )

    errors = np.mean(
        np.square(X - reconstructed),
        axis=1
    )

    return errors


# Original scores
original_errors = calculate_errors(test_df)


# 7. Rate-only experiment
rate_df = test_df.copy()

rate_df["Rate"] = rate_df["Rate"] * 3

rate_errors = calculate_errors(rate_df)


# 8. Duration-only experiment
duration_df = test_df.copy()

duration_df["EntryDurationHours"] = (
    duration_df["EntryDurationHours"] * 5
)

duration_errors = calculate_errors(duration_df)


# 9. Combined experiment
combined_df = test_df.copy()

combined_df["Rate"] = combined_df["Rate"] * 3

combined_df["EntryDurationHours"] = (
    combined_df["EntryDurationHours"] * 5
)

combined_errors = calculate_errors(combined_df)


# 10. Prepare comparison
scenarios = {
    "Original": original_errors,
    "Rate Only": rate_errors,
    "Duration Only": duration_errors,
    "Combined": combined_errors
}

summary_rows = []

for name, errors in scenarios.items():

    flagged = errors > threshold

    summary_rows.append({
        "Scenario": name,
        "Average Anomaly Score": np.mean(errors),
        "Flagged Records": int(flagged.sum()),
        "Detection Rate (%)": (
            flagged.mean() * 100
        )
    })


summary = pd.DataFrame(summary_rows)

print("\n2. SCENARIO COMPARISON")
print(summary.to_string(index=False))


# 11. Save results
output_dir = Path("reports")
output_dir.mkdir(parents=True, exist_ok=True)

summary.to_csv(
    output_dir / "anomaly_scenario_comparison.csv",
    index=False
)

print("\n3. FILE SAVED")
print("reports/anomaly_scenario_comparison.csv")


# 12. Generate comparison chart
plt.figure(figsize=(9, 5))

bars = plt.bar(
    summary["Scenario"],
    summary["Detection Rate (%)"]
)

plt.title("Anomaly Detection Rate Across Different Scenarios")
plt.xlabel("Evaluation Scenario")
plt.ylabel("Detection Rate (%)")

plt.ylim(0, 110)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

# Display exact percentages above bars
for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        f"{height:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    output_dir / "anomaly_scenario_comparison.png",
    dpi=300
)

plt.show()

print("\n4. CHART SAVED")
print("reports/anomaly_scenario_comparison.png")

print("\nSCENARIO EVALUATION COMPLETED!")
