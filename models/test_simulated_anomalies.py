
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from tensorflow.keras.models import load_model

print("=" * 70)
print("BILLTIMEGUARD AI - CONTROLLED ANOMALY TESTING")
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

# 2. Load saved preprocessing objects
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

# 4. Load existing anomaly threshold
threshold_data = pd.read_csv(
    "reports/legal_anomaly_threshold.csv"
)

threshold = threshold_data["AnomalyThreshold"].iloc[0]

print("\n1. DATA AND MODEL LOADED")
print("Test records:", len(test_df))
print("Anomaly threshold:", threshold)

# 5. Define transformation function
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


# 6. Calculate original reconstruction errors
X_original = transform_records(test_df)

original_reconstructed = model.predict(
    X_original,
    verbose=0
)

original_errors = np.mean(
    np.square(X_original - original_reconstructed),
    axis=1
)

# 7. Create controlled modified copies
simulated_df = test_df.copy()

# Scenario A: Unusually high hourly rate
simulated_df["Rate"] = (
    simulated_df["Rate"] * 3
)

# Scenario B: Unusually long billing duration
simulated_df["EntryDurationHours"] = (
    simulated_df["EntryDurationHours"] * 5
)

# 8. Calculate simulated reconstruction errors
X_simulated = transform_records(simulated_df)

simulated_reconstructed = model.predict(
    X_simulated,
    verbose=0
)

simulated_errors = np.mean(
    np.square(
        X_simulated - simulated_reconstructed
    ),
    axis=1
)

# 9. Compare original and modified records
original_flagged = original_errors > threshold
simulated_flagged = simulated_errors > threshold

results = pd.DataFrame({
    "EntryID": test_df["EntryID"],
    "OriginalDuration": test_df["EntryDurationHours"],
    "OriginalRate": test_df["Rate"],
    "SimulatedDuration": simulated_df["EntryDurationHours"],
    "SimulatedRate": simulated_df["Rate"],
    "OriginalAnomalyScore": original_errors,
    "SimulatedAnomalyScore": simulated_errors,
    "OriginalFlagged": original_flagged,
    "SimulatedFlagged": simulated_flagged
})

results["ScoreIncreased"] = (
    results["SimulatedAnomalyScore"]
    > results["OriginalAnomalyScore"]
)

# 10. Print evaluation summary
print("\n2. CONTROLLED ANOMALY TEST RESULTS")

print("Total modified records:", len(results))

print(
    "Original flagged records:",
    int(original_flagged.sum())
)

print(
    "Modified flagged records:",
    int(simulated_flagged.sum())
)

print(
    "Records with increased anomaly scores:",
    int(results["ScoreIncreased"].sum())
)

print(
    "Average original reconstruction error:",
    round(float(np.mean(original_errors)), 7)
)

print(
    "Average simulated reconstruction error:",
    round(float(np.mean(simulated_errors)), 7)
)

print(
    "Average score increase:",
    round(float(np.mean(simulated_errors - original_errors)), 7)
)

# 11. Save results
output_dir = Path("reports")
output_dir.mkdir(parents=True, exist_ok=True)

results.to_csv(
    output_dir / "simulated_anomaly_test_results.csv",
    index=False
)

summary = pd.DataFrame({
    "Metric": [
        "Total modified records",
        "Original flagged records",
        "Modified flagged records",
        "Records with increased scores",
        "Average original error",
        "Average simulated error",
        "Average score increase"
    ],
    "Value": [
        len(results),
        int(original_flagged.sum()),
        int(simulated_flagged.sum()),
        int(results["ScoreIncreased"].sum()),
        float(np.mean(original_errors)),
        float(np.mean(simulated_errors)),
        float(np.mean(simulated_errors - original_errors))
    ]
})

summary.to_csv(
    output_dir / "simulated_anomaly_test_summary.csv",
    index=False
)

print("\n3. FILES SAVED")
print("reports/simulated_anomaly_test_results.csv")
print("reports/simulated_anomaly_test_summary.csv")

print("\nCONTROLLED ANOMALY TESTING COMPLETED!")
