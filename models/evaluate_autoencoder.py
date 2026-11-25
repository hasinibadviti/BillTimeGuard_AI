
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow.keras.models import load_model

print("=" * 70)
print("BILLTIMEGUARD AI - LEGAL BILLING ANOMALY SCORING")
print("=" * 70)

# 1. Load legal billing data
data = np.load(
    "dataset/processed/legal_model_ready/legal_model_data.npz",
    allow_pickle=False
)

X_val = data["X_val"]
X_test = data["X_test"]

val_ids = data["val_entry_ids"]
test_ids = data["test_entry_ids"]

print("\n1. LEGAL DATA LOADED")
print("Validation records:", len(X_val))
print("Test records:", len(X_test))

# 2. Load trained legal Autoencoder
model = load_model(
    "models/saved/best_legal_autoencoder.keras"
)

print("\n2. TRAINED MODEL LOADED")

# 3. Calculate reconstruction errors
print("\n3. CALCULATING RECONSTRUCTION ERRORS")

val_reconstructed = model.predict(X_val, verbose=0)
test_reconstructed = model.predict(X_test, verbose=0)

val_errors = np.mean(
    np.square(X_val - val_reconstructed),
    axis=1
)

test_errors = np.mean(
    np.square(X_test - test_reconstructed),
    axis=1
)

print("Validation errors calculated:", len(val_errors))
print("Test errors calculated:", len(test_errors))

# 4. Select threshold using validation data
print("\n4. SELECTING ANOMALY THRESHOLD")

selected_percentile = 95

threshold = np.percentile(
    val_errors,
    selected_percentile
)

print("Selected percentile:", selected_percentile)
print("Anomaly threshold:", threshold)

# 5. Detect potential anomalies in test data
print("\n5. TEST ANOMALY DETECTION")

test_predictions = (
    test_errors > threshold
).astype(int)

print("Total test entries:", len(test_predictions))
print("Potentially anomalous:", np.sum(test_predictions == 1))
print("Potentially normal:", np.sum(test_predictions == 0))

# 6. Load original legal billing details
original_df = pd.read_csv(
    "dataset/processed/legal_billing_features.csv"
)

test_results = pd.DataFrame({
    "EntryID": test_ids,
    "AnomalyScore": test_errors,
    "PredictedAnomaly": test_predictions
})

# Attach original details for investigation
test_results = test_results.merge(
    original_df,
    on="EntryID",
    how="left",
    validate="one_to_one"
)

test_results["AnomalyStatus"] = np.where(
    test_results["PredictedAnomaly"] == 1,
    "Potential Anomaly",
    "Normal Range"
)

# 7. Sort entries by anomaly score
test_results = test_results.sort_values(
    by="AnomalyScore",
    ascending=False
)

# 8. Save results
output_dir = Path("reports")
output_dir.mkdir(parents=True, exist_ok=True)

Path("dataset/visualizations").mkdir(
    parents=True,
    exist_ok=True
)

test_results.to_csv(
    output_dir / "legal_anomaly_results.csv",
    index=False
)

pd.DataFrame({
    "SelectedPercentile": [selected_percentile],
    "AnomalyThreshold": [threshold],
    "ValidationRecords": [len(val_errors)],
    "TestRecords": [len(test_errors)],
    "PotentialTestAnomalies": [
        int(np.sum(test_predictions == 1))
    ]
}).to_csv(
    output_dir / "legal_anomaly_threshold.csv",
    index=False
)

# 9. Plot validation and test reconstruction errors
plt.figure(figsize=(10, 5))

plt.hist(
    val_errors,
    bins=40,
    alpha=0.6,
    label="Validation Errors"
)

plt.hist(
    test_errors,
    bins=40,
    alpha=0.6,
    label="Test Errors"
)

plt.axvline(
    threshold,
    linestyle="--",
    label="95th Percentile Threshold"
)

plt.title("Legal Billing Reconstruction Error Distribution")
plt.xlabel("Reconstruction Error")
plt.ylabel("Number of Entries")
plt.legend()
plt.grid(True)

plt.savefig(
    "dataset/visualizations/legal_anomaly_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# 10. Display highest-scoring entries
print("\n6. TOP 10 POTENTIAL ANOMALIES")

print(
    test_results[
        [
            "EntryID",
            "Timekeeper",
            "Matter",
            "EntryDurationHours",
            "Rate",
            "BilledAmount",
            "AnomalyScore",
            "AnomalyStatus"
        ]
    ].head(10).to_string(index=False)
)

print("\n7. FILES SAVED")
print("reports/legal_anomaly_results.csv")
print("reports/legal_anomaly_threshold.csv")
print("dataset/visualizations/legal_anomaly_distribution.png")

print("\nLEGAL ANOMALY SCORING COMPLETED!")
