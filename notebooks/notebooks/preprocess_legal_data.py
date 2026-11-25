
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder

print("=" * 65)
print("BILLTIMEGUARD AI - LEGAL DATA PREPROCESSING")
print("=" * 65)

# 1. Load legal billing data
df = pd.read_csv(
    "dataset/processed/legal_billing_features.csv"
)

print("\nOriginal dataset:", df.shape)

# 2. Fill missing position values
df["Position"] = df["Position"].fillna("Unknown")

# 3. Select model features
numeric_features = [
    "EntryDurationHours",
    "Rate",
    "DayOfWeek",
    "Month",
    "Partner"
]

categorical_features = ["Position"]

# 4. Create groups to reduce repeated-entry leakage
group_columns = [
    "Timekeeper",
    "Date",
    "Matter",
    "EntryDurationHours",
    "Rate"
]

df["GroupID"] = (
    df.groupby(group_columns, dropna=False)
      .ngroup()
)

# 5. Split into training, validation and testing sets
split1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42
)

train_idx, temp_idx = next(
    split1.split(df, groups=df["GroupID"])
)

train_df = df.iloc[train_idx].copy()
temp_df = df.iloc[temp_idx].copy()

split2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

val_idx, test_idx = next(
    split2.split(temp_df, groups=temp_df["GroupID"])
)

val_df = temp_df.iloc[val_idx].copy()
test_df = temp_df.iloc[test_idx].copy()

print("\n2. DATA SPLIT")
print("Training:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(test_df))

# 6. Prepare numerical features
scaler = StandardScaler()

X_train_num = scaler.fit_transform(
    train_df[numeric_features]
)

X_val_num = scaler.transform(
    val_df[numeric_features]
)

X_test_num = scaler.transform(
    test_df[numeric_features]
)

# 7. Encode categorical features
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

X_train_cat = encoder.fit_transform(
    train_df[categorical_features]
)

X_val_cat = encoder.transform(
    val_df[categorical_features]
)

X_test_cat = encoder.transform(
    test_df[categorical_features]
)

# 8. Combine numerical and categorical features
X_train = np.hstack([X_train_num, X_train_cat]).astype("float32")
X_val = np.hstack([X_val_num, X_val_cat]).astype("float32")
X_test = np.hstack([X_test_num, X_test_cat]).astype("float32")

# 9. Generate feature names
feature_names = np.concatenate([
    np.array(numeric_features),
    encoder.get_feature_names_out(categorical_features)
])

# 10. Check data
for name, array in [
    ("Training", X_train),
    ("Validation", X_val),
    ("Testing", X_test)
]:
    if not np.isfinite(array).all():
        raise ValueError(f"Invalid values in {name} data")

print("\n3. FINAL DATA DIMENSIONS")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)
print("Features:", len(feature_names))

# 11. Save data
output_dir = Path("dataset/processed/legal_model_ready")
output_dir.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    output_dir / "legal_model_data.npz",
    X_train=X_train,
    X_val=X_val,
    X_test=X_test,
    feature_names=feature_names,
    train_entry_ids=train_df["EntryID"].to_numpy(),
    val_entry_ids=val_df["EntryID"].to_numpy(),
    test_entry_ids=test_df["EntryID"].to_numpy()
)

joblib.dump(
    {
        "scaler": scaler,
        "encoder": encoder,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_names": feature_names
    },
    output_dir / "legal_preprocessor.joblib"
)

print("\n4. SAVED SUCCESSFULLY")
print(output_dir / "legal_model_data.npz")
print(output_dir / "legal_preprocessor.joblib")

print("\nLEGAL DATA PREPROCESSING COMPLETED!")
