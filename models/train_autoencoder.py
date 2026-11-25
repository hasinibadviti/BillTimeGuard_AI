
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

print("=" * 65)
print("BILLTIMEGUARD AI - LEGAL AUTOENCODER TRAINING")
print("=" * 65)

# 1. Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# 2. Load legal billing data
data = np.load(
    "dataset/processed/legal_model_ready/legal_model_data.npz",
    allow_pickle=False
)

X_train = data["X_train"]
X_val = data["X_val"]

print("\n1. LEGAL TRAINING DATA")
print("Training records:", X_train.shape[0])
print("Validation records:", X_val.shape[0])
print("Input features:", X_train.shape[1])

# 3. Load legal Autoencoder
model_path = Path(
    "models/saved/legal_autoencoder_untrained.keras"
)

autoencoder = load_model(model_path)

print("\n2. LEGAL AUTOENCODER LOADED")

# 4. Create output directories
output_dir = Path("models/saved")
output_dir.mkdir(parents=True, exist_ok=True)

Path("reports").mkdir(parents=True, exist_ok=True)

Path("dataset/visualizations").mkdir(
    parents=True,
    exist_ok=True
)

# 5. Configure callbacks
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=0.00001,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath=output_dir / "best_legal_autoencoder.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

# 6. Train the Autoencoder
print("\n3. STARTING LEGAL AUTOENCODER TRAINING...")

history = autoencoder.fit(
    X_train,
    X_train,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_data=(X_val, X_val),
    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ],
    verbose=1
)

# 7. Save trained model
autoencoder.save(
    output_dir / "legal_autoencoder.keras"
)

# 8. Save training history
history_df = pd.DataFrame(history.history)

history_df.to_csv(
    "reports/legal_training_history.csv",
    index=False
)

# 9. Plot training and validation loss
plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "BillTimeGuard AI - Legal Autoencoder Training Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(True)

plt.savefig(
    "dataset/visualizations/legal_training_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# 10. Display results
print("\n4. TRAINING COMPLETED")

print(
    "Epochs completed:",
    len(history.history["loss"])
)

print(
    "Final training loss:",
    history.history["loss"][-1]
)

print(
    "Final validation loss:",
    history.history["val_loss"][-1]
)

print("\nSAVED FILES:")
print("models/saved/legal_autoencoder.keras")
print("models/saved/best_legal_autoencoder.keras")
print("reports/legal_training_history.csv")
print("dataset/visualizations/legal_training_loss.png")

print("\nLEGAL AUTOENCODER TRAINING FINISHED!")
