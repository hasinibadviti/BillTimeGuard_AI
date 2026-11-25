
import numpy as np
import tensorflow as tf

from pathlib import Path
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout

print("=" * 60)
print("BILLTIMEGUARD AI - LEGAL BILLING AUTOENCODER")
print("=" * 60)

# 1. Load legal billing data
data_path = Path(
    "dataset/processed/legal_model_ready/legal_model_data.npz"
)

data = np.load(data_path, allow_pickle=False)

X_train = data["X_train"]
X_val = data["X_val"]

print("\n1. LEGAL DATA LOADED")
print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)

# 2. Define input dimensions
input_dim = X_train.shape[1]

print("\n2. MODEL INPUT FEATURES:", input_dim)

# 3. Build Autoencoder architecture
input_layer = Input(
    shape=(input_dim,),
    name="legal_billing_input"
)

# Encoder
encoded = Dense(128, activation="relu")(input_layer)
encoded = Dropout(0.1)(encoded)

encoded = Dense(64, activation="relu")(encoded)
encoded = Dense(32, activation="relu", name="bottleneck")(encoded)

# Decoder
decoded = Dense(64, activation="relu")(encoded)
decoded = Dense(128, activation="relu")(decoded)

output_layer = Dense(
    input_dim,
    activation="linear",
    name="reconstructed_output"
)(decoded)

# 4. Create model
autoencoder = Model(
    inputs=input_layer,
    outputs=output_layer,
    name="BillTimeGuard_Legal_Autoencoder"
)

# 5. Compile model
autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse"
)

# 6. Display architecture
print("\n3. AUTOENCODER ARCHITECTURE")
autoencoder.summary()

# 7. Save model
model_dir = Path("models/saved")
model_dir.mkdir(parents=True, exist_ok=True)

autoencoder.save(
    model_dir / "legal_autoencoder_untrained.keras"
)

print("\n4. LEGAL AUTOENCODER CREATED SUCCESSFULLY!")
print("Input features:", input_dim)
print("Bottleneck dimensions:", 32)
print("Saved at: models/saved/legal_autoencoder_untrained.keras")
