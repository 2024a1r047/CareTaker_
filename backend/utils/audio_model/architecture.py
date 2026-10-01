"""
High-Performance CRNN for real-time audio classification.

Architecture: CNN feature extractor → Bidirectional GRU → Dense head.
This captures both spectral patterns (CNN) and temporal dynamics (GRU),
which is critical for distinguishing impulsive sounds (cough/sneeze)
from continuous sounds (talking/noise).

Input shape : (batch, 64, 101, 1)   — 64 mel bands × ~101 time frames
Output shape: (batch, 4)            — softmax over [Cough, Sneeze, Talking, Noise]
"""

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    HAS_TF = False
    tf = None
    keras = None
    layers = None


def _conv_block(x, filters, kernel=(3, 3), pool=(2, 2)):
    """Depthwise-separable Conv2D → BatchNorm → ReLU → MaxPool."""
    # Depthwise separable conv is 6-9× faster than standard Conv2D
    x = layers.SeparableConv2D(filters, kernel, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool)(x)
    return x


def _residual_block(x, filters):
    """Residual block with skip connection for better gradient flow."""
    shortcut = layers.Conv2D(filters, (1, 1), padding="same", use_bias=False)(x)
    shortcut = layers.BatchNormalization()(shortcut)

    x = layers.SeparableConv2D(filters, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.SeparableConv2D(filters, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


def build_model(input_shape=(64, 101, 1), num_classes=4):
    if not HAS_TF:
        raise RuntimeError("TensorFlow is not installed. CRNN model is disabled.")
    inp = layers.Input(shape=input_shape, name="mel_input")

    # ── CNN Feature Extractor ───────────────────────────────────────
    x = _conv_block(inp, 32)             # → (32, 50, 32)
    x = _residual_block(x, 64)          # → (32, 50, 64)
    x = layers.MaxPooling2D((2, 2))(x)  # → (16, 25, 64)
    x = _residual_block(x, 128)         # → (16, 25, 128)
    x = layers.MaxPooling2D((2, 2))(x)  # → (8,  12, 128)
    x = _conv_block(x, 128)             # → (4,   6, 128)

    # ── Reshape for RNN: (time_steps, features) ────────────────────
    # Collapse frequency axis, keep time axis for sequential modelling
    shape = x.shape  # (batch, freq, time, channels)
    x = layers.Reshape((shape[1] * shape[3], shape[2]))(x)  # (batch, freq*ch, time)
    x = layers.Permute((2, 1))(x)  # (batch, time, features)

    # ── Temporal Modelling ──────────────────────────────────────────
    x = layers.Bidirectional(
        layers.GRU(64, return_sequences=False, dropout=0.2)
    )(x)

    # ── Classifier Head ────────────────────────────────────────────
    x = layers.Dense(96, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inp, outputs=out, name="SonicGuard_CRNN")
    return model


# ── Quick sanity-check ──────────────────────────────────────────────
if __name__ == "__main__":
    m = build_model()
    m.summary()
    # Test inference speed
    import numpy as np, time
    dummy = np.random.randn(1, 64, 101, 1).astype("float32")
    # Warm-up
    m(dummy, training=False)
    start = time.perf_counter()
    for _ in range(100):
        m(dummy, training=False)
    elapsed = (time.perf_counter() - start) / 100 * 1000
    print(f"\n⚡ Average inference: {elapsed:.1f} ms")
