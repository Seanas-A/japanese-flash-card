"""Entraînement KanjiResNet — Reconnaissance de caractères japonais.

Pipeline optimisé GPU : augmentation TF-native, mixed precision, tf.data prefetch.
Architecture ResNet-18 + Attention. Supporte 96 kana ou 2232 kana+kanji.
Génère des courbes de progression (loss/accuracy) dans plots/.
"""

import json
import random
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Police japonaise multiplateforme
import platform
if platform.system() == "Darwin":
    matplotlib.rcParams["font.family"] = "Hiragino Sans"
else:
    import os
    from matplotlib import font_manager as _fm
    _jp_font = Path(__file__).parent / "data" / "fonts" / "NotoSansJP-Regular.ttf"
    if _jp_font.exists():
        _fm.fontManager.addfont(str(_jp_font))
        matplotlib.rcParams["font.family"] = _fm.FontProperties(fname=str(_jp_font)).get_name()

# Mixed precision — Tensor Cores (float16)
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# Seeds fixes
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Config
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

IMG_SIZE = 28
BATCH_SIZE = 256
EPOCHS = 300
WARMUP_EPOCHS = 5
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.2

print(f"TensorFlow {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")
print(f"Mixed precision: {mixed_precision.global_policy().name}")


# =====================================================================
# Modèle — ResNet-18 + Attention
# =====================================================================

class StochasticDepth(layers.Layer):
    """Drop aléatoire de branches résiduelles (DropPath)."""

    def __init__(self, drop_prob=0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        if not training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        return x / keep * tf.floor(keep + tf.random.uniform(shape, dtype=x.dtype))

    def get_config(self):
        return {**super().get_config(), "drop_prob": self.drop_prob}


def _res_block(x, filters, drop_prob=0.0, name="res"):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False, name=f"{name}_proj")(x)
        shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.ReLU(name=f"{name}_relu1")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    if drop_prob > 0.0:
        x = StochasticDepth(drop_prob, name=f"{name}_drop")(x)
    x = layers.Add(name=f"{name}_add")([shortcut, x])
    x = layers.ReLU(name=f"{name}_relu2")(x)
    return x


def build_model(num_classes, img_size=28, stochastic_depth_rate=0.2):
    inputs = keras.Input(shape=(img_size, img_size, 1), name="input")
    x = layers.Conv2D(64, 3, padding="same", use_bias=False, name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    total_blocks = 8
    drop_probs = [stochastic_depth_rate * i / (total_blocks - 1) for i in range(total_blocks)]
    bi = 0

    for i in range(2):
        x = _res_block(x, 64, drop_prob=drop_probs[bi], name=f"s1_b{i}"); bi += 1
    x = layers.MaxPooling2D(2, name="pool1")(x)
    for i in range(2):
        x = _res_block(x, 128, drop_prob=drop_probs[bi], name=f"s2_b{i}"); bi += 1
    x = layers.MaxPooling2D(2, name="pool2")(x)
    for i in range(2):
        x = _res_block(x, 256, drop_prob=drop_probs[bi], name=f"s3_b{i}"); bi += 1
    for i in range(2):
        x = _res_block(x, 512, drop_prob=drop_probs[bi], name=f"s4_b{i}"); bi += 1

    spatial = img_size // 4
    seq_len = spatial * spatial
    x = layers.Reshape((seq_len, 512), name="reshape_seq")(x)
    x = x + layers.Embedding(seq_len, 512, name="pos_embed")(tf.range(seq_len))
    attn = layers.MultiHeadAttention(num_heads=8, key_dim=64, dropout=0.1, name="mha")(x, x)
    x = layers.LayerNormalization(name="ln")(x + attn)

    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(1024, name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.ReLU(name="fc1_relu")(x)
    x = layers.Dropout(0.4, name="fc1_drop")(x)
    x = layers.Dense(num_classes, name="logits")(x)
    outputs = layers.Activation("softmax", dtype="float32", name="softmax")(x)

    return keras.Model(inputs, outputs, name="kanji_resnet")


# =====================================================================
# Données
# =====================================================================

dataset = np.load(DATA_DIR / "kana_dataset.npz")
images, labels = dataset["images"], dataset["labels"]

with open(DATA_DIR / "labels.json") as f:
    label_info = json.load(f)

NUM_CLASSES = len(label_info)

splits = np.load(DATA_DIR / "split_indices.npz")
train_idx, val_idx, test_idx = splits["train_idx"], splits["val_idx"], splits["test_idx"]

X = images.astype(np.float32) / 255.0
X = X[..., np.newaxis]
y = labels.astype(np.int32)

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

print(f"\n{NUM_CLASSES} classes | Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# Soft labels à la volée (pas de pré-allocation d'une matrice N×C)
CONFUSABLE_PAIRS = [(28, 76)]  # へ ↔ ヘ

_confusable_matrix = np.eye(NUM_CLASSES, dtype=np.float32)
for a, b in CONFUSABLE_PAIRS:
    if a < NUM_CLASSES and b < NUM_CLASSES:
        _confusable_matrix[a, a] = 0.5
        _confusable_matrix[a, b] = 0.5
        _confusable_matrix[b, b] = 0.5
        _confusable_matrix[b, a] = 0.5
_confusable_tf = tf.constant(_confusable_matrix)


def to_soft_label(y_scalar, label_smoothing=0.0):
    one_hot = tf.gather(_confusable_tf, y_scalar)
    if label_smoothing > 0:
        one_hot = one_hot * (1.0 - label_smoothing) + label_smoothing / NUM_CLASSES
    return one_hot


# =====================================================================
# Augmentation GPU-native
# =====================================================================

def _bilinear_sample(image_2d, sy, sx):
    y0 = tf.cast(tf.floor(sy), tf.int32)
    x0 = tf.cast(tf.floor(sx), tf.int32)
    y1, x1 = y0 + 1, x0 + 1
    y0c = tf.clip_by_value(y0, 0, IMG_SIZE - 1)
    y1c = tf.clip_by_value(y1, 0, IMG_SIZE - 1)
    x0c = tf.clip_by_value(x0, 0, IMG_SIZE - 1)
    x1c = tf.clip_by_value(x1, 0, IMG_SIZE - 1)
    wy = sy - tf.cast(y0, tf.float32)
    wx = sx - tf.cast(x0, tf.float32)
    flat = tf.reshape(image_2d, [-1])
    def g(r, c): return tf.reshape(tf.gather(flat, tf.reshape(r * IMG_SIZE + c, [-1])), [IMG_SIZE, IMG_SIZE])
    return g(y0c, x0c)*(1-wy)*(1-wx) + g(y0c, x1c)*(1-wy)*wx + g(y1c, x0c)*wy*(1-wx) + g(y1c, x1c)*wy*wx


def _elastic(img):
    alpha = tf.random.uniform([], 0.6, 1.2)
    dx = tf.image.resize(tf.random.normal([1, 14, 14, 1]), [IMG_SIZE, IMG_SIZE], method='bilinear')[0, :, :, 0] * alpha
    dy = tf.image.resize(tf.random.normal([1, 14, 14, 1]), [IMG_SIZE, IMG_SIZE], method='bilinear')[0, :, :, 0] * alpha
    gy, gx = tf.meshgrid(tf.cast(tf.range(IMG_SIZE), tf.float32),
                          tf.cast(tf.range(IMG_SIZE), tf.float32), indexing='ij')
    S = tf.cast(IMG_SIZE - 1, tf.float32)
    return _bilinear_sample(img[:, :, 0],
                            tf.clip_by_value(gy + dy, 0.0, S),
                            tf.clip_by_value(gx + dx, 0.0, S))[:, :, tf.newaxis]


def _random_erase(img):
    h = tf.random.uniform([], 4, 12, dtype=tf.int32)
    w = tf.random.uniform([], 4, 12, dtype=tf.int32)
    top = tf.random.uniform([], 0, IMG_SIZE - h, dtype=tf.int32)
    left = tf.random.uniform([], 0, IMG_SIZE - w, dtype=tf.int32)
    hole = tf.pad(tf.zeros([h, w, 1], dtype=img.dtype),
                  [[top, IMG_SIZE - top - h], [left, IMG_SIZE - left - w], [0, 0]],
                  constant_values=1.0)
    return img * hole


def augment(image):
    img = tf.cast(image, tf.float32)
    img = tf.cond(tf.random.uniform([]) < 0.9, lambda: _elastic(img), lambda: img)

    angle = tf.random.uniform([], -25.0, 25.0) * (3.14159265 / 180.0)
    scale = tf.random.uniform([], 0.4, 1.6)
    ms = tf.minimum(0.30 / scale, 0.40)
    tx = tf.random.uniform([], -ms, ms) * tf.cast(IMG_SIZE, tf.float32)
    ty = tf.random.uniform([], -ms, ms) * tf.cast(IMG_SIZE, tf.float32)
    cos_a, sin_a = tf.cos(angle) / scale, tf.sin(angle) / scale
    c = (tf.cast(IMG_SIZE, tf.float32) - 1.0) / 2.0
    transform = tf.stack([cos_a, sin_a, c*(1-cos_a-sin_a) - tx/scale,
                          -sin_a, cos_a, c*(1+sin_a-cos_a) - ty/scale, 0.0, 0.0])
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[tf.newaxis], transforms=transform[tf.newaxis],
        output_shape=tf.constant([IMG_SIZE, IMG_SIZE]),
        fill_value=0.0, interpolation="BILINEAR", fill_mode="CONSTANT")[0]

    img = tf.cond(tf.random.uniform([]) < 0.3, lambda: _random_erase(img), lambda: img)
    img = tf.cond(tf.random.uniform([]) < 0.4,
                  lambda: tf.clip_by_value(img + tf.random.normal([IMG_SIZE, IMG_SIZE, 1],
                                           stddev=tf.random.uniform([], 0.01, 0.04)), 0.0, 1.0),
                  lambda: img)
    return img


def mixup(images, labels, alpha=MIXUP_ALPHA):
    bs = tf.shape(images)[0]
    # Gamma en float32 pour éviter les NaN en mixed precision
    lam = tf.random.gamma([bs, 1, 1, 1], alpha, beta=1.0 / alpha, dtype=tf.float32)
    lam2 = tf.random.gamma([bs, 1, 1, 1], alpha, beta=1.0 / alpha, dtype=tf.float32)
    lam = lam / tf.maximum(lam + lam2, 1e-8)  # éviter 0/0
    idx = tf.random.shuffle(tf.range(bs))
    images = tf.cast(images, tf.float32)
    lam_l = tf.reshape(lam, [bs, 1])
    mixed_images = images * lam + tf.gather(images, idx) * (1.0 - lam)
    mixed_labels = labels * lam_l + tf.gather(labels, idx) * (1.0 - lam_l)
    return mixed_images, mixed_labels


# =====================================================================
# Construction et compilation
# =====================================================================

model = build_model(num_classes=NUM_CLASSES)
model.summary()

steps_per_epoch = len(X_train) // BATCH_SIZE
total_steps = steps_per_epoch * EPOCHS
warmup_steps = steps_per_epoch * WARMUP_EPOCHS

lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-6, decay_steps=total_steps - warmup_steps,
    alpha=1e-6, warmup_target=0.001, warmup_steps=warmup_steps,
)

model.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=lr_schedule, weight_decay=0.01, clipnorm=1.0),
    loss="categorical_crossentropy", metrics=["accuracy"],
)


# =====================================================================
# Callback : courbes de progression en temps réel
# =====================================================================

class PlotProgress(keras.callbacks.Callback):
    """Sauvegarde les courbes loss/accuracy toutes les N epochs."""

    def __init__(self, every_n=5):
        self.every_n = every_n
        self.logs_history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

    def on_epoch_end(self, epoch, logs=None):
        for k in self.logs_history:
            self.logs_history[k].append(logs.get(k, 0))

        if (epoch + 1) % self.every_n != 0 and epoch != 0:
            return

        h = self.logs_history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(h["loss"], label="Train", color="#e94560")
        ax1.plot(h["val_loss"], label="Val", color="#4ea8de")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.plot(h["accuracy"], label="Train", color="#e94560")
        ax2.plot(h["val_accuracy"], label="Val", color="#4ea8de")
        ax2.set_title("Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.legend()
        ax2.grid(alpha=0.3)

        fig.suptitle(f"Epoch {epoch + 1} — val_acc: {logs.get('val_accuracy', 0):.4f}", fontsize=13)
        fig.tight_layout()
        fig.savefig(str(PLOTS_DIR / "training_progress.png"), dpi=100)
        plt.close(fig)


callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=30, restore_best_weights=True, verbose=1),
    keras.callbacks.ModelCheckpoint(
        str(MODELS_DIR / "best_model.keras"),
        monitor="val_accuracy", save_best_only=True, verbose=1),
    PlotProgress(every_n=5),
]

# =====================================================================
# Entraînement
# =====================================================================

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = (train_ds
    .shuffle(len(X_train), reshuffle_each_iteration=True)
    .map(lambda x, y: (augment(x), to_soft_label(y, LABEL_SMOOTHING)),
         num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .map(lambda x, y: mixup(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE))

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = (val_ds
    .map(lambda x, y: (x, to_soft_label(y)), num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE))

history = model.fit(train_ds, validation_data=val_ds,
                    epochs=EPOCHS, callbacks=callbacks, verbose=1)

# =====================================================================
# Évaluation
# =====================================================================

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = (test_ds
    .map(lambda x, y: (x, to_soft_label(y)), num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"\nTest loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

y_pred_proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

top5 = np.mean([y_test[i] in np.argsort(y_pred_proba[i])[-5:] for i in range(len(y_test))])
print(f"Top-5 accuracy: {top5:.4f}")

hira_mask = y_test < 48
kata_mask = (y_test >= 48) & (y_test < 96)
print(f"Hiragana: {np.mean(y_pred[hira_mask] == y_test[hira_mask]):.4f}")
print(f"Katakana: {np.mean(y_pred[kata_mask] == y_test[kata_mask]):.4f}")
if NUM_CLASSES > 96:
    kanji_mask = y_test >= 96
    if kanji_mask.any():
        print(f"Kanji:    {np.mean(y_pred[kanji_mask] == y_test[kanji_mask]):.4f}")

# Courbes finales
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history.history["loss"], label="Train", color="#e94560")
ax1.plot(history.history["val_loss"], label="Val", color="#4ea8de")
ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(alpha=0.3)
ax2.plot(history.history["accuracy"], label="Train", color="#e94560")
ax2.plot(history.history["val_accuracy"], label="Val", color="#4ea8de")
ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(alpha=0.3)
fig.suptitle(f"Final — Test acc: {test_acc:.4f} | Top-5: {top5:.4f}", fontsize=13)
fig.tight_layout()
fig.savefig(str(PLOTS_DIR / "training_final.png"), dpi=150)
plt.close(fig)
print(f"\nCourbes → {PLOTS_DIR}/training_final.png")

model.save(str(MODELS_DIR / "final_model.keras"))
print(f"Modèles → {MODELS_DIR}/")
