"""CNN architectures for mask classification.

Both models take raw RGB images (0-255, IMG_SIZE x IMG_SIZE) - preprocessing and
augmentation live inside the model so inference code stays simple.
"""
import tensorflow as tf
from tensorflow.keras import layers, models

from config import IMG_SIZE


def _augmentation() -> tf.keras.Sequential:
    # Only active during training; identity at inference.
    return models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.15),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomContrast(0.2),
            layers.RandomBrightness(0.2, value_range=(0, 255)),
        ],
        name="augmentation",
    )


def build_mobilenet(num_classes: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Transfer learning on MobileNetV2 (ImageNet). Returns (model, base)."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = _augmentation()(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)  # MobileNetV2 expects [-1, 1]
    x = base(x, training=False)                         # keep BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="mask_mobilenetv2"), base


def build_custom_cnn(num_classes: int) -> tf.keras.Model:
    """A small CNN trained from scratch - useful for comparison / learning."""
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = _augmentation()(inputs)
    x = layers.Rescaling(1.0 / 255)(x)
    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="mask_custom_cnn")


def unfreeze_top(base: tf.keras.Model, n_layers: int = 40) -> None:
    """Unfreeze the last n layers of the backbone for fine-tuning (BatchNorm stays frozen)."""
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
