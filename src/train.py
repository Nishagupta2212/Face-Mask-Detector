"""Train the mask classifier.

Usage:
    python -m src.train                       # MobileNetV2 transfer learning + fine-tuning
    python -m src.train --arch cnn --epochs 30
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402

from config import (  # noqa: E402
    BATCH_SIZE, DATA_DIR, EPOCHS, IMG_SIZE, LABELS_PATH, LEARNING_RATE,
    MASK_MODEL_PATH, REPORTS_DIR,
)
from src.model import build_custom_cnn, build_mobilenet, unfreeze_top  # noqa: E402

SEED = 42


def load_datasets(val_split: float):
    common = dict(
        directory=str(DATA_DIR),
        validation_split=val_split,
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(subset="training", **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(subset="validation", **common)
    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    return (
        train_ds.cache().shuffle(1000, seed=SEED).prefetch(autotune),
        val_ds.cache().prefetch(autotune),
        class_names,
    )


def class_weights(class_names: list[str]) -> dict[int, float]:
    counts = np.array([
        sum(1 for p in (DATA_DIR / c).iterdir() if p.is_file()) for c in class_names
    ], dtype=float)
    weights = counts.sum() / (len(counts) * counts)
    return {i: float(w) for i, w in enumerate(weights)}


def callbacks(best_val_acc: float | None = None):
    # best_val_acc keeps the fine-tuning phase from overwriting a better checkpoint.
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(str(MASK_MODEL_PATH), monitor="val_accuracy", save_best_only=True,
                                           initial_value_threshold=best_val_acc),
    ]


def merge_history(*histories) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for h in histories:
        for k, v in h.history.items():
            merged.setdefault(k, []).extend(v)
    return merged


def save_plots(history: dict, cm: np.ndarray, class_names: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ("accuracy", "loss")):
        ax.plot(history[metric], label=f"train {metric}")
        ax.plot(history[f"val_{metric}"], label=f"val {metric}")
        ax.set_xlabel("epoch")
        ax.set_title(metric.capitalize())
        ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "training_curves.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=30)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Confusion matrix (validation)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the face mask classifier")
    parser.add_argument("--arch", choices=["mobilenet", "cnn"], default="mobilenet")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--fine-tune-epochs", type=int, default=5,
                        help="extra epochs with the top of MobileNetV2 unfrozen (0 to skip)")
    parser.add_argument("--val-split", type=float, default=0.2)
    args = parser.parse_args()

    if not DATA_DIR.exists():
        raise SystemExit(f"No dataset at {DATA_DIR}. Run: python scripts/download_dataset.py")

    train_ds, val_ds, class_names = load_datasets(args.val_split)
    print(f"Classes: {class_names}")
    weights = class_weights(class_names)
    print(f"Class weights: {weights}")

    if args.arch == "mobilenet":
        model, base = build_mobilenet(len(class_names))
    else:
        model, base = build_custom_cnn(len(class_names)), None

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE if base else 1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    histories = [model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                           class_weight=weights, callbacks=callbacks())]

    if base is not None and args.fine_tune_epochs > 0:
        print("\nFine-tuning top layers of MobileNetV2...")
        unfreeze_top(base)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(LEARNING_RATE / 10),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        histories.append(model.fit(train_ds, validation_data=val_ds,
                                   epochs=args.fine_tune_epochs, class_weight=weights,
                                   callbacks=callbacks(max(histories[0].history["val_accuracy"]))))

    # Evaluate the best checkpoint on the validation set.
    best = tf.keras.models.load_model(MASK_MODEL_PATH)
    y_true = np.concatenate([y.numpy() for _, y in val_ds])
    y_pred = np.argmax(best.predict(val_ds, verbose=0), axis=1)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    cm = confusion_matrix(y_true, y_pred)

    LABELS_PATH.write_text(json.dumps(class_names, indent=2))
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(
        {"arch": args.arch, "classes": class_names, "report": report,
         "confusion_matrix": cm.tolist()}, indent=2))
    save_plots(merge_history(*histories), cm, class_names)
    print(f"\nSaved model -> {MASK_MODEL_PATH}")
    print(f"Saved plots & metrics -> {REPORTS_DIR}")
    try:  # fast ONNX copy for the real-time apps
        from src.export import export
        export()
    except Exception as exc:  # optional: the apps fall back to the Keras model
        print(f"ONNX export skipped ({exc}). Run `python -m src.export` to retry.")


if __name__ == "__main__":
    main()
