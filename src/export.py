"""Export the trained Keras model to ONNX for fast, TensorFlow-free inference.

    python -m src.export          # writes models/mask_detector.onnx, checks parity, benchmarks

ONNX Runtime loads in ~1 s using a fraction of TensorFlow's memory, and runs MobileNetV2
several times faster on CPU. The detector picks the ONNX model automatically when present.
"""
import time

import numpy as np
import tensorflow as tf
import tf2onnx

from config import DATA_DIR, IMG_SIZE, MASK_MODEL_PATH, MASK_ONNX_PATH


def export() -> None:
    model = tf.keras.models.load_model(MASK_MODEL_PATH)

    @tf.function(input_signature=[tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32, name="image")])
    def infer(x):
        return model(x, training=False)  # augmentation layers are identity at inference

    # Skip tf2onnx's graph optimiser: it needs several GB of RAM on this model, and
    # ONNX Runtime applies equivalent optimisations when it loads the model anyway.
    from tf2onnx import optimizer
    original = optimizer.optimize_graph
    optimizer.optimize_graph = lambda graph, *args, **kwargs: graph
    try:
        tf2onnx.convert.from_function(infer, input_signature=infer.input_signature, opset=17,
                                      output_path=str(MASK_ONNX_PATH))
    finally:
        optimizer.optimize_graph = original
    print(f"Saved {MASK_ONNX_PATH} ({MASK_ONNX_PATH.stat().st_size / 1e6:.1f} MB)")
    verify(model)


def _sample_batch(n: int = 32) -> np.ndarray:
    import cv2
    files = sorted(p for p in DATA_DIR.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    picks = files[:: max(1, len(files) // n)][:n]
    return np.stack([cv2.resize(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE))
                     for p in picks]).astype("float32")


def verify(model) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(MASK_ONNX_PATH), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    batch = _sample_batch()
    chunks = range(0, len(batch), 4)  # small chunks keep peak RAM low
    keras_out = np.concatenate([model(batch[i:i + 4], training=False).numpy() for i in chunks])
    onnx_out = np.concatenate([sess.run(None, {name: batch[i:i + 4]})[0] for i in chunks])
    agree = (keras_out.argmax(1) == onnx_out.argmax(1)).mean()
    print(f"Parity on {len(batch)} images: max prob diff {np.abs(keras_out - onnx_out).max():.2e}, "
          f"label agreement {agree:.0%}")

    def bench(fn, x, runs=30):
        fn(x)  # warm-up
        t = time.perf_counter()
        for _ in range(runs):
            fn(x)
        return 1000 * (time.perf_counter() - t) / runs

    for bs in (1, 4):
        x = batch[:bs]
        k = bench(lambda v: model(v, training=False).numpy(), x)
        o = bench(lambda v: sess.run(None, {name: v}), x)
        print(f"batch {bs}: Keras {k:.1f} ms  |  ONNX Runtime {o:.1f} ms  ({k / o:.1f}x faster)")


if __name__ == "__main__":
    import sys
    if "--verify-only" in sys.argv and MASK_ONNX_PATH.exists():
        verify(tf.keras.models.load_model(MASK_MODEL_PATH))
    else:
        export()
