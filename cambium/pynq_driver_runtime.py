"""Runtime copied into each generated axi_rfr_driver.py with model constants."""

import time

import numpy as np


MODEL_CONFIG = {}
N_FEATURES = MODEL_CONFIG["n_features"]
N_TARGETS = MODEL_CONFIG["n_targets"]
FIXED_POINT_BITS = MODEL_CONFIG["width"]
FEATURE_INT_BITS = MODEL_CONFIG["int_bits"]
SCORE_INT_BITS = FEATURE_INT_BITS
IS_FLOAT = MODEL_CONFIG["float"]


def encode_fixed_point(values):
    """Pack signed ap_fixed<W,I> values into the low W bits of AXI words."""
    values = np.asarray(values, dtype=np.float64)
    if IS_FLOAT:
        return np.asarray(values, dtype=np.float32).view(np.uint32)
    scale = 1 << (FIXED_POINT_BITS - FEATURE_INT_BITS)
    codes = np.floor(values * scale)
    low = -(1 << (FIXED_POINT_BITS - 1))
    high = (1 << (FIXED_POINT_BITS - 1)) - 1
    if not np.all(np.isfinite(codes)) or np.any(codes < low) or np.any(codes > high):
        raise ValueError("Input cannot be represented by the model's fixed-point format")
    return (codes.astype(np.int64) & ((1 << FIXED_POINT_BITS) - 1)).astype(np.uint32)


def decode_fixed_point(words):
    """Sign-extend the low W bits and restore their fractional scale."""
    words = np.asarray(words, dtype=np.uint32)
    if IS_FLOAT:
        return words.view(np.float32).astype(np.float64)
    mask = (1 << FIXED_POINT_BITS) - 1
    sign = 1 << (FIXED_POINT_BITS - 1)
    code = words.astype(np.int64) & mask
    signed = (code ^ sign) - sign
    return signed.astype(np.float64) / (1 << (FIXED_POINT_BITS - SCORE_INT_BITS))


def _scale_x(x):
    spec = MODEL_CONFIG["x_scaler"]
    if spec["kind"] == "identity":
        return x
    if spec["kind"] == "minmax":
        return x * np.asarray(spec["scale"]) + np.asarray(spec["offset"])
    if spec["kind"] == "standard":
        return (x - np.asarray(spec["center"])) / np.asarray(spec["scale"])
    if spec["kind"] == "robust":
        return (x - np.asarray(spec["center"])) / np.asarray(spec["scale"])
    raise ValueError("Unsupported feature scaler")


def _unscale_y(y):
    spec = MODEL_CONFIG["y_scaler"]
    if spec is None:
        return y
    return y * np.asarray(spec["range"]) + np.asarray(spec["min"])


class rfrOverlay:
    """Simple-DMA Cambium overlay; one stream word per feature/output."""

    def __init__(self, bitfile, x_shape=None, y_shape=None):
        from pynq import Overlay

        self.overlay = Overlay(bitfile)
        self.dma = self.overlay.axi_dma_0
        self.n_features = N_FEATURES
        self.n_targets = N_TARGETS
        if x_shape is not None and len(x_shape) > 1 and x_shape[1] != N_FEATURES:
            raise ValueError(f"Bitstream expects {N_FEATURES} features, got {x_shape[1]}")
        if y_shape is not None and len(y_shape) > 1 and y_shape[1] != N_TARGETS:
            raise ValueError(f"Bitstream emits {N_TARGETS} outputs, got {y_shape[1]}")
        self.input_buffer = None
        self.output_buffer = None

    def _buffer(self, current, length):
        from pynq import allocate

        if current is not None and current.shape != (length,):
            current.freebuffer()
            current = None
        return current if current is not None else allocate(shape=(length,), dtype=np.uint32)

    def predict(self, X, profile=False, scaled=False, denorm=True):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != N_FEATURES or X.shape[0] == 0:
            raise ValueError(f"Expected a nonempty array with {N_FEATURES} features")
        n_samples = X.shape[0]
        values = X if scaled else _scale_x(X)
        encoded = encode_fixed_point(values).reshape(-1)
        self.input_buffer = self._buffer(self.input_buffer, n_samples * N_FEATURES)
        self.output_buffer = self._buffer(self.output_buffer, n_samples * N_TARGETS)
        self.input_buffer[:] = encoded
        self.input_buffer.flush()
        start = time.perf_counter()
        # Arm the receiver before the sender so a fast accelerator cannot stall it.
        self.dma.recvchannel.transfer(self.output_buffer)
        self.dma.sendchannel.transfer(self.input_buffer)
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()
        elapsed = time.perf_counter() - start
        self.output_buffer.invalidate()
        if profile:
            print(f"DMA: {n_samples} samples in {elapsed:.6f} s "
                  f"({n_samples / elapsed:.2f} inferences/s)")
        result = decode_fixed_point(self.output_buffer).reshape(n_samples, N_TARGETS)
        return _unscale_y(result) if denorm else result

    def cleanup(self):
        for name in ("input_buffer", "output_buffer"):
            buffer = getattr(self, name)
            if buffer is not None:
                buffer.freebuffer()
                setattr(self, name, None)
