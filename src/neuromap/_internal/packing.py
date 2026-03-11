"""Weight nibble packing/unpacking for ASIC SRAM transfer.

4-bit signed weights are packed two per byte (high nibble first, low nibble
second) in row-major order.  This matches the NeuroSoC-v1 SPI register layout.
"""

from __future__ import annotations

import numpy as np


def pack_weights_nibble(weights: np.ndarray, bits: int = 4) -> bytes:
    """Pack an int8 weight matrix into nibble-packed bytes for ASIC SRAM.

    Two signed weights are stored per byte: the first weight in the high
    nibble (bits [7:4]) and the second in the low nibble (bits [3:0]).
    Values are masked to *bits* width (default 4).

    Args:
        weights: 2-D int8 array of quantized weights (row-major).
        bits: Bit-width per weight (must be 4).

    Returns:
        Packed bytes ready for SPI transfer.

    Raises:
        ValueError: If *bits* is not 4 or the array is not 2-D.
    """
    if bits != 4:
        raise ValueError(f"Only 4-bit packing is supported, got {bits}.")
    if weights.ndim != 2:
        raise ValueError(f"Expected 2-D weight matrix, got {weights.ndim}-D.")

    flat = weights.astype(np.int8).ravel()
    mask = 0x0F  # 4-bit mask

    # Pad to even length if needed
    if len(flat) % 2 != 0:
        flat = np.append(flat, np.int8(0))

    packed = bytearray(len(flat) // 2)
    for i in range(0, len(flat), 2):
        hi = int(flat[i]) & mask
        lo = int(flat[i + 1]) & mask
        packed[i // 2] = (hi << 4) | lo

    return bytes(packed)


def unpack_weights_nibble(
    data: bytes, rows: int, cols: int, bits: int = 4
) -> np.ndarray:
    """Unpack nibble-packed bytes back to an int8 weight matrix.

    Args:
        data: Packed weight bytes.
        rows: Number of rows in the original matrix.
        cols: Number of columns in the original matrix.
        bits: Bit-width per weight (must be 4).

    Returns:
        2-D int8 array of shape ``(rows, cols)``.

    Raises:
        ValueError: If *bits* is not 4 or data length is insufficient.
    """
    if bits != 4:
        raise ValueError(f"Only 4-bit unpacking is supported, got {bits}.")

    total = rows * cols
    needed_bytes = (total + 1) // 2
    if len(data) < needed_bytes:
        raise ValueError(
            f"Need at least {needed_bytes} bytes for {rows}x{cols} matrix, "
            f"got {len(data)}."
        )

    flat = np.empty(total, dtype=np.int8)
    for i in range(needed_bytes):
        hi = (data[i] >> 4) & 0x0F
        lo = data[i] & 0x0F
        # Sign-extend from 4 bits
        if hi >= 8:
            hi -= 16
        if lo >= 16:  # can't happen for 4-bit but guard
            lo -= 16
        if lo >= 8:
            lo -= 16
        idx = i * 2
        if idx < total:
            flat[idx] = hi
        if idx + 1 < total:
            flat[idx + 1] = lo

    return flat[:total].reshape(rows, cols)
