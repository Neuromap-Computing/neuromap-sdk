"""Tests for weight nibble packing/unpacking."""

import numpy as np
import pytest

from neuromap._internal.packing import pack_weights_nibble, unpack_weights_nibble


class TestPackUnpack:
    """Round-trip tests for nibble packing."""

    def test_round_trip_even(self) -> None:
        """Pack and unpack a 4x4 matrix (even number of elements)."""
        w = np.array(
            [[ 1, -2,  3, -4],
             [ 5, -6,  7, -7],
             [ 0,  1, -1,  2],
             [-3,  4, -5,  6]],
            dtype=np.int8,
        )
        packed = pack_weights_nibble(w)
        recovered = unpack_weights_nibble(packed, 4, 4)
        np.testing.assert_array_equal(recovered, w)

    def test_round_trip_odd_cols(self) -> None:
        """Pack and unpack a 2x3 matrix (odd number of elements)."""
        w = np.array([[1, -1, 2], [3, -3, 0]], dtype=np.int8)
        packed = pack_weights_nibble(w)
        recovered = unpack_weights_nibble(packed, 2, 3)
        np.testing.assert_array_equal(recovered, w)

    def test_neurosoc_v1_layer_size(self) -> None:
        """16x16 matrix produces 128 packed bytes."""
        w = np.random.randint(-7, 8, size=(16, 16), dtype=np.int8)
        packed = pack_weights_nibble(w)
        assert len(packed) == 128  # 256 / 2
        recovered = unpack_weights_nibble(packed, 16, 16)
        np.testing.assert_array_equal(recovered, w)

    def test_boundary_values(self) -> None:
        """Test min/max 4-bit values: -8 to +7."""
        w = np.array([[-8, 7], [-1, 0]], dtype=np.int8)
        packed = pack_weights_nibble(w)
        recovered = unpack_weights_nibble(packed, 2, 2)
        np.testing.assert_array_equal(recovered, w)

    def test_all_zeros(self) -> None:
        """All-zero matrix round-trips correctly."""
        w = np.zeros((4, 4), dtype=np.int8)
        packed = pack_weights_nibble(w)
        assert packed == b"\x00" * 8
        recovered = unpack_weights_nibble(packed, 4, 4)
        np.testing.assert_array_equal(recovered, w)

    def test_rejects_non_4bit(self) -> None:
        """Only 4-bit packing is supported."""
        w = np.zeros((2, 2), dtype=np.int8)
        with pytest.raises(ValueError, match="Only 4-bit"):
            pack_weights_nibble(w, bits=8)

    def test_rejects_1d(self) -> None:
        """Rejects 1-D arrays."""
        with pytest.raises(ValueError, match="2-D"):
            pack_weights_nibble(np.array([1, 2, 3], dtype=np.int8))

    def test_unpack_insufficient_data(self) -> None:
        """Raises on too-short packed data."""
        with pytest.raises(ValueError, match="Need at least"):
            unpack_weights_nibble(b"\x00", 4, 4)
