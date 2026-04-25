#!/usr/bin/env python3
"""Unit tests for deterministic.py and its integration with mfu_tracker.py."""

import hashlib
import io
import os
import struct
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))

import deterministic
import mfu_tracker


class TestLossChecksumTracker(unittest.TestCase):
    """Tests for the LossChecksumTracker class."""

    def test_empty_tracker(self):
        t = deterministic.LossChecksumTracker()
        self.assertEqual(t.count, 0)
        self.assertEqual(len(t.hexdigest()), 64)  # SHA-256 hex

    def test_single_update(self):
        t = deterministic.LossChecksumTracker()
        t.update("3.14")
        self.assertEqual(t.count, 1)

    def test_deterministic_across_instances(self):
        """Two trackers fed the same values must produce the same digest."""
        t1 = deterministic.LossChecksumTracker()
        t2 = deterministic.LossChecksumTracker()
        for v in ["12.345", "6.789", "0.001"]:
            t1.update(v)
            t2.update(v)
        self.assertEqual(t1.hexdigest(), t2.hexdigest())
        self.assertEqual(t1.count, 3)

    def test_different_values_differ(self):
        t1 = deterministic.LossChecksumTracker()
        t2 = deterministic.LossChecksumTracker()
        t1.update("1.0")
        t2.update("2.0")
        self.assertNotEqual(t1.hexdigest(), t2.hexdigest())

    def test_order_matters(self):
        t1 = deterministic.LossChecksumTracker()
        t2 = deterministic.LossChecksumTracker()
        t1.update("1.0")
        t1.update("2.0")
        t2.update("2.0")
        t2.update("1.0")
        self.assertNotEqual(t1.hexdigest(), t2.hexdigest())

    def test_packs_as_float32(self):
        """Verify the tracker uses IEEE-754 float32 (not float64)."""
        t = deterministic.LossChecksumTracker()
        t.update("3.14")
        expected = hashlib.sha256(struct.pack("!f", 3.14)).hexdigest()
        self.assertEqual(t.hexdigest(), expected)


class TestExtractLoss(unittest.TestCase):
    """Tests for extract_loss() regex matching and tracker delegation."""

    def setUp(self):
        deterministic.tracker = deterministic.LossChecksumTracker()

    def test_matches_maxtext_log_line(self):
        line = "completed step: 5, seconds: 1.23, loss: 12.345, lr: 0.001"
        deterministic.extract_loss(line)
        self.assertEqual(deterministic.tracker.count, 1)

    def test_ignores_unrelated_line(self):
        deterministic.extract_loss("Starting training on 8 GPUs...")
        self.assertEqual(deterministic.tracker.count, 0)

    def test_ignores_partial_match(self):
        deterministic.extract_loss("loss: 5.0 but no completed step")
        self.assertEqual(deterministic.tracker.count, 0)

    def test_multiple_lines(self):
        lines = [
            "completed step: 1, seconds: 2.1, loss: 10.5, lr: 0.01",
            "completed step: 2, seconds: 2.0, loss: 9.8, lr: 0.01",
            "completed step: 3, seconds: 2.1, loss: 9.2, lr: 0.01",
        ]
        for line in lines:
            deterministic.extract_loss(line)
        self.assertEqual(deterministic.tracker.count, 3)


class TestVerifyEnv(unittest.TestCase):
    """Tests for verify_env() — checks it warns or passes correctly."""

    def _capture_verify(self, env_overrides):
        env = {
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "XLA_FLAGS": "--xla_gpu_deterministic_ops=true",
            "TF_DETERMINISTIC_OPS": "1",
            "HIPBLASLT_DETERMINISTIC": "1",
            "NVTE_FUSED_ATTN": "0",
        }
        env.update(env_overrides)
        buf = io.StringIO()
        with mock.patch.dict(os.environ, env, clear=False), \
             mock.patch("sys.stdout", buf):
            deterministic.verify_env()
        return buf.getvalue()

    def test_all_flags_pass(self):
        output = self._capture_verify({})
        self.assertIn("All env-var checks passed", output)
        self.assertNotIn("WARNING", output)

    def test_missing_nvte_fused_attn_warns(self):
        output = self._capture_verify({"NVTE_FUSED_ATTN": "1"})
        self.assertIn("WARNING", output)
        self.assertIn("NVTE_FUSED_ATTN", output)

    def test_missing_tf_deterministic_ops_warns(self):
        output = self._capture_verify({"TF_DETERMINISTIC_OPS": "0"})
        self.assertIn("WARNING", output)
        self.assertIn("TF_DETERMINISTIC_OPS", output)

    def test_missing_xla_flag_warns(self):
        output = self._capture_verify({"XLA_FLAGS": ""})
        self.assertIn("WARNING", output)
        self.assertIn("xla_gpu_deterministic_ops", output)

    def test_missing_hipblaslt_warns(self):
        output = self._capture_verify({"HIPBLASLT_DETERMINISTIC": ""})
        self.assertIn("WARNING", output)
        self.assertIn("HIPBLASLT_DETERMINISTIC", output)


class TestApplyPatches(unittest.TestCase):
    """Tests for apply_patches() — monkey-patching MaxText's initialize()."""

    def test_noop_when_mode_off(self):
        """When DETERMINISTIC_MODE is off and no PRNG override, no patching."""
        fake_train = mock.MagicMock()
        orig_init = fake_train.initialize
        with mock.patch.dict(os.environ, {"DETERMINISTIC_MODE": "0"}, clear=False):
            os.environ.pop("JAX_DEFAULT_PRNG_IMPL", None)
            deterministic.apply_patches(fake_train)
        self.assertIs(fake_train.initialize, orig_init)

    def test_patches_when_mode_on(self):
        """When DETERMINISTIC_MODE=1, initialize should be replaced."""
        fake_train = mock.MagicMock()
        orig_init = fake_train.initialize
        env = {"DETERMINISTIC_MODE": "1", "JAX_DEFAULT_PRNG_IMPL": "threefry2x32"}
        with mock.patch.dict(os.environ, env, clear=False):
            deterministic.apply_patches(fake_train)
        self.assertIsNot(fake_train.initialize, orig_init)

    def test_patched_init_calls_original(self):
        """The patched initialize() must call the original."""
        call_log = []
        fake_train = mock.MagicMock()
        fake_train.initialize = lambda argv: call_log.append(("init", argv))

        all_env = {
            "DETERMINISTIC_MODE": "1",
            "JAX_DEFAULT_PRNG_IMPL": "threefry2x32",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "XLA_FLAGS": "--xla_gpu_deterministic_ops=true",
            "TF_DETERMINISTIC_OPS": "1",
            "HIPBLASLT_DETERMINISTIC": "1",
            "NVTE_FUSED_ATTN": "0",
        }
        with mock.patch.dict(os.environ, all_env, clear=False):
            deterministic.apply_patches(fake_train)
            with mock.patch("builtins.print"):
                fake_train.initialize(["--arg1"])

        self.assertEqual(len(call_log), 1)
        self.assertEqual(call_log[0], ("init", ["--arg1"]))


class TestPrintLossChecksum(unittest.TestCase):
    """Tests for print_loss_checksum()."""

    def setUp(self):
        deterministic.tracker = deterministic.LossChecksumTracker()

    def test_no_output_when_empty(self):
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            deterministic.print_loss_checksum()
        self.assertEqual(buf.getvalue(), "")

    def test_prints_when_has_data(self):
        deterministic.tracker.update("5.0")
        deterministic.tracker.update("4.5")
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            deterministic.print_loss_checksum()
        output = buf.getvalue()
        self.assertIn("[determinism]", output)
        self.assertIn("loss_checksum=", output)
        self.assertIn("steps=2", output)


class TestMFUStreamDelegation(unittest.TestCase):
    """Tests that _MFUStream correctly delegates loss extraction to deterministic."""

    def setUp(self):
        deterministic.tracker = deterministic.LossChecksumTracker()

    def test_loss_line_feeds_tracker(self):
        wrapped = io.StringIO()
        stream = mfu_tracker._MFUStream(wrapped, peak_tflops=1000.0)
        stream.write("completed step: 1, seconds: 2.0, loss: 7.123, lr: 0.001\n")
        self.assertEqual(deterministic.tracker.count, 1)

    def test_tflops_line_gets_mfu(self):
        wrapped = io.StringIO()
        stream = mfu_tracker._MFUStream(wrapped, peak_tflops=1000.0)
        stream.write("TFLOP/s/device: 500.0\n")
        output = wrapped.getvalue()
        self.assertIn("MFU: 50.00%", output)

    def test_plain_line_passes_through(self):
        wrapped = io.StringIO()
        stream = mfu_tracker._MFUStream(wrapped, peak_tflops=1000.0)
        stream.write("hello world\n")
        self.assertEqual(wrapped.getvalue(), "hello world\n")
        self.assertEqual(deterministic.tracker.count, 0)

    def test_combined_line(self):
        """A line with both TFLOP/s and loss (unlikely but possible)."""
        wrapped = io.StringIO()
        stream = mfu_tracker._MFUStream(wrapped, peak_tflops=2500.0)
        stream.write("completed step: 5, seconds: 2.0, TFLOP/s/device: 968.0, loss: 3.5, lr: 0.001\n")
        self.assertEqual(deterministic.tracker.count, 1)
        self.assertIn("MFU:", wrapped.getvalue())


class TestMFUTrackerGPUDetection(unittest.TestCase):
    """Sanity checks for GPU detection and peak TFLOPS (unchanged code)."""

    def test_peak_tflops_table_has_entries(self):
        self.assertGreater(len(mfu_tracker._GPU_PEAK_TFLOPS), 10)

    def test_mi355x_bf16(self):
        self.assertEqual(mfu_tracker._GPU_PEAK_TFLOPS["MI355X"]["bf16"], 2500)

    def test_gfx_to_gpu_mapping(self):
        self.assertEqual(mfu_tracker._GFX_TO_GPU["gfx950"], "MI355X")
        self.assertEqual(mfu_tracker._GFX_TO_GPU["gfx942"], "MI300X")

    def test_resolve_dtype_default(self):
        self.assertEqual(mfu_tracker.resolve_compute_dtype([]), "bf16")

    def test_resolve_dtype_cli(self):
        self.assertEqual(mfu_tracker.resolve_compute_dtype(["dtype=float16"]), "fp16")

    def test_resolve_dtype_fp8(self):
        self.assertEqual(mfu_tracker.resolve_compute_dtype(["quantization=fp8"]), "fp8")


if __name__ == "__main__":
    unittest.main()
