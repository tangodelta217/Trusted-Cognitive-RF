"""
Unit tests for V0 feature extraction.
"""

import unittest
import numpy as np
from pathlib import Path
import json
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[3]))

from common.features.extract import extract_features, extract_batch, load_config, get_output_shape
from common.features.golden.make_golden_v0 import compute_stable_hash


class TestFeatureExtraction(unittest.TestCase):
    """Tests for feature extraction pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Load config once for all tests."""
        cls.config = load_config()
        cls.n_samples = cls.config["input"]["samples_per_example"]
        cls.expected_shape = get_output_shape(cls.config)
    
    def _generate_random_iq(self, seed: int = 123) -> np.ndarray:
        """Generate random IQ signal for testing."""
        rng = np.random.default_rng(seed)
        return (rng.standard_normal(self.n_samples) + 
                1j * rng.standard_normal(self.n_samples)).astype(np.complex64)
    
    def test_shape_dtype(self):
        """Test that output has correct shape and dtype."""
        iq = self._generate_random_iq()
        features = extract_features(iq, self.config)
        
        self.assertEqual(features.shape, self.expected_shape)
        self.assertEqual(features.dtype, np.float32)
    
    def test_no_nans(self):
        """Test that output contains no NaNs."""
        iq = self._generate_random_iq()
        features = extract_features(iq, self.config)
        
        self.assertFalse(np.any(np.isnan(features)))
    
    def test_no_infs(self):
        """Test that output contains no infinities."""
        iq = self._generate_random_iq()
        features = extract_features(iq, self.config)
        
        self.assertFalse(np.any(np.isinf(features)))
    
    def test_determinism_same_input(self):
        """Test that same input produces same output."""
        iq = self._generate_random_iq(seed=42)
        
        features1 = extract_features(iq.copy(), self.config)
        features2 = extract_features(iq.copy(), self.config)
        
        np.testing.assert_array_equal(features1, features2)
    
    def test_determinism_hash(self):
        """Test that hash is stable across runs."""
        iq = self._generate_random_iq(seed=42)
        
        features1 = extract_features(iq.copy(), self.config)
        features2 = extract_features(iq.copy(), self.config)
        
        hash1 = compute_stable_hash(features1)
        hash2 = compute_stable_hash(features2)
        
        self.assertEqual(hash1, hash2)
    
    def test_batch_extraction(self):
        """Test batch feature extraction."""
        batch_size = 5
        iq_batch = np.stack([
            self._generate_random_iq(seed=i) for i in range(batch_size)
        ])
        
        features_batch = extract_batch(iq_batch, self.config)
        
        expected_batch_shape = (batch_size,) + self.expected_shape
        self.assertEqual(features_batch.shape, expected_batch_shape)
        self.assertEqual(features_batch.dtype, np.float32)
    
    def test_batch_matches_individual(self):
        """Test that batch extraction matches individual extraction."""
        batch_size = 3
        iq_batch = np.stack([
            self._generate_random_iq(seed=i) for i in range(batch_size)
        ])
        
        # Batch extraction
        features_batch = extract_batch(iq_batch, self.config)
        
        # Individual extraction
        for i in range(batch_size):
            features_individual = extract_features(iq_batch[i], self.config)
            np.testing.assert_array_equal(
                features_batch[i], features_individual,
                err_msg=f"Batch and individual differ at index {i}"
            )
    
    def test_different_inputs_different_outputs(self):
        """Test that different inputs produce different outputs."""
        iq1 = self._generate_random_iq(seed=1)
        iq2 = self._generate_random_iq(seed=2)
        
        features1 = extract_features(iq1, self.config)
        features2 = extract_features(iq2, self.config)
        
        self.assertFalse(np.allclose(features1, features2))
    
    def test_normalization(self):
        """Test that output is approximately standardized."""
        iq = self._generate_random_iq()
        features = extract_features(iq, self.config)
        
        # For per-example standardization, mean should be ~0, std ~1
        mean = np.mean(features)
        std = np.std(features)
        
        self.assertAlmostEqual(mean, 0.0, places=5)
        self.assertAlmostEqual(std, 1.0, places=1)
    
    def test_wrong_length_raises(self):
        """Test that wrong input length raises ValueError."""
        wrong_length = 1024
        iq = np.zeros(wrong_length, dtype=np.complex64)
        
        with self.assertRaises(ValueError):
            extract_features(iq, self.config)
    
    def test_golden_matches(self):
        """Test that extracted features match golden examples."""
        golden_dir = Path(__file__).parent.parent / "golden"
        inputs_path = golden_dir / "golden_inputs_v0.npz"
        features_path = golden_dir / "golden_features_v0.npz"
        hashes_path = golden_dir / "golden_hashes_v0.json"
        
        # Skip if golden files don't exist
        if not all(p.exists() for p in [inputs_path, features_path, hashes_path]):
            self.skipTest("Golden files not found. Run make_golden_v0.py first.")
        
        golden_inputs = np.load(inputs_path)
        golden_features = np.load(features_path)
        with open(hashes_path, "r") as f:
            golden_hashes = json.load(f)
        
        for name in golden_inputs.files:
            with self.subTest(signal=name):
                iq = golden_inputs[name]
                features = extract_features(iq, self.config)
                
                # Check allclose
                np.testing.assert_allclose(
                    features, golden_features[name],
                    rtol=1e-5, atol=1e-6,
                    err_msg=f"Features don't match golden for {name}"
                )
                
                # Check hash
                current_hash = compute_stable_hash(features)
                self.assertEqual(
                    current_hash, golden_hashes[name],
                    f"Hash mismatch for {name}"
                )


if __name__ == "__main__":
    unittest.main()
