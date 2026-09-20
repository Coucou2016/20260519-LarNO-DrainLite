from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "extended_study"))

import run_final_hybrid_major_revision as v4  # noqa: E402
import run_reviewer_v3_drainlite as core  # noqa: E402


class DrainLiteCoreTests(unittest.TestCase):
    def test_masked_neighbourhood_excludes_inactive_wall_values(self) -> None:
        values = np.ones((5, 5), dtype=np.float32)
        values[2, 2] = 1000.0
        active = np.ones((5, 5), dtype=bool)
        active[2, 2] = False
        mean = core.masked_uniform_mean(values, active, size=3)
        self.assertAlmostEqual(float(mean[2, 1]), 1.0, places=6)
        self.assertAlmostEqual(float(mean[2, 2]), 1.0, places=6)

    def test_training_event_prior_is_event_excluded(self) -> None:
        arrays = {
            "e1": {"surface": np.zeros((2, 2, 2), np.float32), "coupled": np.ones((2, 2, 2), np.float32)},
            "e2": {"surface": np.zeros((2, 2, 2), np.float32), "coupled": np.full((2, 2, 2), 3.0, np.float32)},
        }
        prior_for_e1 = v4.mean_residual(arrays, ["e2"])
        prior_for_e2 = v4.mean_residual(arrays, ["e1"])
        np.testing.assert_allclose(prior_for_e1, 3.0)
        np.testing.assert_allclose(prior_for_e2, 1.0)

    def test_zero_network_control_regenerates_zero_descriptors(self) -> None:
        shape = (4, 5)
        static = {
            "dem_m": np.zeros(shape, np.float32),
            "dem_slope": np.zeros(shape, np.float32),
            "active_mask": np.ones(shape, bool),
        }
        for name in core.PRIMITIVE_DRAINAGE:
            static[name] = np.ones(shape, np.float32)
        transformed = v4.drop_network_static(static)
        derived = [
            "pipe_density_3x3",
            "pipe_density_7x7",
            "inlet_density_7x7",
            "capacity_density_7x7",
        ]
        for name in core.PRIMITIVE_DRAINAGE + derived:
            np.testing.assert_array_equal(transformed[name], 0.0)

    def test_csi_empty_union_is_perfect(self) -> None:
        empty = np.zeros(4, dtype=np.float32)
        self.assertEqual(v4.csi_subset(empty, empty, 0.03), 1.0)


if __name__ == "__main__":
    unittest.main()
