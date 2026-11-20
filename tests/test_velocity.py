import unittest

import numpy as np

from dhara.velocity import velocity_between


class VelocityContract(unittest.TestCase):
    def test_signed_velocity_is_invariant_to_accepted_capture_interval(self):
        previous = [3.0, -2.0, 1.0]
        cases = [
            (100_000_000, [3.05, -2.025, 1.0125]),
            (20_000_000, [3.01, -2.005, 1.0025]),
            (270_000_000, [3.135, -2.0675, 1.03375]),
        ]
        for interval_ns, current in cases:
            with self.subTest(interval_ns=interval_ns):
                actual = velocity_between(previous, current, 1_000_000_000, 1_000_000_000 + interval_ns)
                np.testing.assert_allclose(actual, [0.5, -0.25, 0.125], rtol=0, atol=1e-12)

    def test_large_clock_epoch_does_not_erase_a_short_interval(self):
        base = 1_800_000_000_000_000_000
        actual = velocity_between([0, 0, 0], [0.00005, 0, 0], base, base + 100_000)
        np.testing.assert_allclose(actual, [0.5, 0, 0], rtol=0, atol=1e-12)

    def test_duplicate_and_backwards_capture_times_are_not_velocities(self):
        for first, second in [(100, 100), (101, 100)]:
            with self.subTest(first=first, second=second), self.assertRaises(ValueError):
                velocity_between([0, 0, 0], [0.05, 0, 0], first, second)
        with self.assertRaises(TypeError):
            velocity_between([0, 0, 0], [0.05, 0, 0], 1.0, 1.1)

    def test_returned_velocity_cannot_corrupt_pose_inputs(self):
        previous = np.array([3.0, -2.0, 1.0])
        current = np.array([3.05, -2.0, 1.0])
        actual = velocity_between(previous, current, 100, 100_000_100)
        np.testing.assert_allclose(actual, [0.5, 0, 0], rtol=0, atol=1e-12)
        actual[:] = 99
        np.testing.assert_array_equal(previous, [3.0, -2.0, 1.0])
        np.testing.assert_array_equal(current, [3.05, -2.0, 1.0])

    def test_missing_components_and_nonfinite_positions_are_refused(self):
        for current in ([1, 2], [[1], [2], [3]], [np.nan, 0, 0], [np.inf, 0, 0]):
            with self.subTest(current=current), self.assertRaises(ValueError):
                velocity_between([0, 0, 0], current, 100, 200)

    def test_unrepresentable_rate_is_not_returned_as_infinite_velocity(self):
        with self.assertRaises(FloatingPointError):
            velocity_between([-1e308, 0, 0], [1e308, 0, 0], 100, 101)


if __name__ == "__main__":
    unittest.main()
