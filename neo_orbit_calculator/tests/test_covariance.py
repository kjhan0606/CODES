import numpy as np

from neo_orbit_calculator.covariance import (
    CovarianceSolution,
    _dense_encounter_grid,
    sample_virtual_asteroids,
)


def _solution() -> CovarianceSolution:
    covariance = np.diag(
        [2.0e-10, 4.0e-12, 9.0e-4, 1.6e-8, 2.5e-8, 3.6e-8]
    )
    covariance[0, 1] = covariance[1, 0] = 1.2e-11
    return CovarianceSolution(
        designation="synthetic",
        fullname="Synthetic test orbit",
        orbit_id="test",
        epoch_jd_tdb=2460000.5,
        labels=("e", "q", "tp", "node", "peri", "i"),
        mean=np.array([0.2, 0.8, 2459900.0, 10.0, 20.0, 5.0]),
        covariance=covariance,
        model_parameters={},
    )


def test_virtual_asteroid_sampling_is_reproducible() -> None:
    solution = _solution()
    first = sample_virtual_asteroids(solution, clones=32, seed=17)
    second = sample_virtual_asteroids(solution, clones=32, seed=17)
    np.testing.assert_array_equal(first, second)


def test_virtual_asteroid_sampling_preserves_full_covariance() -> None:
    solution = _solution()
    samples = sample_virtual_asteroids(solution, clones=100_000, seed=5)
    measured = np.cov(samples, rowvar=False)
    np.testing.assert_allclose(
        np.diag(measured),
        np.diag(solution.covariance),
        rtol=0.025,
    )
    expected_correlation = (
        solution.covariance[0, 1]
        / np.sqrt(solution.covariance[0, 0] * solution.covariance[1, 1])
    )
    measured_correlation = np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]
    assert np.isclose(measured_correlation, expected_correlation, atol=0.01)


def test_dense_encounter_grid_resolves_ten_minutes() -> None:
    coarse = np.linspace(0.0, 40.0 * 86400.0, 41)
    dense, lower, upper = _dense_encounter_grid(coarse, 20)
    assert lower == 15.0 * 86400.0
    assert upper == 25.0 * 86400.0
    in_window = dense[(dense >= lower) & (dense <= upper)]
    assert np.max(np.diff(in_window)) <= 600.0
