from __future__ import annotations

import numpy as np
import pytest

from src.problems.topology.support.policy import (
    constitutive_plane_stress,
    message_is_converged,
    relative_state_change,
    staircase_p_step,
)


def test_constitutive_plane_stress_matches_reference_formula():
    mat = constitutive_plane_stress(2.0, 0.25)
    expected = (2.0 / (1.0 - 0.25**2)) * np.array(
        [[1.0, 0.25, 0.0], [0.25, 1.0, 0.0], [0.0, 0.0, 0.375]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(mat, expected)


def test_constitutive_plane_stress_validates_inputs():
    with pytest.raises(ValueError):
        constitutive_plane_stress(0.0, 0.25)
    with pytest.raises(ValueError):
        constitutive_plane_stress(1.0, 0.75)


def test_message_is_converged_accepts_expected_markers():
    assert message_is_converged("Converged by tolerance")
    assert message_is_converged("outer tolerance satisfied")
    assert not message_is_converged("maximum number of iterations reached")


def test_staircase_p_step_applies_schedule():
    assert staircase_p_step(1.0, p_max=3.0, p_increment=0.5, continuation_interval=2, outer_it=1) == 0.0
    assert staircase_p_step(1.0, p_max=3.0, p_increment=0.5, continuation_interval=2, outer_it=2) == 0.5
    assert staircase_p_step(
        2.8,
        p_max=3.0,
        p_increment=0.5,
        continuation_interval=2,
        outer_it=2,
    ) == pytest.approx(0.2)


def test_relative_state_change_uses_only_freedofs():
    current = np.array([1.0, 2.0, 8.0, 4.0])
    previous = np.array([0.0, 2.0, 4.0, 4.0])
    freedofs = np.array([0, 2])
    expected = np.linalg.norm([1.0, 4.0]) / max(1.0, np.linalg.norm([0.0, 4.0]))
    assert relative_state_change(current, previous, freedofs) == pytest.approx(expected)
    assert np.isinf(relative_state_change(current, None, freedofs))
