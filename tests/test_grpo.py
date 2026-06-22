from __future__ import annotations

import numpy as np

from ftpipe.align.grpo import group_advantages, parse_completion, reward_completion


def test_structured_reward_levels():
    weights = {"exact": 1.0, "format": 0.5, "category": 0.25}
    assert reward_completion("billing high", "billing", "high", weights) == 1.75
    assert reward_completion("billing low", "billing", "high", weights) == 0.75
    assert reward_completion("shipping high", "billing", "high", weights) == 0.5
    assert reward_completion("not a valid answer", "billing", "high", weights) == 0.0
    assert parse_completion("billing high") == (True, "billing", "high")
    assert parse_completion("billing high please") == (False, None, None)


def test_group_advantages_are_zero_mean_and_flat_for_ties():
    rewards = np.asarray([1.0, 0.5, 0.5, 0.0])
    advantages = group_advantages(rewards)
    assert abs(float(advantages.mean())) < 1e-12
    assert np.allclose(group_advantages(np.asarray([1.0, 1.0, 1.0])), 0.0)
