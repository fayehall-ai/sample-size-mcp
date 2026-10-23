import random

import pytest

from sample_size_mcp import stats


def test_unpaired_matches_textbook():
    # Fleiss (no continuity correction), 0.85 vs 0.65, alpha=.05 two-sided, power=.8:
    # (1.96*sqrt(2*.75*.25) + 0.8416*sqrt(.1275+.2275))^2 / .2^2 = 72.4 -> 73
    assert stats.sample_size(0.85, 0.65, paired=False).n == 73


def test_paired_needs_fewer_when_models_agree_more():
    # With independent errors pairing buys ~nothing; correlated errors are where it pays.
    independent = stats.sample_size(0.82, 0.78).n
    unpaired = stats.sample_size(0.82, 0.78, paired=False).n
    correlated = stats.sample_size(0.82, 0.78, discordance=0.08).n
    assert independent == pytest.approx(unpaired, rel=0.01)
    assert correlated < independent / 3


def test_symmetric_and_shrinks_with_gap():
    assert stats.sample_size(0.8, 0.75).n == stats.sample_size(0.75, 0.8).n
    assert stats.sample_size(0.8, 0.70).n < stats.sample_size(0.8, 0.75).n


def test_min_detectable_gap_inverts_sample_size():
    n = stats.sample_size(0.80, 0.75).n
    gap = stats.min_detectable_gap(n, 0.75)
    assert gap == pytest.approx(0.05, abs=0.002)


@pytest.mark.parametrize(
    "bad", [dict(acc_a=0.8, acc_b=0.8), dict(acc_a=1.2, acc_b=0.8), dict(acc_a=0.8, acc_b=0.7, discordance=0.05)]
)
def test_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        stats.sample_size(**bad)


def test_compare_paired():
    c = stats.compare_paired(500, a_only=41, b_only=22, both=360)
    assert c.acc_a == pytest.approx(0.802) and c.acc_b == pytest.approx(0.764)
    assert c.p_value < 0.05 and c.significant
    assert c.ci_low < c.gap < c.ci_high
    assert not stats.compare_paired(500, 30, 25, 360).significant


def test_compare_unpaired():
    c = stats.compare_unpaired(1000, 820, 1000, 780)
    assert c.p_value == pytest.approx(0.025, abs=0.002)


def _simulated_power(n, acc_a, acc_b, paired, trials=2000, seed=0):
    """Fraction of simulated evals that come out significant."""
    rng = random.Random(seed)
    hits = 0
    for _ in range(trials):
        if paired:  # independent errors, matching the default discordance assumption
            a = [rng.random() < acc_a for _ in range(n)]
            b = [rng.random() < acc_b for _ in range(n)]
            a_only = sum(x and not y for x, y in zip(a, b))
            b_only = sum(y and not x for x, y in zip(a, b))
            both = sum(x and y for x, y in zip(a, b))
            hits += stats.compare_paired(n, a_only, b_only, both).significant
        else:
            ca = sum(rng.random() < acc_a for _ in range(n))
            cb = sum(rng.random() < acc_b for _ in range(n))
            hits += stats.compare_unpaired(n, ca, n, cb).significant
    return hits / trials


@pytest.mark.parametrize("paired", [True, False])
def test_recommended_n_actually_delivers_power(paired):
    n = stats.sample_size(0.80, 0.72, paired=paired).n
    assert _simulated_power(n, 0.80, 0.72, paired) == pytest.approx(0.80, abs=0.04)
