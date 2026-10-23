"""How many eval cases do you need to tell model A from model B?

Two designs:

* paired   -- both models answer the SAME test cases (the normal eval setup).
              Only cases where the models disagree carry signal, so this is a
              McNemar test; sample size from Connor (1987).
* unpaired -- each model is scored on a DIFFERENT random sample of cases.
              Classic two-proportion z-test (Fleiss).

All numbers are per-model test cases. Accuracies are proportions in (0, 1).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import NormalDist

_N = NormalDist()


def _z(p: float) -> float:
    return _N.inv_cdf(p)


def _z_alpha(alpha: float, two_sided: bool) -> float:
    return _z(1 - alpha / 2) if two_sided else _z(1 - alpha)


def _check_prob(name: str, v: float) -> None:
    if not 0 < v < 1:
        raise ValueError(f"{name} must be strictly between 0 and 1, got {v}")


def independent_discordance(acc_a: float, acc_b: float) -> float:
    """Fraction of cases where exactly one model is right, if their errors were
    independent. Real models make correlated mistakes, so the true value is
    usually lower -- which means you need FEWER cases than this assumes."""
    return acc_a * (1 - acc_b) + acc_b * (1 - acc_a)


@dataclass
class SampleSize:
    n: int  # test cases needed (per model; for paired, the shared set size)
    design: str
    acc_a: float
    acc_b: float
    gap: float
    alpha: float
    power: float
    discordance: float | None = None  # paired only
    expected_disagreements: float | None = None  # paired only

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


def sample_size(
    acc_a: float,
    acc_b: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    paired: bool = True,
    discordance: float | None = None,
    two_sided: bool = True,
) -> SampleSize:
    """Test cases needed to detect acc_a != acc_b with the given power.

    discordance: (paired only) fraction of cases where the models disagree.
    If you have a pilot run, measure it; otherwise it defaults to the
    independent-errors estimate, which is conservative for similar models.
    """
    for name, v in [("acc_a", acc_a), ("acc_b", acc_b), ("alpha", alpha), ("power", power)]:
        _check_prob(name, v)
    gap = acc_a - acc_b
    if gap == 0:
        raise ValueError("acc_a and acc_b are equal; no finite sample can detect a zero gap")
    d = abs(gap)
    za, zb = _z_alpha(alpha, two_sided), _z(power)

    if paired:
        psi = independent_discordance(acc_a, acc_b) if discordance is None else discordance
        if not d - 1e-9 <= psi <= 1:
            raise ValueError(
                f"discordance must be between |gap|={d:.4g} and 1 "
                f"(models must disagree at least as often as their accuracies differ), got {psi}"
            )
        psi = max(psi, d)  # absorb float noise from acc_a - acc_b
        if psi - d < 1e-9:  # all disagreements go one way: degenerate, use exact sign test bound
            n = math.ceil(math.log(alpha / (2 if two_sided else 1)) / math.log(1 - d))
        else:
            n = math.ceil((za * math.sqrt(psi) + zb * math.sqrt(psi - d * d)) ** 2 / d**2)
        return SampleSize(n, "paired", acc_a, acc_b, gap, alpha, power, psi, round(n * psi, 1))

    p_bar = (acc_a + acc_b) / 2
    n = math.ceil(
        (
            za * math.sqrt(2 * p_bar * (1 - p_bar))
            + zb * math.sqrt(acc_a * (1 - acc_a) + acc_b * (1 - acc_b))
        )
        ** 2
        / d**2
    )
    return SampleSize(n, "unpaired", acc_a, acc_b, gap, alpha, power)


def min_detectable_gap(
    n: int,
    baseline_acc: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    paired: bool = True,
    discordance: float | None = None,
    two_sided: bool = True,
) -> float:
    """Smallest accuracy improvement over baseline_acc that n cases can reliably
    detect. Inverse of sample_size, by bisection on the gap."""
    _check_prob("baseline_acc", baseline_acc)
    if n < 1:
        raise ValueError("n must be >= 1")
    lo, hi = 1e-6, 1 - baseline_acc - 1e-9

    def need(gap: float) -> int:
        psi = discordance
        if paired and psi is not None:
            psi = max(psi, gap)  # discordance can't be below the gap
        return sample_size(
            baseline_acc + gap, baseline_acc, alpha=alpha, power=power,
            paired=paired, discordance=psi, two_sided=two_sided,
        ).n

    if need(hi) > n:
        raise ValueError(f"{n} cases can't reliably detect any improvement over {baseline_acc}")
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if need(mid) > n else (lo, mid)
    return hi


@dataclass
class Comparison:
    design: str
    acc_a: float
    acc_b: float
    gap: float
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


def _binom_two_sided_p(k: int, n: int) -> float:
    """Exact two-sided binomial test of k successes in n trials at p=0.5."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def compare_paired(
    n: int, a_only: int, b_only: int, both: int, *, alpha: float = 0.05
) -> Comparison:
    """Did A beat B on a shared test set?

    n: total cases. a_only: A right & B wrong. b_only: B right & A wrong.
    both: both right. Uses the exact McNemar test and a Wald CI on the gap.
    """
    if min(n, a_only, b_only, both) < 0 or a_only + b_only + both > n or n == 0:
        raise ValueError("need n > 0 and a_only + b_only + both <= n, all non-negative")
    acc_a, acc_b = (both + a_only) / n, (both + b_only) / n
    gap = acc_a - acc_b
    p10, p01 = a_only / n, b_only / n
    se = math.sqrt(max(p10 + p01 - gap * gap, 0) / n)
    z = _z(1 - alpha / 2)
    p = _binom_two_sided_p(a_only, a_only + b_only)
    return Comparison("paired", acc_a, acc_b, gap, gap - z * se, gap + z * se, p, p < alpha, 1 - alpha)


def compare_unpaired(
    n_a: int, correct_a: int, n_b: int, correct_b: int, *, alpha: float = 0.05
) -> Comparison:
    """Did A beat B when each was scored on its own sample? Two-proportion z-test."""
    if n_a <= 0 or n_b <= 0 or not (0 <= correct_a <= n_a and 0 <= correct_b <= n_b):
        raise ValueError("need n > 0 and 0 <= correct <= n for both models")
    pa, pb = correct_a / n_a, correct_b / n_b
    gap = pa - pb
    pooled = (correct_a + correct_b) / (n_a + n_b)
    se0 = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    p = 1.0 if se0 == 0 else 2 * (1 - _N.cdf(abs(gap) / se0))
    se = math.sqrt(pa * (1 - pa) / n_a + pb * (1 - pb) / n_b)
    z = _z(1 - alpha / 2)
    return Comparison("unpaired", pa, pb, gap, gap - z * se, gap + z * se, p, p < alpha, 1 - alpha)
