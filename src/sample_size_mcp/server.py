"""MCP server exposing the eval sample-size calculator as tools."""

from contextlib import contextmanager

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import stats

mcp = MCPServer("eval-sample-size")


@contextmanager
def _bad_input():
    """Surface ValueError text to the model (plain exceptions are hidden)."""
    try:
        yield
    except ValueError as e:
        raise ToolError(str(e)) from e


@mcp.tool()
def eval_sample_size(
    acc_a: float,
    acc_b: float,
    alpha: float = 0.05,
    power: float = 0.8,
    paired: bool = True,
    discordance: float | None = None,
) -> dict:
    """How many test cases are needed to trust that model A's accuracy really
    differs from model B's.

    acc_a, acc_b: expected accuracies as proportions (e.g. 0.82, 0.78).
    paired: True if both models run on the same test cases (usual case).
    discordance: paired only -- fraction of cases where exactly one model is
      right. Measure it from a pilot run if you can; default assumes
      independent errors, which overestimates n for similar models.
    """
    with _bad_input():
        return stats.sample_size(
            acc_a, acc_b, alpha=alpha, power=power, paired=paired, discordance=discordance
        ).to_dict()


@mcp.tool()
def eval_min_detectable_gap(
    n: int,
    baseline_acc: float,
    alpha: float = 0.05,
    power: float = 0.8,
    paired: bool = True,
    discordance: float | None = None,
) -> dict:
    """Smallest accuracy improvement over baseline_acc that an eval with n test
    cases can reliably detect. Use this to sanity-check an existing eval set."""
    with _bad_input():
        gap = stats.min_detectable_gap(
            n, baseline_acc, alpha=alpha, power=power, paired=paired, discordance=discordance
        )
    return {"n": n, "baseline_acc": baseline_acc, "min_detectable_gap": round(gap, 4)}


@mcp.tool()
def eval_compare_paired(n: int, a_only: int, b_only: int, both: int, alpha: float = 0.05) -> dict:
    """Is A's win over B on a shared test set statistically real?

    n: total cases; a_only: A right & B wrong; b_only: B right & A wrong;
    both: both right. Returns exact McNemar p-value and a CI on the gap."""
    with _bad_input():
        return stats.compare_paired(n, a_only, b_only, both, alpha=alpha).to_dict()


@mcp.tool()
def eval_compare_unpaired(
    n_a: int, correct_a: int, n_b: int, correct_b: int, alpha: float = 0.05
) -> dict:
    """Is A's win over B real when each was scored on a different sample?
    Two-proportion z-test with a CI on the gap."""
    with _bad_input():
        return stats.compare_unpaired(n_a, correct_a, n_b, correct_b, alpha=alpha).to_dict()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
