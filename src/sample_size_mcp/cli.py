"""evalsize: command-line front end.

  evalsize n 0.82 0.78                  # cases needed (paired, same test set)
  evalsize n 0.82 0.78 --discordance 0.1
  evalsize n 0.82 0.78 --unpaired
  evalsize gap 500 0.78                 # smallest gap 500 cases can detect
  evalsize compare 500 --a-only 41 --b-only 22 --both 360
"""

import argparse
import sys

from . import stats


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--alpha", type=float, default=0.05, help="false-positive rate (default 0.05)")
    p.add_argument("--power", type=float, default=0.8, help="chance of detecting a real gap (default 0.8)")
    p.add_argument("--unpaired", action="store_true", help="models scored on different samples")
    p.add_argument("--discordance", type=float, help="paired: fraction of cases where models disagree")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evalsize", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("n", help="test cases needed to detect a gap")
    p.add_argument("acc_a", type=float)
    p.add_argument("acc_b", type=float)
    _common(p)

    p = sub.add_parser("gap", help="smallest gap n cases can detect")
    p.add_argument("n", type=int)
    p.add_argument("baseline_acc", type=float)
    _common(p)

    p = sub.add_parser("compare", help="is an observed paired difference real?")
    p.add_argument("n", type=int)
    p.add_argument("--a-only", type=int, required=True, help="cases only A got right")
    p.add_argument("--b-only", type=int, required=True, help="cases only B got right")
    p.add_argument("--both", type=int, required=True, help="cases both got right")
    p.add_argument("--alpha", type=float, default=0.05)

    a = ap.parse_args(argv)
    try:
        if a.cmd == "n":
            r = stats.sample_size(a.acc_a, a.acc_b, alpha=a.alpha, power=a.power,
                                  paired=not a.unpaired, discordance=a.discordance)
            print(f"{r.n} test cases {'(shared by both models)' if r.design == 'paired' else 'per model'}")
            print(f"  to detect {r.acc_a:.1%} vs {r.acc_b:.1%} (gap {abs(r.gap):.1%}) "
                  f"at alpha={r.alpha}, power={r.power:.0%}")
            if r.discordance is not None:
                src = "given" if a.discordance is not None else "assumed independent errors"
                print(f"  discordance {r.discordance:.1%} ({src}); "
                      f"~{r.expected_disagreements:.0f} cases will carry the signal")
        elif a.cmd == "gap":
            g = stats.min_detectable_gap(a.n, a.baseline_acc, alpha=a.alpha, power=a.power,
                                         paired=not a.unpaired, discordance=a.discordance)
            print(f"{a.n} cases can reliably detect a gap of {g:.1%} "
                  f"({a.baseline_acc:.1%} -> {a.baseline_acc + g:.1%})")
        else:
            c = stats.compare_paired(a.n, a.a_only, a.b_only, a.both, alpha=a.alpha)
            verdict = "SIGNIFICANT" if c.significant else "not significant"
            print(f"A {c.acc_a:.1%} vs B {c.acc_b:.1%}: gap {c.gap:+.1%}, "
                  f"{c.confidence:.0%} CI [{c.ci_low:+.1%}, {c.ci_high:+.1%}], "
                  f"p={c.p_value:.3g} -> {verdict}")
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
