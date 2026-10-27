"""Command-line report, until the Streamlit dashboard exists.

Usage:
    python -m simulator templates/guesthouse.yaml
    python -m simulator templates/guesthouse.yaml -n 20000 --seed 7
"""

from __future__ import annotations

import argparse

from . import compare, load_business, simulate, summarize, tornado
from .formatting import money


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m simulator", description="Simulate a business template.")
    parser.add_argument("template", help="path to a YAML or JSON business file")
    parser.add_argument("-n", "--scenarios", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    business = load_business(args.template)
    s = summarize(simulate(business, n=args.scenarios, seed=args.seed))
    cur, per = s.currency, s.period

    print(f"\n{business.name}: {s.n:,} scenarios over {s.horizon} {per}s\n")
    tp = s.total_profit
    print(f"  Total profit      P10 {money(tp.p10)}   median {money(tp.p50)}   P90 {money(tp.p90)} {cur}")
    ec = s.ending_cash
    print(f"  Ending cash       P10 {money(ec.p10)}   median {money(ec.p50)}   P90 {money(ec.p90)} {cur}")
    print(f"  Chance of a loss  {s.prob_loss:.1%}")
    when = f", typically by {per} {s.cash_runs_out_period:.0f}" if s.cash_runs_out_period else ""
    print(f"  Cash runs out     {s.prob_cash_runs_out:.1%} of scenarios{when}")
    print(f"  Break-even        {s.breakeven_ratio.p50:.0%} of sales volume (median)")
    for name, units in s.breakeven_units.items():
        print(f"                    {name}: about {units:,.0f} units per {per}")

    t = tornado(business)
    print(f"\n  What moves profit most (base case {money(t.base_profit)}):")
    for row in t.table.head(5).itertuples():
        print(f"    {row.input:<40} {money(row.profit_at_low):>8} to {money(row.profit_at_high):>8}")

    for plan in business.plans:
        c = compare(business, plan, n=args.scenarios, seed=args.seed)
        d = c.profit_difference
        print(f"\n  Plan B: {plan.name}")
        print(f"    Beats the current plan in {c.prob_b_better:.0%} of scenarios")
        print(f"    Profit change     P10 {money(d.p10)}   median {money(d.p50)}   P90 {money(d.p90)}")
        print(f"    Chance of a loss  {c.a.prob_loss:.1%} -> {c.b.prob_loss:.1%}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
