"""Greedy mixed-precision bit allocation from measured sensitivity profiles.

    python scripts/allocate_bits.py --profile sensitivity_3bit.json \
        --profile2 sensitivity_2bit.json --target-bits 3.0 --out layer_bits.json

Ranks layers by drop/params (damage per byte saved) and demotes the cheapest
ones first until the average bits/weight target is hit. Layers above
--protect-drop are never touched. We use this measured ranking rather than a
Hessian-based proxy since we already have the real per-layer accuracy drop.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def get_args(argv=None):
    p = argparse.ArgumentParser(description="Greedy mixed-precision bit allocation")
    p.add_argument("--profile", required=True,
                   help="sensitivity.json from scripts/sensitivity.py (higher probe bits)")
    p.add_argument("--profile2", default=None,
                   help="Optional second profile at a lower probe bit-width; a layer "
                        "is only demoted to --low-bits if it is tolerant in BOTH.")
    p.add_argument("--low-bits", type=int, default=2)
    p.add_argument("--mid-bits", type=int, default=3)
    p.add_argument("--high-bits", type=int, default=8,
                   help="Reserved for sensitive layers (never demoted).")
    p.add_argument("--target-bits", type=float, default=3.0,
                   help="Target average nominal bits/weight across quantized layers.")
    p.add_argument("--protect-drop", type=float, default=0.5,
                   help="Layers whose measured drop exceeds this keep --high-bits.")
    p.add_argument("--out", default="layer_bits.json")
    return p.parse_args(argv)


def main(argv=None):
    args = get_args(argv)
    prof = json.load(open(args.profile))
    layers = prof["layers"]
    drop2 = {}
    if args.profile2:
        drop2 = {r["layer"]: r["drop"] for r in json.load(open(args.profile2))["layers"]}

    total_params = sum(r["params"] for r in layers)

    # 1) protect the genuinely sensitive layers outright
    protected = [r for r in layers if r["drop"] >= args.protect_drop]
    candidates = [r for r in layers if r["drop"] < args.protect_drop]

    # 2) rank the rest by damage per parameter -- cheapest damage first
    for r in candidates:
        d = max(r["drop"], 0.0)
        if drop2:
            d = max(d, max(drop2.get(r["layer"], 0.0), 0.0))   # must survive both probes
        r["_eff"] = d / max(r["params"], 1)
    candidates.sort(key=lambda r: r["_eff"])

    bits = {r["layer"]: args.high_bits for r in protected}
    for r in candidates:
        bits[r["layer"]] = args.mid_bits

    # 3) demote greedily until the average bits target is reached
    def avg_bits():
        return sum(bits[r["layer"]] * r["params"] for r in layers) / total_params

    demoted = []
    for r in candidates:
        if avg_bits() <= args.target_bits:
            break
        bits[r["layer"]] = args.low_bits
        demoted.append(r)

    print(f"profile: {args.profile} (probe {prof['probe_bits']}-bit, "
          f"reference {prof['reference_acc']:.2f}%)")
    print(f"{len(protected)} layers protected at {args.high_bits}-bit "
          f"({sum(r['params'] for r in protected):,} params)")
    print(f"{len(demoted)} layers demoted to {args.low_bits}-bit "
          f"({sum(r['params'] for r in demoted):,} params, "
          f"{100 * sum(r['params'] for r in demoted) / total_params:.1f}% of weights)")
    print(f"average nominal bits/weight: {avg_bits():.3f}  (target {args.target_bits})")

    print(f"\nprotected ({args.high_bits}-bit):")
    for r in sorted(protected, key=lambda r: -r["drop"])[:10]:
        print(f"  {r['drop']:>6.2f} drop  {r['layer'][:44]:<46}{r['params']:>9,}")
    print(f"\nlargest demoted ({args.low_bits}-bit):")
    for r in sorted(demoted, key=lambda r: -r["params"])[:10]:
        print(f"  {r['drop']:>6.2f} drop  {r['layer'][:44]:<46}{r['params']:>9,}")

    with open(args.out, "w") as f:
        json.dump(bits, f, indent=2)
    print(f"\nwrote {args.out} ({len(bits)} layers)")
    print(f"use with: python scripts/run_qat.py --layer-bits {args.out} ...")
    return bits


if __name__ == "__main__":
    main()
