#!/usr/bin/env python3
"""Parse SFA training log and report val NLL trajectory + position-stratified gains."""
import re, sys, argparse

ap = argparse.ArgumentParser()
ap.add_argument("logfile")
ap.add_argument("--noop-nll", type=float, default=23.4654,
                help="NO-OP @ same swap layer val NLL (control)")
args = ap.parse_args()

vals = []
with open(args.logfile) as f:
    for line in f:
        m = re.search(r'\[step\s*(\d+)\] loss=([\d.]+) ema=([\d.]+)', line)
        if m:
            step = int(m.group(1))
        m = re.search(
            r'\[val val\s*\] nll=([\d.]+).*pos=\[([^\]]+)\]', line)
        if m and step is not None:
            nll = float(m.group(1))
            pos = [float(x) for x in m.group(2).split('/')]
            vals.append((step, nll, pos))

if not vals:
    print(f"No val entries found in {args.logfile}", file=sys.stderr)
    sys.exit(1)

# Header
print(f"NO-OP control NLL: {args.noop_nll:.4f}")
print()
print(f"{'Step':>7} {'NLL':>7} {'Δ_NOOP':>8} | " + " ".join(f"p{i:1d}" for i in range(8)))
print(f"{'─'*7} {'─'*7} {'─'*8} | " + " ".join("─"*4 for _ in range(8)))

# Use first val as reference for position deltas? Or NO-OP control?
# Print absolute position NLLs.
for step, nll, pos in vals:
    delta = nll - args.noop_nll
    pos_str = " ".join(f"{p:4.1f}" for p in pos)
    print(f"{step:7d} {nll:7.4f} {delta:+8.4f} | {pos_str}")

# Compare final to first
if len(vals) >= 2:
    s0, n0, p0 = vals[0]
    s1, n1, p1 = vals[-1]
    print()
    print(f"Δ position NLL (final {s1} vs first {s0}):")
    for i, (a, b) in enumerate(zip(p0, p1)):
        print(f"  pos {i}: {a:5.2f} → {b:5.2f}  Δ {b-a:+.3f}")
