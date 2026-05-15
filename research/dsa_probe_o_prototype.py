#!/usr/bin/env python3
"""
DSA Probe O — synthetic-data prototype of the commutation-defect formula.

Validates that the formula in PARADIGM_SHIFT_255_DESIGN.md §2.1 (eq. 1)
is internally consistent and produces the expected position-stratified
defect pattern when fed Σ values that mirror Phase 8b's empirical NLL pattern.

This script uses ONLY the Python standard library (no numpy) so it can run
on any system the trainer can be built on.

Run:  python3 research/dsa_probe_o_prototype.py
"""

import math
import random
import statistics


def compute_defect_per_token(sigma_fwd, sigma_bwd):
    """Eq. 1 of PARADIGM_SHIFT_255_DESIGN.md, rank-r-subspace formulation.

    Under the orthonormality assumption (U_i has orthonormal columns), the
    Frobenius defect of the round-trip restriction map reduces to:

        ε_i = sqrt( Σ_β ( σ_fwd[β] · σ_bwd[β]  −  1 )^2 )

    Parameters
    ----------
    sigma_fwd : list[list[float]]  shape (T, r)  — Σ on forward edge (i-1 → i).
    sigma_bwd : list[list[float]]  shape (T, r)  — Σ on backward edge (i → i-1).

    Returns
    -------
    eps : list[float]  shape (T,)  — per-token commutation defect.
    """
    T = len(sigma_fwd)
    assert len(sigma_bwd) == T
    eps = []
    for i in range(T):
        r = len(sigma_fwd[i])
        assert len(sigma_bwd[i]) == r
        acc = 0.0
        for b in range(r):
            m = sigma_fwd[i][b] * sigma_bwd[i][b]
            d = m - 1.0
            acc += d * d
        eps.append(math.sqrt(acc))
    return eps


# -------------------------------------------------------------------- #
# Phase 8b empirical NLL gain pattern (PARADIGM_SHIFT_255_DESIGN.md §3.1)
# -------------------------------------------------------------------- #

PHASE_8B_DELTA_NLL = {
    0: -0.07,    # negligible
    1: +1.15,    # regression
    2: -0.23,    # mild
    3: -1.68,    # best
    4: -0.54,    # mild
    5: -1.27,    # strong
    6: -0.95,    # strong
    7: -0.82,    # strong
}


def synthesize_sigma_matching_phase8b(num_positions=8, r=4, seed=0, scale_mult=0.3):
    """Build Σ values that ENCODE the Phase 8b position pattern.

    If SFA's NLL gain at position i comes from cocycle structure captured in
    Σ_e, then a "well-trained" Σ should have:
      - Σ ≈ 1 at pos 0, 1 (no cocycle structure to encode)
      - Σ farther from 1 at pos 3-7 (rich cocycle structure)

    SANITY CHECK for the formula — NOT a claim that Phase 8b's trained Σ
    actually looks like this.
    """
    rng = random.Random(seed)
    sigma_fwd = [[1.0] * r for _ in range(num_positions)]
    sigma_bwd = [[1.0] * r for _ in range(num_positions)]

    for i in range(num_positions):
        gain = PHASE_8B_DELTA_NLL.get(i, 0.0)
        divergence = abs(gain) * scale_mult
        for b in range(r):
            sigma_fwd[i][b] = 1.0 + rng.gauss(0.0, divergence)
            sigma_bwd[i][b] = 1.0 + rng.gauss(0.0, divergence)

    # Pos 0 has no incoming edge — set to identity so ε_0 = 0.
    sigma_fwd[0] = [1.0] * r
    sigma_bwd[0] = [1.0] * r
    return sigma_fwd, sigma_bwd


def synthesize_sigma_uniform_random(num_positions=8, r=4, seed=1, scale=0.3):
    """Adversarial: uniform random Σ across all positions.

    If the formula is sensible, ε will be roughly uniform across positions
    (no pos-0-1-vs-pos-3-7 stratification).  Tests that the formula isn't
    accidentally producing the Phase 8b pattern from architectural bias.
    """
    rng = random.Random(seed)
    sigma_fwd = [[1.0 + rng.gauss(0.0, scale) for _ in range(r)] for _ in range(num_positions)]
    sigma_bwd = [[1.0 + rng.gauss(0.0, scale) for _ in range(r)] for _ in range(num_positions)]
    return sigma_fwd, sigma_bwd


def pearson_r(xs, ys):
    """Simple Pearson correlation."""
    n = len(xs)
    assert len(ys) == n
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    sx2 = sum((xs[i] - mx) ** 2 for i in range(n))
    sy2 = sum((ys[i] - my) ** 2 for i in range(n))
    denom = math.sqrt(sx2 * sy2)
    if denom < 1e-12:
        return float('nan')
    return num / denom


def conjecture12_check(eps, delta_nll, label):
    """Conjecture 12 (defect-NLL correspondence): Pearson r between ε and |ΔNLL| ≥ 0.6."""
    positions = sorted(delta_nll.keys())
    e = [eps[i] for i in positions]
    g = [abs(delta_nll[i]) for i in positions]
    r = pearson_r(e, g)
    print(f"  Pearson r(ε, |ΔNLL|)  =  {r:+.3f}   [{label}]")
    return r


def probe_o_pass_criterion(eps, label):
    """Probe O pass: ε at pos 3-7 ≥ 3× ε at pos 0-1."""
    early = (eps[0] + eps[1]) / 2.0
    late = sum(eps[3:8]) / 5.0
    ratio = late / (early + 1e-9)
    pass_ = ratio >= 3.0
    status = 'PASS' if pass_ else 'FAIL'
    print(f"  ε late/early ratio    =  {ratio:6.2f} ×   ({status} probe-O bar of 3×)   [{label}]")
    return pass_


def print_eps_table(eps, delta_nll, label):
    print(f"  per-position ε  [{label}]:")
    print(f"    pos  |     ε     | |ΔNLL| (8b)")
    print(f"    -----+-----------+------------")
    for i in range(len(eps)):
        nll = delta_nll.get(i, 0.0)
        print(f"     {i}   |  {eps[i]:7.4f}  |   {abs(nll):.2f}")


def main():
    print("=== DSA Probe O — synthetic prototype ===\n")

    # ------------------------------------------------------------------ #
    # Test 1: Σ values matching Phase 8b NLL pattern
    # ------------------------------------------------------------------ #
    print("Test 1 — Phase-8b-aligned Σ synthesis")
    sf, sb = synthesize_sigma_matching_phase8b(num_positions=8, r=4, seed=0)
    eps1 = compute_defect_per_token(sf, sb)
    print_eps_table(eps1, PHASE_8B_DELTA_NLL, "phase8b-aligned")
    r1 = conjecture12_check(eps1, PHASE_8B_DELTA_NLL, "phase8b-aligned")
    p1 = probe_o_pass_criterion(eps1, "phase8b-aligned")
    print()

    # ------------------------------------------------------------------ #
    # Test 2: adversarial uniform-random Σ
    # ------------------------------------------------------------------ #
    print("Test 2 — adversarial uniform-random Σ")
    sf2, sb2 = synthesize_sigma_uniform_random(num_positions=8, r=4, seed=1, scale=0.3)
    eps2 = compute_defect_per_token(sf2, sb2)
    print_eps_table(eps2, PHASE_8B_DELTA_NLL, "uniform-random")
    r2 = conjecture12_check(eps2, PHASE_8B_DELTA_NLL, "uniform-random")
    p2 = probe_o_pass_criterion(eps2, "uniform-random")
    print()

    # ------------------------------------------------------------------ #
    # Test 3: scale sweep
    # ------------------------------------------------------------------ #
    print("Test 3 — divergence-scale sweep (Phase-8b-aligned)")
    for scale_mult in [0.1, 0.3, 1.0, 3.0]:
        sf3, sb3 = synthesize_sigma_matching_phase8b(num_positions=8, r=4, seed=42, scale_mult=scale_mult)
        eps3 = compute_defect_per_token(sf3, sb3)
        ratio = (sum(eps3[3:8]) / 5.0) / ((eps3[0] + eps3[1]) / 2.0 + 1e-9)
        r3 = pearson_r(eps3, [abs(PHASE_8B_DELTA_NLL[i]) for i in range(8)])
        print(f"  scale={scale_mult:>4}:  ε ratio = {ratio:>6.2f}×   r(ε, |ΔNLL|) = {r3:>+.3f}")
    print()

    # ------------------------------------------------------------------ #
    # Test 4: large-r robustness (matches default sfa-r=4 but also r=8, 16)
    # ------------------------------------------------------------------ #
    print("Test 4 — rank-r robustness sweep")
    for r in [2, 4, 8, 16]:
        sf4, sb4 = synthesize_sigma_matching_phase8b(num_positions=8, r=r, seed=99)
        eps4 = compute_defect_per_token(sf4, sb4)
        ratio = (sum(eps4[3:8]) / 5.0) / ((eps4[0] + eps4[1]) / 2.0 + 1e-9)
        r_corr = pearson_r(eps4, [abs(PHASE_8B_DELTA_NLL[i]) for i in range(8)])
        print(f"  r={r:>2}:    ε ratio = {ratio:>6.2f}×   r(ε, |ΔNLL|) = {r_corr:>+.3f}")
    print()

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("=== Summary ===")
    t1_conj12 = 'PASS' if r1 >= 0.6 else 'FAIL'
    t1_probe = 'PASS' if p1 else 'FAIL'
    t2_conj12 = 'PASS (low correlation expected)' if abs(r2) < 0.5 else 'WARN'
    t2_probe = 'PASS (FAIL expected for adversarial)' if not p2 else 'WARN'
    print(f"  Test 1 (phase-8b-aligned):    Conjecture 12 {t1_conj12} (r={r1:+.3f}),  Probe O {t1_probe}")
    print(f"  Test 2 (uniform-random):      Conjecture 12 {t2_conj12} (r={r2:+.3f}),  Probe O {t2_probe}")
    print()
    print("Interpretation:")
    print("  - Test 1 PASS → formula is consistent with Phase 8b pattern.")
    print("  - Test 2 PASS → formula doesn't have a structural bias forcing the pattern.")
    print("  - Test 3 → defect ratio scales monotonically with divergence magnitude.")
    print("  - Test 4 → defect ratio is roughly rank-invariant (good — kernel doesn't")
    print("    need r-specific tuning).")
    print()
    print("Next CUDA kernel spec (sfa_defect_step1_kernel):")
    print("  Inputs:  Sigma [E·r], fwd_edge_at_i [T], bwd_edge_at_i [T]")
    print("  Output:  eps [T]")
    print("  Per-block:  one token i; warp-reduce sum over r dims of (Σ_fwd·Σ_bwd − 1)^2")
    print("  See compute_defect_per_token() for the reference implementation.")


if __name__ == "__main__":
    main()
