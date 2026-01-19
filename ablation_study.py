"""
Ablation Study: Demonstrating CFL Constraint Importance

This script demonstrates what happens when individual CFL constraints
are removed or violated:

1. Remove feasibility bound → overflow failures (exp(at) exceeds DBL_MAX)
2. Remove aliasing constraint → wraparound artifacts (periodic copies leak in)
3. Remove ε_Im check → undetected bad results

Also includes ε_Im calibration analysis:
- Confusion matrix: ε_Im pass/fail vs actual error
- Counterexamples where ε_Im alone fails
- N-doubling as complementary diagnostic

Author: Gorgi Pavlov
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from problems import get_problem, first_order_lag, fopdt, semi_infinite_diffusion
from nilt_fft import fft_nilt, eps_im
from tuner import tune_params, check_cfl_feasibility


@dataclass
class AblationResult:
    """Results from ablation test."""
    test_name: str
    condition_violated: str
    a: float
    T: float
    N: int
    rmse: float
    max_error: float
    eps_im: float
    overflow_detected: bool
    warning_message: str


def test_remove_feasibility_bound(
    problem_name: str = "diffusion",
    t_end: float = 10.0,
    verbose: bool = True
) -> AblationResult:
    """
    Test: What happens without feasibility bound check?

    The feasibility condition ensures a*t_max < L - δ_s (about 700 for float64)
    to prevent overflow in exp(a*t).

    We deliberately set a too large and show overflow.
    """
    problem = get_problem(problem_name)

    # Get normal (feasible) parameters
    tuned_ok = tune_params(t_end=t_end, alpha_c=problem.alpha_c, C=problem.C)

    # Now violate: set a >> a_max
    a_bad = tuned_ok.a_max * 2  # Double the maximum

    if verbose:
        print(f"\n[ABLATION 1] Remove feasibility bound")
        print(f"  Problem: {problem_name}")
        print(f"  Normal a_max = {tuned_ok.a_max:.2f}")
        print(f"  Violated a = {a_bad:.2f} (2x a_max)")

    T = tuned_ok.T
    N = 512

    # Try to compute
    overflow = False
    try:
        f, t, z = fft_nilt(problem.F, a_bad, T, N)

        # Check for overflow/inf
        if np.any(~np.isfinite(f)):
            overflow = True
            rmse = np.inf
            max_err = np.inf
            current_eps_im = np.inf
            warning = "OVERFLOW: exp(at) exceeded float64 range"
        else:
            # Compute metrics even if bad
            if problem.f_ref is not None:
                mask = (t > 0.1) & (t <= t_end)
                f_ref = problem.f_ref(t)
                rmse = np.sqrt(np.mean((f[mask] - f_ref[mask])**2))
                max_err = np.max(np.abs(f[mask] - f_ref[mask]))
            else:
                rmse = np.nan
                max_err = np.nan
            current_eps_im = eps_im(z)
            warning = f"Large errors: RMSE = {rmse:.2e}"

    except (OverflowError, FloatingPointError) as e:
        overflow = True
        rmse = np.inf
        max_err = np.inf
        current_eps_im = np.inf
        warning = f"EXCEPTION: {str(e)}"

    if verbose:
        print(f"  Overflow detected: {overflow}")
        print(f"  Warning: {warning}")

    return AblationResult(
        test_name="remove_feasibility_bound",
        condition_violated="a*t_max > L - δ_s",
        a=a_bad,
        T=T,
        N=N,
        rmse=rmse,
        max_error=max_err,
        eps_im=current_eps_im,
        overflow_detected=overflow,
        warning_message=warning
    )


def test_remove_aliasing_constraint(
    problem_name: str = "lag",
    t_end: float = 10.0,
    verbose: bool = True
) -> AblationResult:
    """
    Test: What happens without aliasing constraint?

    The aliasing constraint ensures a > α_c + ln(C/ε_tail)/(2T-t_end)
    to suppress wraparound from periodic extension.

    We deliberately set a close to α_c (violating the aliasing margin).
    """
    problem = get_problem(problem_name)

    # Get normal parameters
    tuned_ok = tune_params(t_end=t_end, alpha_c=problem.alpha_c, C=problem.C)

    # Violate: set a very close to α_c (no aliasing margin)
    a_bad = problem.alpha_c + 0.001  # Barely above α_c, no aliasing suppression

    if verbose:
        print(f"\n[ABLATION 2] Remove aliasing constraint")
        print(f"  Problem: {problem_name}")
        print(f"  α_c = {problem.alpha_c:.4f}")
        print(f"  Normal a = {tuned_ok.a:.4f} (includes aliasing margin)")
        print(f"  Violated a = {a_bad:.4f} (just above α_c)")

    T = tuned_ok.T
    N = 512

    # Compute with bad parameters
    f, t, z = fft_nilt(problem.F, a_bad, T, N)

    # Compute error metrics
    if problem.f_ref is not None:
        mask = (t > 0.1) & (t <= t_end)
        f_ref = problem.f_ref(t)
        rmse = np.sqrt(np.mean((f[mask] - f_ref[mask])**2))
        max_err = np.max(np.abs(f[mask] - f_ref[mask]))
    else:
        rmse = np.nan
        max_err = np.nan

    current_eps_im = eps_im(z)

    # Compare with normal
    f_ok, t_ok, z_ok = fft_nilt(problem.F, tuned_ok.a, T, N)
    if problem.f_ref is not None:
        rmse_ok = np.sqrt(np.mean((f_ok[mask] - f_ref[mask])**2))
    else:
        rmse_ok = np.nan

    if verbose:
        print(f"  RMSE (violated): {rmse:.2e}")
        print(f"  RMSE (normal):   {rmse_ok:.2e}")
        print(f"  Error increase:  {rmse/rmse_ok:.1f}x")
        print(f"  Aliasing artifact visible as increased error near t → 2T")

    return AblationResult(
        test_name="remove_aliasing_constraint",
        condition_violated="a ≈ α_c (no aliasing margin)",
        a=a_bad,
        T=T,
        N=N,
        rmse=rmse,
        max_error=max_err,
        eps_im=current_eps_im,
        overflow_detected=False,
        warning_message=f"Aliasing error increased by {rmse/rmse_ok:.1f}x"
    )


def test_eps_im_calibration(
    problem_name: str = "secondorder",
    t_end: float = 10.0,
    N_values: List[int] = None,
    eps_im_threshold: float = 0.1,
    error_threshold: float = 0.01,
    verbose: bool = True
) -> Dict:
    """
    Calibrate ε_Im as diagnostic across parameter settings.

    Creates confusion matrix:
    - True Positive: ε_Im > threshold AND RMSE > error_threshold
    - False Positive: ε_Im > threshold AND RMSE < error_threshold (false alarm)
    - True Negative: ε_Im < threshold AND RMSE < error_threshold
    - False Negative: ε_Im < threshold AND RMSE > error_threshold (missed error!)

    The key concern is False Negatives: cases where ε_Im says "OK" but
    the actual error is bad.
    """
    if N_values is None:
        N_values = [64, 128, 256, 512, 1024, 2048]

    problem = get_problem(problem_name)

    # Sweep multiple a values around α_c
    tuned = tune_params(t_end=t_end, alpha_c=problem.alpha_c, C=problem.C)

    # a values from barely feasible to well-above minimum
    a_values = np.linspace(tuned.a_min * 0.5, tuned.a * 2, 10)
    a_values = a_values[(a_values > problem.alpha_c) & (a_values < tuned.a_max)]

    T = tuned.T

    results = []
    for a in a_values:
        for N in N_values:
            f, t, z = fft_nilt(problem.F, a, T, N)

            if problem.f_ref is not None:
                mask = (t > 0.1) & (t <= t_end)
                f_ref = problem.f_ref(t)
                rmse = np.sqrt(np.mean((f[mask] - f_ref[mask])**2))
            else:
                rmse = np.nan

            current_eps_im = eps_im(z[mask])

            results.append({
                'a': a,
                'N': N,
                'rmse': rmse,
                'eps_im': current_eps_im,
                'eps_im_pass': current_eps_im < eps_im_threshold,
                'error_pass': rmse < error_threshold
            })

    # Build confusion matrix
    TP = sum(1 for r in results if not r['eps_im_pass'] and not r['error_pass'])
    FP = sum(1 for r in results if not r['eps_im_pass'] and r['error_pass'])
    TN = sum(1 for r in results if r['eps_im_pass'] and r['error_pass'])
    FN = sum(1 for r in results if r['eps_im_pass'] and not r['error_pass'])

    if verbose:
        print(f"\n[ABLATION 3] ε_Im Calibration")
        print(f"  Problem: {problem_name}")
        print(f"  ε_Im threshold: {eps_im_threshold}")
        print(f"  Error threshold: {error_threshold}")
        print(f"  Total tests: {len(results)}")
        print(f"\n  Confusion Matrix:")
        print(f"                    Actual Good    Actual Bad")
        print(f"  ε_Im Pass        {TN:>8}       {FN:>8} (FALSE NEGATIVE!)")
        print(f"  ε_Im Fail        {FP:>8}       {TP:>8}")
        print(f"\n  False Negative Rate: {100*FN/(FN+TP) if (FN+TP) > 0 else 0:.1f}%")
        print(f"  (Cases where ε_Im misses actual errors)")

        if FN > 0:
            print(f"\n  Counterexamples (ε_Im OK but high error):")
            fn_cases = [r for r in results if r['eps_im_pass'] and not r['error_pass']]
            for r in fn_cases[:3]:
                print(f"    a={r['a']:.3f}, N={r['N']}, ε_Im={r['eps_im']:.2e}, RMSE={r['rmse']:.2e}")

    return {
        'confusion_matrix': {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN},
        'false_negative_rate': FN / (FN + TP) if (FN + TP) > 0 else 0,
        'results': results
    }


def test_n_doubling_catches_failures(
    problem_name: str = "diffusion",
    t_end: float = 10.0,
    verbose: bool = True
) -> Dict:
    """
    Demonstrate that N-doubling catches errors that ε_Im may miss.

    N-doubling compares f_N with f_{2N}. Large differences indicate
    insufficient N, regardless of ε_Im value.
    """
    problem = get_problem(problem_name)
    tuned = tune_params(t_end=t_end, alpha_c=problem.alpha_c, C=problem.C)
    a, T = tuned.a, tuned.T

    if verbose:
        print(f"\n[ABLATION 4] N-doubling catches missed errors")
        print(f"  Problem: {problem_name}")
        print("-" * 50)
        print(f"  {'N':>6} {'ε_Im':>12} {'RMSE':>12} {'E_N (N-dbl)':>12}")

    results = []
    for N in [32, 64, 128, 256, 512, 1024, 2048]:
        f, t, z = fft_nilt(problem.F, a, T, N)
        f2, t2, z2 = fft_nilt(problem.F, a, T, 2 * N)

        # Interpolate to compare
        mask = (t > 0.1) & (t <= t_end)
        f_interp = np.interp(t[mask], t2, f2)
        E_N = np.sqrt(np.mean((f[mask] - f_interp)**2)) / np.sqrt(np.mean(f_interp**2))

        if problem.f_ref is not None:
            f_ref = problem.f_ref(t)
            rmse = np.sqrt(np.mean((f[mask] - f_ref[mask])**2))
        else:
            rmse = np.nan

        current_eps_im = eps_im(z[mask])

        results.append({
            'N': N,
            'eps_im': current_eps_im,
            'rmse': rmse,
            'E_N': E_N
        })

        if verbose:
            print(f"  {N:>6} {current_eps_im:>12.2e} {rmse:>12.2e} {E_N:>12.2e}")

    if verbose:
        print(f"\n  Note: E_N (N-doubling) provides convergence information")
        print(f"  that is independent of ε_Im.")

    return results


def run_ablation_study(verbose: bool = True) -> Dict:
    """
    Run complete ablation study.

    Demonstrates importance of each CFL constraint and the role of diagnostics.
    """
    if verbose:
        print("=" * 60)
        print("ABLATION STUDY: CFL Constraint Importance")
        print("=" * 60)

    results = {}

    # Test 1: Remove feasibility bound
    results['feasibility'] = test_remove_feasibility_bound(verbose=verbose)

    # Test 2: Remove aliasing constraint
    results['aliasing'] = test_remove_aliasing_constraint(verbose=verbose)

    # Test 3: ε_Im calibration
    results['eps_im_calibration'] = test_eps_im_calibration(verbose=verbose)

    # Test 4: N-doubling as complementary diagnostic
    results['n_doubling'] = test_n_doubling_catches_failures(verbose=verbose)

    if verbose:
        print("\n" + "=" * 60)
        print("ABLATION STUDY SUMMARY")
        print("=" * 60)
        print("\n1. Feasibility bound: CRITICAL")
        print("   Without it: overflow failures, inf values")
        print(f"   Result: {results['feasibility'].warning_message}")

        print("\n2. Aliasing constraint: IMPORTANT")
        print("   Without it: wraparound artifacts, increased error")
        print(f"   Result: {results['aliasing'].warning_message}")

        print("\n3. ε_Im diagnostic: USEFUL BUT NOT SUFFICIENT")
        print(f"   False negative rate: {100*results['eps_im_calibration']['false_negative_rate']:.1f}%")
        print("   Must combine with N-doubling for robust quality assessment")

        print("\n4. N-doubling: ROBUST CONVERGENCE CHECK")
        print("   Catches truncation errors independently of ε_Im")
        print("   Should be used in combination with ε_Im")

    return results


if __name__ == "__main__":
    results = run_ablation_study(verbose=True)
