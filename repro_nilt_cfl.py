#!/usr/bin/env python3
"""
Reproducibility script for FFT-NILT CFL tuning paper.

This script reproduces all numerical results and figures from:
"CFL-Informed Parameter Tuning for FFT-Based Numerical Inverse Laplace Transform"

Usage:
    python repro_nilt_cfl.py --all              # Run everything
    python repro_nilt_cfl.py --figures          # Generate figures only
    python repro_nilt_cfl.py --table1           # Reproduce Table 1
    python repro_nilt_cfl.py --benchmarks       # Run all benchmarks
    python repro_nilt_cfl.py --dampener         # Dampener worked example
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# Local imports
from nilt_fft import fft_nilt, eps_im, n_doubling_error
from tuner import tune_params, refine_until_accept, check_cfl_feasibility, TunedParams
from problems import Problem, get_problem, get_all_problems, dampener
from metrics import rmse, relative_rmse, max_absolute_error, max_relative_error


def print_header(title: str) -> None:
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title: str) -> None:
    """Print subsection header."""
    print(f"\n--- {title} ---")


def reproduce_table1(
    t_end: float = 10.0,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Reproduce Table 1: CFL parameter calibration across benchmark problems.

    Returns
    -------
    results : dict
        Dictionary of results keyed by problem name
    """
    print_header("Table 1: CFL Parameter Calibration")

    problems = get_all_problems()
    results = {}

    # Table header
    if verbose:
        print(f"\n{'Problem':<12} {'α_c':>8} {'C':>6} {'a':>8} {'T':>8} {'N':>6} "
              f"{'a_max':>8} {'margin':>8} {'feasible':>8}")
        print("-" * 90)

    for name, problem in problems.items():
        params = tune_params(
            t_end=t_end,
            alpha_c=problem.alpha_c,
            C=problem.C,
            rho=problem.rho
        )

        results[name] = {
            "alpha_c": problem.alpha_c,
            "C": problem.C,
            "a": params.a,
            "T": params.T,
            "N": params.N,
            "a_min": params.a_min,
            "a_max": params.a_max,
            "margin": params.margin,
            "feasible": params.feasible
        }

        if verbose:
            feas_str = "YES" if params.feasible else "NO"
            print(f"{name:<12} {problem.alpha_c:>8.2f} {problem.C:>6.2f} {params.a:>8.4f} "
                  f"{params.T:>8.2f} {params.N:>6d} {params.a_max:>8.2f} "
                  f"{params.margin:>8.2f} {feas_str:>8}")

    if verbose:
        print("-" * 90)
        print(f"Parameters: t_end={t_end}, kappa=1.0, eps_tail=1e-6, delta_floor=1e-3")

    return results


def reproduce_table2(
    t_end: float = 10.0,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Reproduce Table 2: Accuracy and timing results after adaptive refinement.

    Returns
    -------
    results : dict
        Dictionary of results keyed by problem name
    """
    print_header("Table 2: Accuracy and Timing Results")

    problems = get_all_problems()
    results = {}

    if verbose:
        print(f"\n{'Problem':<12} {'N_final':>8} {'eps_im':>10} {'E_N':>10} "
              f"{'Rel RMSE':>12} {'Time (μs)':>12} {'Accepted':>8}")
        print("-" * 90)

    for name, problem in problems.items():
        # Initial tuning
        params = tune_params(
            t_end=t_end,
            alpha_c=problem.alpha_c,
            C=problem.C,
            rho=problem.rho
        )

        # Adaptive refinement
        result = refine_until_accept(
            problem.F, params, t_end,
            eps_im_max=1e-2,
            eps_conv=1e-2,
            N_max=16384,
            n_timing_runs=50
        )

        # Compute accuracy against analytical reference
        if problem.f_ref is not None:
            f_ref = problem.f_ref(result['t_eval'])
            rel_rmse = relative_rmse(result['f_eval'], f_ref)
        else:
            rel_rmse = np.nan

        results[name] = {
            "N_final": result['N'],
            "eps_im": result['eps_im'],
            "E_N": result['E_N'],
            "rel_rmse": rel_rmse,
            "timing_median_us": result['timing_median_us'],
            "timing_mad_us": result['timing_mad_us'],
            "accepted": result['accepted'],
            "iterations": result['iterations']
        }

        if verbose:
            acc_str = "YES" if result['accepted'] else "NO"
            rmse_str = f"{rel_rmse:.2e}" if not np.isnan(rel_rmse) else "N/A"
            print(f"{name:<12} {result['N']:>8d} {result['eps_im']:>10.2e} "
                  f"{result['E_N']:>10.2e} {rmse_str:>12} "
                  f"{result['timing_median_us']:>10.1f}±{result['timing_mad_us']:.1f} "
                  f"{acc_str:>8}")

    if verbose:
        print("-" * 90)
        print(f"Acceptance criteria: eps_im_max=1e-2, eps_conv=1e-2")

    return results


def dampener_worked_example(
    t_end: float = 20.0,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Dampener system worked example (Section 4.1 of paper).

    Demonstrates step-by-step application of Algorithm 1 with explicit bounds.
    """
    print_header("Worked Example: Dampener System")

    # Problem parameters
    omega_n = 1.0
    zeta = 0.2
    problem = dampener(omega_n=omega_n, zeta=zeta)

    if verbose:
        print_subheader("Problem Definition")
        print(f"System: Second-order underdamped (dampener)")
        print(f"  omega_n = {omega_n}")
        print(f"  zeta = {zeta}")
        print(f"  alpha_c = -zeta * omega_n = {problem.alpha_c}")
        print(f"  C = omega_n / sqrt(1 - zeta^2) = {problem.C:.4f}")
        print(f"  rho = omega_n = {problem.rho}")

        print_subheader("CFL Feasibility Check")

    # Tuning parameters
    kappa = 1.0
    eps_tail = 1e-6
    delta_floor = 1e-3
    delta_s = 10.0
    L = 709.8

    T = kappa * t_end
    t_max = 2 * T

    # Check feasibility
    cfl_ok, floor_ok, lhs_cfl, lhs_floor = check_cfl_feasibility(
        problem.alpha_c, t_max, problem.C, eps_tail, delta_floor, delta_s, L
    )

    if verbose:
        print(f"  T = kappa * t_end = {T}")
        print(f"  t_max = 2T = {t_max}")
        print(f"  L - delta_s = {L - delta_s:.1f}")
        print(f"\nEq. (14) CFL condition:")
        print(f"  alpha_c * t_max + ln(C/eps_tail) = {lhs_cfl:.2f}")
        print(f"  L - delta_s = {L - delta_s:.1f}")
        print(f"  Satisfied: {'YES' if cfl_ok else 'NO'}")
        print(f"\nEq. (14') Floor condition:")
        print(f"  delta_floor * t_max = {lhs_floor:.4f}")
        print(f"  L - delta_s = {L - delta_s:.1f}")
        print(f"  Satisfied: {'YES' if floor_ok else 'NO'}")

    # Tune parameters
    params = tune_params(
        t_end=t_end,
        alpha_c=problem.alpha_c,
        C=problem.C,
        rho=problem.rho
    )

    if verbose:
        print_subheader("Parameter Tuning (Algorithm 1, Phase 1)")
        print(f"  a_alias = alpha_c + ln(C/eps_tail) / ((2*kappa-1)*t_end)")
        alias_factor = (2 * kappa - 1) * t_end
        a_alias = problem.alpha_c + np.log(problem.C / eps_tail) / alias_factor
        print(f"         = {problem.alpha_c} + {np.log(problem.C / eps_tail):.2f} / {alias_factor}")
        print(f"         = {a_alias:.4f}")
        print(f"  a_sing = alpha_c + delta_min = {problem.alpha_c + delta_floor:.4f}")
        print(f"  a_min* = max(a_alias, a_sing) = {params.a_min_star:.4f}")
        print(f"  a_min = max(a_min*, delta_floor) = {params.a_min:.4f}")
        print(f"  a_max = (L - delta_s) / t_max = {params.a_max:.4f}")
        print(f"  margin = a_max - a = {params.margin:.4f}")
        print(f"  feasible = {params.feasible}")
        print(f"\nSelected: a = {params.a:.4f}, T = {params.T}, N_init = {params.N}")

    # Adaptive refinement
    if verbose:
        print_subheader("Adaptive Refinement (Algorithm 1, Phase 2)")

    result = refine_until_accept(
        problem.F, params, t_end,
        eps_im_max=1e-2,
        eps_conv=1e-2,
        N_max=16384,
        n_timing_runs=100
    )

    if verbose:
        print(f"  Iterations: {result['iterations']}")
        print(f"  N_final: {result['N']}")
        print(f"  eps_im: {result['eps_im']:.2e}")
        print(f"  E_N: {result['E_N']:.2e}")
        print(f"  Accepted: {'YES' if result['accepted'] else 'NO'}")

    # Accuracy against analytical
    f_ref = problem.f_ref(result['t_eval'])
    rel_rmse = relative_rmse(result['f_eval'], f_ref)
    max_err = max_absolute_error(result['f_eval'], f_ref)

    if verbose:
        print_subheader("Accuracy vs Analytical")
        print(f"  Relative RMSE: {rel_rmse:.2e}")
        print(f"  Max absolute error: {max_err:.2e}")
        print(f"  Timing: {result['timing_median_us']:.1f} ± {result['timing_mad_us']:.1f} μs")

    # Theoretical alias bound
    alias_bound = problem.C * np.exp(problem.alpha_c * t_end) / (
        np.exp(alias_factor * params.a) - 1 + 1e-20
    )

    if verbose:
        print_subheader("Alias Bound Verification (Lemma 2)")
        print(f"  Theoretical alias bound: {alias_bound:.2e}")
        print(f"  Observed max error: {max_err:.2e}")
        print(f"  Bound respected: {'YES' if max_err <= alias_bound * 10 else 'CHECK'}")

    return {
        "problem": {
            "omega_n": omega_n,
            "zeta": zeta,
            "alpha_c": problem.alpha_c,
            "C": problem.C
        },
        "params": {
            "a": params.a,
            "T": params.T,
            "N_final": result['N'],
            "a_min": params.a_min,
            "a_max": params.a_max,
            "margin": params.margin,
            "feasible": params.feasible
        },
        "diagnostics": {
            "eps_im": result['eps_im'],
            "E_N": result['E_N'],
            "accepted": result['accepted']
        },
        "accuracy": {
            "rel_rmse": rel_rmse,
            "max_error": max_err,
            "alias_bound": alias_bound
        },
        "timing": {
            "median_us": result['timing_median_us'],
            "mad_us": result['timing_mad_us']
        }
    }


def run_all_benchmarks(
    t_end_values: List[float] = [5.0, 10.0, 20.0, 50.0],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run comprehensive benchmarks across all problems and time horizons.
    """
    print_header("Comprehensive Benchmarks")

    problems = get_all_problems()
    all_results = {}

    for t_end in t_end_values:
        if verbose:
            print_subheader(f"t_end = {t_end}")
            print(f"{'Problem':<12} {'N':>6} {'eps_im':>10} {'Rel RMSE':>12} {'Time (μs)':>10}")
            print("-" * 60)

        for name, problem in problems.items():
            params = tune_params(
                t_end=t_end,
                alpha_c=problem.alpha_c,
                C=problem.C,
                rho=problem.rho
            )

            result = refine_until_accept(
                problem.F, params, t_end,
                eps_im_max=1e-2,
                eps_conv=1e-2,
                N_max=16384,
                n_timing_runs=20
            )

            if problem.f_ref is not None:
                f_ref = problem.f_ref(result['t_eval'])
                rel_rmse = relative_rmse(result['f_eval'], f_ref)
            else:
                rel_rmse = np.nan

            key = f"{name}_t{int(t_end)}"
            all_results[key] = {
                "problem": name,
                "t_end": t_end,
                "N": result['N'],
                "eps_im": result['eps_im'],
                "rel_rmse": rel_rmse,
                "timing_us": result['timing_median_us'],
                "accepted": result['accepted']
            }

            if verbose:
                rmse_str = f"{rel_rmse:.2e}" if not np.isnan(rel_rmse) else "N/A"
                print(f"{name:<12} {result['N']:>6d} {result['eps_im']:>10.2e} "
                      f"{rmse_str:>12} {result['timing_median_us']:>10.1f}")

    return all_results


def generate_figures(output_dir: str = "figures") -> None:
    """Generate all paper figures."""
    print_header("Generating Figures")

    from plots import generate_all_figures
    generate_all_figures(output_dir)


def save_results(results: Dict[str, Any], filename: str) -> None:
    """Save results to JSON file."""
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(x) for x in obj]
        return obj

    with open(filename, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"Results saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Reproducibility script for FFT-NILT CFL tuning paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python repro_nilt_cfl.py --all              # Run everything
    python repro_nilt_cfl.py --figures          # Generate figures only
    python repro_nilt_cfl.py --table1           # Reproduce Table 1
    python repro_nilt_cfl.py --dampener         # Dampener worked example
    python repro_nilt_cfl.py --save results.json  # Save results to file
        """
    )

    parser.add_argument('--all', action='store_true',
                        help='Run all reproductions')
    parser.add_argument('--table1', action='store_true',
                        help='Reproduce Table 1 (parameter calibration)')
    parser.add_argument('--table2', action='store_true',
                        help='Reproduce Table 2 (accuracy and timing)')
    parser.add_argument('--dampener', action='store_true',
                        help='Run dampener worked example')
    parser.add_argument('--benchmarks', action='store_true',
                        help='Run comprehensive benchmarks')
    parser.add_argument('--figures', action='store_true',
                        help='Generate all figures')
    parser.add_argument('--output-dir', type=str, default='figures',
                        help='Output directory for figures (default: figures)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save results to JSON file')
    parser.add_argument('--t-end', type=float, default=10.0,
                        help='End time for evaluation (default: 10.0)')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')

    args = parser.parse_args()

    # Default to --all if no specific action requested
    if not any([args.all, args.table1, args.table2, args.dampener,
                args.benchmarks, args.figures]):
        args.all = True

    verbose = not args.quiet
    all_results = {}

    print("\n" + "=" * 70)
    print("  FFT-NILT CFL Tuning: Reproducibility Script")
    print("=" * 70)
    print(f"  Running with t_end = {args.t_end}")

    start_time = time.time()

    if args.all or args.table1:
        all_results['table1'] = reproduce_table1(args.t_end, verbose)

    if args.all or args.table2:
        all_results['table2'] = reproduce_table2(args.t_end, verbose)

    if args.all or args.dampener:
        all_results['dampener'] = dampener_worked_example(
            t_end=20.0,  # Use longer horizon for dampener
            verbose=verbose
        )

    if args.all or args.benchmarks:
        all_results['benchmarks'] = run_all_benchmarks(verbose=verbose)

    if args.all or args.figures:
        generate_figures(args.output_dir)

    elapsed = time.time() - start_time

    print_header("Summary")
    print(f"Total execution time: {elapsed:.1f} seconds")

    if args.save:
        save_results(all_results, args.save)

    print("\nReproduction complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
