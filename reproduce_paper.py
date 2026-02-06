#!/usr/bin/env python
"""
Master Reproducibility Script for Paper 2: FFT-NILT with CFL Tuning

This script reproduces all figures and tables from the paper.
Run with: python reproduce_paper.py

Author: Gorgi Pavlov
"""

from __future__ import annotations
import sys
import os
import json
from pathlib import Path

# Ensure we can import local modules
sys.path.insert(0, os.path.dirname(__file__))

def print_header(text: str):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60 + "\n")


def reproduce_figures():
    """Generate all paper figures."""
    print_header("Generating Paper Figures")

    from generate_paper_figures import generate_all_figures
    generate_all_figures(output_dir="figures")


def reproduce_table5():
    """Reproduce Table 5: Accuracy comparison (default vs CFL-informed)."""
    print_header("Table 5: Accuracy Comparison (CFL-informed vs Default)")

    from repro_nilt_cfl import reproduce_table2
    from nilt_fft import fft_nilt as fft_nilt_numpy
    from problems import get_all_problems
    from metrics import relative_rmse
    import numpy as np

    # CFL-informed results
    cfl_results = reproduce_table2(t_end=10.0, verbose=True)

    # Default parameter results for comparison
    problems = get_all_problems()
    default_a, default_T, default_N = 1.0, 10.0, 512

    print("\nTable 5 data:")
    print(f"{'Problem':<20} {'Default RMSE':>15} {'CFL RMSE':>15} {'Improvement':>12}")
    print("-" * 65)

    for name, problem in problems.items():
        if problem.f_ref is None:
            continue
        # Default parameters
        f_def, t_def, _ = fft_nilt_numpy(problem.F, default_a, default_T, default_N)
        f_ref_def = problem.f_ref(t_def)
        default_rmse = relative_rmse(f_def, f_ref_def)

        # CFL result
        cfl_rmse = cfl_results[name].get('rel_rmse', float('nan'))

        if default_rmse > 0 and cfl_rmse > 0 and not np.isnan(cfl_rmse):
            improvement = default_rmse / cfl_rmse
            print(f"{name:<20} {default_rmse:>15.2e} {cfl_rmse:>15.2e} {improvement:>12.1f}x")


def reproduce_table6():
    """Reproduce Table 6: FFT-NILT vs de Hoog comparison."""
    print_header("Table 6: FFT-NILT vs de Hoog Comparison")

    from benchmark_pareto import benchmark_fft_nilt, benchmark_de_hoog

    fft_results = benchmark_fft_nilt("lag", [256, 512, 2048], t_end=10.0)
    dehoog_results = benchmark_de_hoog("lag", [20], t_end=10.0)

    print("\nTable 6 data:")
    print(f"{'Method':<20} {'RMSE':>15} {'Time (μs)':>12} {'Rel. Speed':>12}")
    print("-" * 62)

    # de Hoog baseline
    dh = dehoog_results[0]
    print(f"de Hoog (M=20)       {dh.rmse:>15.2e} {dh.timing_us:>12.0f}       1.0x")

    for r in fft_results:
        rel_speed = dh.timing_us / r.timing_us
        print(f"FFT-NILT (N={r.N:<4})    {r.rmse:>15.2e} {r.timing_us:>12.0f}   {rel_speed:>8.1f}x")


def reproduce_table8():
    """Reproduce Table 8: GPU benchmark results."""
    print_header("Table 8: Three-Way GPU Benchmark (NumPy vs JAX vs PyTorch)")

    results_file = Path("results_three_way.json")
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)

        print("\nTable 8 data (from cached results):")
        print(f"{'N':<8} {'NumPy (μs)':>12} {'JAX (μs)':>12} {'PyTorch (μs)':>14}")
        print("-" * 50)

        for result in results.get('results', []):
            print(f"{result['N']:<8} {result['numpy_median']:>12.1f} "
                  f"{result['jax_median']:>12.1f} {result['torch_median']:>14.1f}")
    else:
        print("Run benchmark_numpy_jax_torch.py first to generate results.")
        print("Note: Requires GPU with CUDA for meaningful results.")


def reproduce_case_study():
    """Reproduce case study results."""
    print_header("Case Study: Fixed-Bed Adsorption Breakthrough")

    from case_study_adsorption import run_case_study_demo
    results = run_case_study_demo(verbose=True)

    print("\nCase study complete.")
    print(f"Speedup (NILT vs MOL): {results['estimation_comparison']['speedup']:.1f}x")


def reproduce_ablation():
    """Reproduce ablation study."""
    print_header("Ablation Study: CFL Constraint Importance")

    from ablation_study import run_ablation_study
    run_ablation_study(verbose=True)


def run_all():
    """Run complete reproduction suite."""
    print_header("FFT-NILT Paper 2: Complete Reproducibility Suite")
    print("This script reproduces all figures and tables from the paper.")
    print("Estimated runtime: 5-10 minutes (CPU), 2-3 minutes (GPU)")

    # Figures
    reproduce_figures()

    # Tables
    reproduce_table5()
    reproduce_table6()
    reproduce_table8()

    # Case study
    reproduce_case_study()

    # Ablation
    reproduce_ablation()

    print_header("Reproduction Complete")
    print("All figures saved to: figures/")
    print("Review output above for table data.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reproduce Paper 2 results")
    parser.add_argument("--figures", action="store_true", help="Generate figures only")
    parser.add_argument("--tables", action="store_true", help="Generate tables only")
    parser.add_argument("--case-study", action="store_true", help="Run case study only")
    parser.add_argument("--ablation", action="store_true", help="Run ablation only")
    parser.add_argument("--all", action="store_true", help="Run everything (default)")

    args = parser.parse_args()

    if args.figures:
        reproduce_figures()
    elif args.tables:
        reproduce_table5()
        reproduce_table6()
        reproduce_table8()
    elif args.case_study:
        reproduce_case_study()
    elif args.ablation:
        reproduce_ablation()
    else:
        run_all()
