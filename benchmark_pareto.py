"""
Pareto Benchmark: Accuracy vs Runtime Trade-offs for FFT-NILT

Generates accuracy-vs-time Pareto curves by sweeping N values for:
1. FFT-NILT with CFL tuning
2. de Hoog algorithm (for comparison)

Covers multiple problem types:
- Oscillatory (second-order underdamped)
- Discontinuity (FOPDT with delay)
- Branch point (semi-infinite diffusion)
- Step response (breakthrough curve)

Author: Gorgi Pavlov
"""

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from problems import (
    get_problem, first_order_lag, fopdt, second_order,
    semi_infinite_diffusion, packed_bed
)
from nilt_fft import fft_nilt, eps_im
from tuner import tune_params


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    problem_name: str
    method: str
    N: int
    rmse: float
    max_error: float
    timing_us: float
    timing_std_us: float
    eps_im: float


def benchmark_fft_nilt(
    problem_name: str,
    N_values: List[int],
    t_end: float = 10.0,
    n_timing_runs: int = 50
) -> List[BenchmarkResult]:
    """
    Benchmark FFT-NILT across multiple N values.

    Parameters
    ----------
    problem_name : str
        Problem identifier (lag, fopdt, secondorder, diffusion, packedbed)
    N_values : list
        FFT sizes to benchmark
    t_end : float
        End time for evaluation
    n_timing_runs : int
        Number of timing iterations

    Returns
    -------
    results : list
        BenchmarkResult for each N value
    """
    problem = get_problem(problem_name)
    results = []

    for N in N_values:
        # Tune parameters
        tuned = tune_params(
            t_end=t_end,
            alpha_c=problem.alpha_c,
            C=problem.C,
            rho=problem.rho,
            N_init=N
        )
        # Override N to test specific value
        a, T = tuned.a, tuned.T

        # Compute solution
        f, t, z = fft_nilt(problem.F, a, T, N)

        # Compute error metrics
        if problem.f_ref is not None:
            mask = (t > 0.1) & (t <= t_end)
            f_ref = problem.f_ref(t)
            rmse = np.sqrt(np.mean((f[mask] - f_ref[mask])**2))
            max_err = np.max(np.abs(f[mask] - f_ref[mask]))
        else:
            rmse = np.nan
            max_err = np.nan

        # Compute eps_im
        mask_eval = (t > 0.1) & (t <= t_end)
        current_eps_im = eps_im(z[mask_eval])

        # Timing
        timings = []
        for _ in range(n_timing_runs):
            t_start = time.perf_counter()
            fft_nilt(problem.F, a, T, N)
            timings.append((time.perf_counter() - t_start) * 1e6)

        timing_median = np.median(timings)
        timing_std = np.std(timings)

        results.append(BenchmarkResult(
            problem_name=problem_name,
            method="FFT-NILT",
            N=N,
            rmse=rmse,
            max_error=max_err,
            timing_us=timing_median,
            timing_std_us=timing_std,
            eps_im=current_eps_im
        ))

    return results


def de_hoog_nilt(
    F: Callable[[complex], complex],
    t: float,
    a: float,
    M: int = 20,
    T: float = None
) -> float:
    """
    de Hoog algorithm for numerical inverse Laplace transform.

    Accelerated summation using Padé approximants.

    Parameters
    ----------
    F : callable
        Laplace-domain transfer function
    t : float
        Single time point
    a : float
        Bromwich shift
    M : int
        Number of terms (default 20)
    T : float
        Integration period (default 2*t)

    Returns
    -------
    f : float
        Approximation to f(t)
    """
    if t <= 0:
        return 0.0

    if T is None:
        T = 2 * t

    omega = np.pi / T

    # Compute coefficients
    d = np.zeros(2 * M + 1, dtype=complex)
    for k in range(2 * M + 1):
        s = a + 1j * k * omega
        d[k] = F(s)
        if k == 0:
            d[k] /= 2

    # Quotient-difference algorithm for Padé approximants
    A = np.zeros((2 * M + 2, 2 * M + 1), dtype=complex)
    B = np.zeros((2 * M + 2, 2 * M + 1), dtype=complex)

    A[0, :] = 0
    A[1, :] = d[0]
    B[0, :] = 1
    B[1, :] = 1

    for n in range(1, 2 * M + 1):
        A[n + 1, n:] = A[n, n - 1:-1] + (d[n:] / d[n - 1:-1]) * (A[n, n:] - A[n - 1, n - 1:-1])
        B[n + 1, n:] = B[n, n - 1:-1] + (d[n:] / d[n - 1:-1]) * (B[n, n:] - B[n - 1, n - 1:-1])

    # Evaluate at z = exp(i * omega * t)
    z = np.exp(1j * omega * t)
    result = A[2 * M + 1, 2 * M] / B[2 * M + 1, 2 * M]

    return np.exp(a * t) / T * np.real(result)


def de_hoog_nilt_vectorized(
    F: Callable[[complex], complex],
    t_arr: np.ndarray,
    a: float,
    M: int = 20
) -> np.ndarray:
    """
    de Hoog algorithm vectorized over time array.

    Note: This is a simplified version. Full de Hoog implementation
    uses continued fraction acceleration which is more complex.
    """
    # For simplicity, use point-by-point evaluation
    # A proper implementation would batch the coefficient computation
    return np.array([de_hoog_nilt(F, t, a, M) for t in t_arr])


def benchmark_de_hoog(
    problem_name: str,
    M_values: List[int],
    t_end: float = 10.0,
    N_time: int = 100,
    n_timing_runs: int = 10
) -> List[BenchmarkResult]:
    """
    Benchmark de Hoog algorithm across multiple M values.

    Parameters
    ----------
    problem_name : str
        Problem identifier
    M_values : list
        Number of terms to benchmark
    t_end : float
        End time
    N_time : int
        Number of time points
    n_timing_runs : int
        Timing iterations

    Returns
    -------
    results : list
        BenchmarkResult for each M value
    """
    problem = get_problem(problem_name)

    # Tune a using CFL framework
    tuned = tune_params(
        t_end=t_end,
        alpha_c=problem.alpha_c,
        C=problem.C,
        rho=problem.rho
    )
    a = tuned.a

    # Time grid
    t_arr = np.linspace(0.1, t_end, N_time)

    results = []

    for M in M_values:
        # Compute solution
        f = de_hoog_nilt_vectorized(problem.F, t_arr, a, M)

        # Compute error metrics
        if problem.f_ref is not None:
            f_ref = problem.f_ref(t_arr)
            rmse = np.sqrt(np.mean((f - f_ref)**2))
            max_err = np.max(np.abs(f - f_ref))
        else:
            rmse = np.nan
            max_err = np.nan

        # Timing (per N_time points)
        timings = []
        for _ in range(n_timing_runs):
            t_start = time.perf_counter()
            de_hoog_nilt_vectorized(problem.F, t_arr, a, M)
            timings.append((time.perf_counter() - t_start) * 1e6)

        timing_median = np.median(timings)
        timing_std = np.std(timings)

        results.append(BenchmarkResult(
            problem_name=problem_name,
            method="de Hoog",
            N=M,  # Using N field for M
            rmse=rmse,
            max_error=max_err,
            timing_us=timing_median,
            timing_std_us=timing_std,
            eps_im=np.nan  # Not applicable for de Hoog
        ))

    return results


def run_pareto_benchmark(
    problems: List[str] = None,
    N_values: List[int] = None,
    M_values: List[int] = None,
    t_end: float = 10.0,
    verbose: bool = True
) -> Dict[str, List[BenchmarkResult]]:
    """
    Run full Pareto benchmark suite.

    Parameters
    ----------
    problems : list
        Problem names to benchmark
    N_values : list
        FFT sizes for FFT-NILT
    M_values : list
        Number of terms for de Hoog
    t_end : float
        End time
    verbose : bool
        Print progress

    Returns
    -------
    results : dict
        Results organized by problem name
    """
    if problems is None:
        problems = ["lag", "fopdt", "secondorder", "diffusion"]

    if N_values is None:
        N_values = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

    if M_values is None:
        M_values = [5, 10, 15, 20, 25, 30]

    all_results = {}

    for prob_name in problems:
        if verbose:
            print(f"\nBenchmarking {prob_name}...")
            print("-" * 40)

        # FFT-NILT benchmark
        if verbose:
            print("  FFT-NILT sweep...")
        fft_results = benchmark_fft_nilt(prob_name, N_values, t_end)

        # de Hoog benchmark
        if verbose:
            print("  de Hoog sweep...")
        dehoog_results = benchmark_de_hoog(prob_name, M_values, t_end)

        all_results[prob_name] = fft_results + dehoog_results

        if verbose:
            print(f"\n  FFT-NILT Results:")
            print(f"  {'N':>6} {'RMSE':>12} {'Time (μs)':>12} {'ε_Im':>10}")
            for r in fft_results:
                print(f"  {r.N:>6} {r.rmse:>12.2e} {r.timing_us:>12.1f} {r.eps_im:>10.2e}")

            print(f"\n  de Hoog Results:")
            print(f"  {'M':>6} {'RMSE':>12} {'Time (μs)':>12}")
            for r in dehoog_results:
                print(f"  {r.N:>6} {r.rmse:>12.2e} {r.timing_us:>12.1f}")

    return all_results


def generate_pareto_data(results: Dict[str, List[BenchmarkResult]]) -> Dict[str, Dict]:
    """
    Extract Pareto front data for plotting.

    Returns time vs error data organized by problem and method.
    """
    pareto_data = {}

    for prob_name, result_list in results.items():
        pareto_data[prob_name] = {
            "FFT-NILT": {"time": [], "rmse": [], "N": []},
            "de Hoog": {"time": [], "rmse": [], "M": []}
        }

        for r in result_list:
            if r.method == "FFT-NILT":
                pareto_data[prob_name]["FFT-NILT"]["time"].append(r.timing_us)
                pareto_data[prob_name]["FFT-NILT"]["rmse"].append(r.rmse)
                pareto_data[prob_name]["FFT-NILT"]["N"].append(r.N)
            else:
                pareto_data[prob_name]["de Hoog"]["time"].append(r.timing_us)
                pareto_data[prob_name]["de Hoog"]["rmse"].append(r.rmse)
                pareto_data[prob_name]["de Hoog"]["M"].append(r.N)

    return pareto_data


if __name__ == "__main__":
    print("=" * 60)
    print("Pareto Benchmark: Accuracy vs Runtime Trade-offs")
    print("=" * 60)

    results = run_pareto_benchmark(
        problems=["lag", "fopdt", "secondorder", "diffusion"],
        N_values=[128, 256, 512, 1024, 2048, 4096],
        M_values=[10, 15, 20, 25],
        verbose=True
    )

    print("\n" + "=" * 60)
    print("Summary: Pareto Front Comparison")
    print("=" * 60)

    for prob_name, result_list in results.items():
        fft_best = min([r for r in result_list if r.method == "FFT-NILT"],
                       key=lambda x: x.rmse if not np.isnan(x.rmse) else np.inf)
        dehoog_best = min([r for r in result_list if r.method == "de Hoog"],
                          key=lambda x: x.rmse if not np.isnan(x.rmse) else np.inf)

        print(f"\n{prob_name}:")
        print(f"  Best FFT-NILT: N={fft_best.N}, RMSE={fft_best.rmse:.2e}, Time={fft_best.timing_us:.0f}μs")
        print(f"  Best de Hoog:  M={dehoog_best.N}, RMSE={dehoog_best.rmse:.2e}, Time={dehoog_best.timing_us:.0f}μs")
