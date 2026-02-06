#!/usr/bin/env python3
"""
Three-way benchmark: NumPy vs JAX vs PyTorch for FFT-NILT.

Addresses the question: Which GPU framework is best for NILT?

Usage:
    python benchmark_numpy_jax_torch.py                    # JSON to stdout
    python benchmark_numpy_jax_torch.py --save results.json
    python benchmark_numpy_jax_torch.py --pretty
"""

from __future__ import annotations
import argparse
import json
import time
import sys
from typing import Dict, Any, List, Optional
import numpy as np

# NumPy implementation
from nilt_fft import fft_nilt as fft_nilt_numpy
from tuner import tune_params
from problems import dampener, get_all_problems


def check_jax_available() -> Dict[str, Any]:
    """Check JAX availability."""
    try:
        import jax
        return {
            "available": True,
            "backend": str(jax.default_backend()),
            "devices": [str(d) for d in jax.devices()],
            "version": jax.__version__
        }
    except ImportError:
        return {"available": False, "reason": "not installed"}
    except Exception as e:
        return {"available": False, "reason": str(e)}


def check_torch_available() -> Dict[str, Any]:
    """Check PyTorch availability."""
    try:
        import torch
        info = {
            "available": True,
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
        return info
    except ImportError:
        return {"available": False, "reason": "not installed"}
    except Exception as e:
        return {"available": False, "reason": str(e)}


# ============================================================================
# NumPy FFT-NILT (baseline)
# ============================================================================

def benchmark_numpy(F, a: float, T: float, N: int, n_runs: int = 100, n_warmup: int = 10) -> Dict[str, float]:
    """Benchmark NumPy FFT-NILT."""
    for _ in range(n_warmup):
        fft_nilt_numpy(F, a, T, N)

    timings = []
    for _ in range(n_runs):
        t_start = time.perf_counter()
        fft_nilt_numpy(F, a, T, N)
        t_end = time.perf_counter()
        timings.append((t_end - t_start) * 1e6)

    timings = np.array(timings)
    return {
        "median_us": float(np.median(timings)),
        "mad_us": float(np.median(np.abs(timings - np.median(timings)))),
        "min_us": float(np.min(timings)),
        "max_us": float(np.max(timings))
    }


# ============================================================================
# JAX FFT-NILT
# ============================================================================

_jax_fft_nilt = None

def get_jax_fft_nilt():
    """Create JAX FFT-NILT function with JIT compilation."""
    global _jax_fft_nilt
    if _jax_fft_nilt is not None:
        return _jax_fft_nilt

    import jax
    import jax.numpy as jnp
    from functools import partial

    @partial(jax.jit, static_argnums=(0, 3))
    def fft_nilt_jax(F, a: float, T: float, N: int):
        """
        FFT-NILT using JAX.

        Parameters
        ----------
        F : callable
            Transfer function F(s) - must be JAX-traceable
        a : float
            Bromwich shift
        T : float
            Half-period
        N : int
            FFT size
        """
        delta_omega = jnp.pi / T
        delta_t = 2 * T / N

        k = jnp.arange(N)
        t = k * delta_t
        omega = k * delta_omega

        s = a + 1j * omega
        G = jax.vmap(F)(s)
        G = G.at[0].set(G[0] / 2)

        z_ifft = N * jnp.fft.ifft(G)
        f_complex = jnp.exp(a * t) / T * z_ifft
        f = jnp.real(f_complex)

        return f, t, z_ifft

    _jax_fft_nilt = fft_nilt_jax
    return fft_nilt_jax


def benchmark_jax(F_jax, a: float, T: float, N: int, n_runs: int = 100, n_warmup: int = 10) -> Optional[Dict[str, float]]:
    """Benchmark JAX FFT-NILT."""
    try:
        import jax
        fft_nilt = get_jax_fft_nilt()
    except ImportError:
        return None

    # Warmup (JIT compilation)
    for _ in range(n_warmup):
        f, t, z = fft_nilt(F_jax, a, T, N)
        jax.block_until_ready(f)

    timings = []
    for _ in range(n_runs):
        t_start = time.perf_counter()
        f, t, z = fft_nilt(F_jax, a, T, N)
        jax.block_until_ready(f)
        t_end = time.perf_counter()
        timings.append((t_end - t_start) * 1e6)

    timings = np.array(timings)
    return {
        "median_us": float(np.median(timings)),
        "mad_us": float(np.median(np.abs(timings - np.median(timings)))),
        "min_us": float(np.min(timings)),
        "max_us": float(np.max(timings)),
        "backend": str(jax.default_backend())
    }


# ============================================================================
# PyTorch FFT-NILT
# ============================================================================

_torch_fft_nilt = None

def get_torch_fft_nilt():
    """Create PyTorch FFT-NILT function."""
    global _torch_fft_nilt
    if _torch_fft_nilt is not None:
        return _torch_fft_nilt

    import torch

    def fft_nilt_torch(F, a: float, T: float, N: int, device: str = "cuda"):
        """
        FFT-NILT using PyTorch.

        Parameters
        ----------
        F : callable
            Transfer function F(s) - must accept torch tensors
        a : float
            Bromwich shift
        T : float
            Half-period
        N : int
            FFT size
        device : str
            'cuda' or 'cpu'
        """
        # Frequency and time spacing
        delta_omega = torch.pi / T
        delta_t = 2 * T / N

        # Create grids on device
        k = torch.arange(N, device=device, dtype=torch.float64)
        t = k * delta_t
        omega = k * delta_omega

        # Laplace variable along Bromwich contour
        s = a + 1j * omega

        # Evaluate transfer function (vectorized)
        G = F(s)

        # Apply trapezoidal weights
        G[0] = G[0] / 2

        # Inverse FFT
        z_ifft = N * torch.fft.ifft(G)

        # Apply exponential scaling
        f_complex = torch.exp(a * t) / T * z_ifft

        # Extract real part
        f = f_complex.real

        return f, t, z_ifft

    _torch_fft_nilt = fft_nilt_torch
    return fft_nilt_torch


def benchmark_torch(F_torch, a: float, T: float, N: int, n_runs: int = 100, n_warmup: int = 10, device: str = "cuda") -> Optional[Dict[str, float]]:
    """Benchmark PyTorch FFT-NILT."""
    try:
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        fft_nilt = get_torch_fft_nilt()
    except ImportError:
        return None

    # Warmup
    for _ in range(n_warmup):
        f, t, z = fft_nilt(F_torch, a, T, N, device=device)
        if device == "cuda":
            torch.cuda.synchronize()

    timings = []
    for _ in range(n_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        f, t, z = fft_nilt(F_torch, a, T, N, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        timings.append((t_end - t_start) * 1e6)

    timings = np.array(timings)
    return {
        "median_us": float(np.median(timings)),
        "mad_us": float(np.median(np.abs(timings - np.median(timings)))),
        "min_us": float(np.min(timings)),
        "max_us": float(np.max(timings)),
        "device": device
    }


# ============================================================================
# Transfer function definitions for each framework
# ============================================================================

def get_dampener_functions():
    """Get dampener transfer function for each framework."""
    omega_n = 1.0
    zeta = 0.2

    # NumPy version
    def F_numpy(s):
        return omega_n**2 / (s**2 + 2*zeta*omega_n*s + omega_n**2)

    # JAX version (inline, no external dependency)
    F_jax = None
    try:
        import jax

        def F_jax(s):
            return omega_n**2 / (s**2 + 2*zeta*omega_n*s + omega_n**2)
    except ImportError:
        pass

    # PyTorch version
    F_torch = None
    try:
        import torch

        def F_torch(s):
            return omega_n**2 / (s**2 + 2*zeta*omega_n*s + omega_n**2)
    except ImportError:
        pass

    return {
        "numpy": F_numpy,
        "jax": F_jax,
        "torch": F_torch,
        "alpha_c": -zeta * omega_n,
        "C": omega_n / np.sqrt(1 - zeta**2),
        "rho": omega_n
    }


# ============================================================================
# Main benchmark
# ============================================================================

def run_comparison(
    N_values: List[int] = [128, 256, 512, 1024, 2048, 4096, 8192, 16384],
    n_runs: int = 100
) -> Dict[str, Any]:
    """Run three-way comparison."""

    funcs = get_dampener_functions()
    params = tune_params(
        t_end=20.0,
        alpha_c=funcs["alpha_c"],
        C=funcs["C"],
        rho=funcs["rho"]
    )

    jax_info = check_jax_available()
    torch_info = check_torch_available()

    results = []
    for N in N_values:
        row = {"N": N}

        # NumPy
        row["numpy"] = benchmark_numpy(funcs["numpy"], params.a, params.T, N, n_runs)

        # JAX
        if jax_info["available"] and funcs["jax"]:
            row["jax"] = benchmark_jax(funcs["jax"], params.a, params.T, N, n_runs)
        else:
            row["jax"] = None

        # PyTorch GPU
        if torch_info["available"] and funcs["torch"]:
            if torch_info.get("cuda_available"):
                row["torch_gpu"] = benchmark_torch(funcs["torch"], params.a, params.T, N, n_runs, device="cuda")
            else:
                row["torch_gpu"] = None
            row["torch_cpu"] = benchmark_torch(funcs["torch"], params.a, params.T, N, n_runs, device="cpu")
        else:
            row["torch_gpu"] = None
            row["torch_cpu"] = None

        # Compute speedups
        numpy_time = row["numpy"]["median_us"]
        row["speedup_jax"] = numpy_time / row["jax"]["median_us"] if row["jax"] else None
        row["speedup_torch_gpu"] = numpy_time / row["torch_gpu"]["median_us"] if row["torch_gpu"] else None
        row["speedup_torch_cpu"] = numpy_time / row["torch_cpu"]["median_us"] if row["torch_cpu"] else None

        results.append(row)

    return {
        "problem": "dampener",
        "params": {"a": params.a, "T": params.T},
        "n_runs": n_runs,
        "results": results
    }


def format_summary(results: Dict[str, Any], jax_info: Dict, torch_info: Dict) -> str:
    """Format human-readable summary."""
    lines = [
        "=" * 90,
        "  FFT-NILT Benchmark: NumPy vs JAX vs PyTorch",
        "=" * 90,
        "",
        f"JAX: {jax_info.get('backend', 'N/A')} ({jax_info.get('version', 'N/A')})",
        f"PyTorch: {torch_info.get('version', 'N/A')}, CUDA: {torch_info.get('cuda_device', 'N/A')}",
        "",
        f"{'N':>8} {'NumPy':>12} {'JAX':>12} {'Torch GPU':>12} {'Torch CPU':>12} | {'JAX':>8} {'T-GPU':>8} {'T-CPU':>8}",
        f"{'':>8} {'(μs)':>12} {'(μs)':>12} {'(μs)':>12} {'(μs)':>12} | {'speedup':>8} {'speedup':>8} {'speedup':>8}",
        "-" * 90
    ]

    for r in results["results"]:
        numpy_us = r["numpy"]["median_us"]
        jax_us = r["jax"]["median_us"] if r["jax"] else None
        torch_gpu_us = r["torch_gpu"]["median_us"] if r["torch_gpu"] else None
        torch_cpu_us = r["torch_cpu"]["median_us"] if r["torch_cpu"] else None

        jax_str = f"{jax_us:.1f}" if jax_us else "N/A"
        tgpu_str = f"{torch_gpu_us:.1f}" if torch_gpu_us else "N/A"
        tcpu_str = f"{torch_cpu_us:.1f}" if torch_cpu_us else "N/A"

        jax_sp = f"{r['speedup_jax']:.2f}×" if r['speedup_jax'] else "N/A"
        tgpu_sp = f"{r['speedup_torch_gpu']:.2f}×" if r['speedup_torch_gpu'] else "N/A"
        tcpu_sp = f"{r['speedup_torch_cpu']:.2f}×" if r['speedup_torch_cpu'] else "N/A"

        lines.append(
            f"{r['N']:>8} {numpy_us:>12.1f} {jax_str:>12} {tgpu_str:>12} {tcpu_str:>12} | {jax_sp:>8} {tgpu_sp:>8} {tcpu_sp:>8}"
        )

    lines.append("-" * 90)
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="NumPy vs JAX vs PyTorch FFT-NILT benchmark")
    parser.add_argument('--save', type=str, help='Save to JSON file')
    parser.add_argument('--pretty', action='store_true', help='Human-readable output')
    parser.add_argument('--n-runs', type=int, default=100, help='Benchmark runs')

    args = parser.parse_args()

    jax_info = check_jax_available()
    torch_info = check_torch_available()

    results = run_comparison(n_runs=args.n_runs)

    all_results = {
        "metadata": {
            "n_runs": args.n_runs,
            "jax": jax_info,
            "torch": torch_info
        },
        "comparison": results
    }

    if args.pretty:
        print(format_summary(results, jax_info, torch_info), file=sys.stderr)

    json_output = json.dumps(all_results, indent=2)

    if args.save:
        with open(args.save, 'w') as f:
            f.write(json_output)
        print(f"Saved to: {args.save}", file=sys.stderr)
    else:
        print(json_output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
