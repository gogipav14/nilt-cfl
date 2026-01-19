"""
CFL-informed parameter tuning for FFT-NILT.

Implements Algorithm 1 from the paper with adaptive refinement.
"""

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any

from nilt_fft import fft_nilt, eps_im, n_doubling_error


@dataclass
class TunedParams:
    """Container for tuned NILT parameters."""
    a: float           # Bromwich shift
    T: float           # Half-period
    N: int             # FFT size
    delta_t: float     # Time step
    t_max: float       # Maximum time = 2T
    a_min_star: float  # Tentative lower bound before floor
    a_min: float       # Actual lower bound after floor
    a_max: float       # Upper bound from dynamic range
    margin: float      # a_max - a (feasibility margin)
    feasible: bool     # Whether CFL conditions are satisfied


def tune_params(
    t_end: float,
    alpha_c: float,
    C: float = 1.0,
    kappa: float = 1.0,
    eps_tail: float = 1e-6,
    delta_min: float = 1e-3,
    delta_floor: float = 1e-3,
    delta_s: float = 10.0,
    N_init: int = 512,
    rho: Optional[float] = None,
    gamma: float = 1.5,
    L: float = 709.8  # ln(DBL_MAX)
) -> TunedParams:
    """
    Tune NILT parameters using CFL-informed framework.

    Implements Algorithm 1 Phase 1 (parameter initialization).

    Parameters
    ----------
    t_end : float
        End time for evaluation
    alpha_c : float
        Abscissa of convergence
    C : float
        Tail envelope constant (default 1.0)
    kappa : float
        Period factor T = kappa * t_end (default 1.0)
    eps_tail : float
        Aliasing tolerance (default 1e-6)
    delta_min : float
        Singularity margin (default 1e-3)
    delta_floor : float
        Minimum positive shift floor (default 1e-3)
    delta_s : float
        Dynamic range safety margin (default 10.0)
    N_init : int
        Initial FFT size (default 512)
    rho : float, optional
        Spectral radius for frequency heuristic
    gamma : float
        Oversampling factor for spectral heuristic (default 1.5)
    L : float
        ln(DBL_MAX) for precision (default 709.8 for float64)

    Returns
    -------
    params : TunedParams
        Tuned parameter set
    """
    # Step 1: Set half-period
    T = kappa * t_end

    # Step 2: Compute t_max
    t_max = 2 * T

    # Step 3: Dynamic range limit (Proposition 1)
    a_max = (L - delta_s) / t_max

    # Step 4-5: Aliasing and singularity constraints
    # Using interval worst-case bound from Lemma 2 (t = t_end)
    alias_factor = (2 * kappa - 1) * t_end
    if alias_factor > 0:
        a_alias = alpha_c + np.log(C / eps_tail) / alias_factor
    else:
        a_alias = alpha_c + delta_min

    a_sing = alpha_c + delta_min

    # Step 6: Tentative lower bound
    a_min_star = max(a_alias, a_sing)

    # Step 7: Enforce positivity floor
    a_min = max(a_min_star, delta_floor)

    # Step 8: Check feasibility (Theorem 1)
    feasible = (a_min <= a_max)

    # Step 9: Select a (minimal shift to reduce roundoff)
    a = a_min if feasible else a_max

    # Step 10: Initialize N
    if rho is not None:
        omega_max_init = gamma * rho
        N_from_spectrum = 2 ** int(np.ceil(np.log2(2 * T * omega_max_init / np.pi)))
        N = max(N_init, N_from_spectrum)
    else:
        N = N_init

    # Ensure N is power of 2
    N = 2 ** int(np.ceil(np.log2(N)))

    # Step 11: Time step
    delta_t = 2 * T / N

    # Margin
    margin = a_max - a

    return TunedParams(
        a=a,
        T=T,
        N=N,
        delta_t=delta_t,
        t_max=t_max,
        a_min_star=a_min_star,
        a_min=a_min,
        a_max=a_max,
        margin=margin,
        feasible=feasible
    )


def refine_until_accept(
    F: Callable[[complex], complex],
    params: TunedParams,
    t_end: float,
    eps_im_max: float = 1e-2,
    eps_conv: float = 1e-2,
    N_max: int = 16384,
    t_eval_min: float = 0.1,
    n_timing_runs: int = 100
) -> Dict[str, Any]:
    """
    Refine N until acceptance criteria are met.

    Implements Algorithm 1 Phase 2 (adaptive refinement).

    Parameters
    ----------
    F : callable
        Transfer function F(s)
    params : TunedParams
        Initial tuned parameters
    t_end : float
        End time for evaluation
    eps_im_max : float
        Imaginary leakage threshold (default 1e-2)
    eps_conv : float
        N-doubling convergence threshold (default 1e-2)
    N_max : int
        Maximum N before giving up (default 16384)
    t_eval_min : float
        Minimum time for evaluation (default 0.1)
    n_timing_runs : int
        Number of runs for timing (default 100)

    Returns
    -------
    result : dict
        Dictionary with solution and diagnostics
    """
    a = params.a
    T = params.T
    N = params.N

    accepted = False
    iterations = 0
    max_iterations = int(np.log2(N_max / N)) + 2

    while not accepted and iterations < max_iterations:
        # Step 12: Compute solution at current N
        f_full, t_full, z_ifft = fft_nilt(F, a, T, N)

        # Step 13: Compute eps_im
        mask = (t_full >= t_eval_min) & (t_full <= t_end)
        current_eps_im = eps_im(z_ifft[mask])

        # Step 14-15: Compute N-doubling error
        E_N, _, _ = n_doubling_error(F, a, T, N, t_eval_min, t_end)

        # Step 16: Check acceptance criteria
        if current_eps_im <= eps_im_max and E_N <= eps_conv:
            accepted = True
        else:
            # Step 17: Double N
            N = 2 * N
            if N > N_max:
                break

        iterations += 1

    # Recompute final solution
    f_full, t_full, z_ifft = fft_nilt(F, a, T, N)

    # Timing
    timings = []
    for _ in range(n_timing_runs):
        t_start = time.perf_counter()
        fft_nilt(F, a, T, N)
        t_end_timing = time.perf_counter()
        timings.append((t_end_timing - t_start) * 1e6)  # microseconds

    timing_median = np.median(timings)
    timing_mad = np.median(np.abs(timings - timing_median))

    # Extract evaluation interval
    mask = (t_full >= t_eval_min) & (t_full <= t_end)
    t_eval = t_full[mask]
    f_eval = f_full[mask]
    final_eps_im = eps_im(z_ifft[mask])

    # Final E_N
    final_E_N, _, _ = n_doubling_error(F, a, T, N, t_eval_min, t_end)

    return {
        "f_full": f_full,
        "t_full": t_full,
        "f_eval": f_eval,
        "t_eval": t_eval,
        "z_ifft": z_ifft,
        "a": a,
        "T": T,
        "N": N,
        "eps_im": final_eps_im,
        "E_N": final_E_N,
        "accepted": accepted,
        "iterations": iterations,
        "timing_median_us": timing_median,
        "timing_mad_us": timing_mad
    }


def check_cfl_feasibility(
    alpha_c: float,
    t_max: float,
    C: float = 1.0,
    eps_tail: float = 1e-6,
    delta_floor: float = 1e-3,
    delta_s: float = 10.0,
    L: float = 709.8
) -> Tuple[bool, bool, float, float]:
    """
    Check CFL feasibility conditions (Theorem 1).

    Parameters
    ----------
    alpha_c : float
        Abscissa of convergence
    t_max : float
        Maximum time (= 2T)
    C : float
        Tail envelope constant
    eps_tail : float
        Aliasing tolerance
    delta_floor : float
        Minimum positive shift
    delta_s : float
        Safety margin
    L : float
        ln(DBL_MAX)

    Returns
    -------
    cfl_ok : bool
        Whether Eq. (14) is satisfied
    floor_ok : bool
        Whether Eq. (14') is satisfied
    lhs_cfl : float
        Left-hand side of Eq. (14)
    lhs_floor : float
        Left-hand side of Eq. (14')
    """
    # Eq. (14): alpha_c * t_max + ln(C/eps_tail) <= L - delta_s
    lhs_cfl = alpha_c * t_max + np.log(C / eps_tail)
    rhs = L - delta_s
    cfl_ok = lhs_cfl <= rhs

    # Eq. (14'): delta_floor * t_max <= L - delta_s
    lhs_floor = delta_floor * t_max
    floor_ok = lhs_floor <= rhs

    return cfl_ok, floor_ok, lhs_cfl, lhs_floor
