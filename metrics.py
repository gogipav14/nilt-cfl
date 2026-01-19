"""
Error metrics and diagnostics for FFT-NILT.
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Tuple, Optional


def rmse(f_computed: np.ndarray, f_reference: np.ndarray) -> float:
    """
    Compute root-mean-square error.

    RMSE = sqrt(mean((f_computed - f_reference)^2))

    Parameters
    ----------
    f_computed : ndarray
        Computed function values
    f_reference : ndarray
        Reference (analytical or high-precision) values

    Returns
    -------
    error : float
        RMSE value
    """
    return np.sqrt(np.mean((f_computed - f_reference)**2))


def relative_rmse(f_computed: np.ndarray, f_reference: np.ndarray) -> float:
    """
    Compute relative RMSE.

    Relative RMSE = RMSE / RMS(f_reference)

    Parameters
    ----------
    f_computed : ndarray
        Computed function values
    f_reference : ndarray
        Reference values

    Returns
    -------
    error : float
        Relative RMSE
    """
    rms_ref = np.sqrt(np.mean(f_reference**2))
    if rms_ref < 1e-300:
        return np.inf
    return rmse(f_computed, f_reference) / rms_ref


def max_absolute_error(f_computed: np.ndarray, f_reference: np.ndarray) -> float:
    """
    Compute maximum absolute error.

    Parameters
    ----------
    f_computed : ndarray
        Computed function values
    f_reference : ndarray
        Reference values

    Returns
    -------
    error : float
        Maximum absolute error
    """
    return np.max(np.abs(f_computed - f_reference))


def max_relative_error(f_computed: np.ndarray, f_reference: np.ndarray) -> float:
    """
    Compute maximum relative error.

    Parameters
    ----------
    f_computed : ndarray
        Computed function values
    f_reference : ndarray
        Reference values

    Returns
    -------
    error : float
        Maximum relative error
    """
    abs_ref = np.abs(f_reference)
    mask = abs_ref > 1e-10  # Avoid division by small numbers
    if not np.any(mask):
        return np.inf
    return np.max(np.abs(f_computed[mask] - f_reference[mask]) / abs_ref[mask])


def contour_residual(
    F: Callable[[complex], complex],
    t: np.ndarray,
    f: np.ndarray,
    a: float,
    T: float,
    n_pts: int = 16
) -> float:
    """
    Compute contour residual by forward Laplace transform.

    For L sample points s_l on the contour, compute the numerical
    Laplace transform of f(t) and compare to F(s_l).

    R = max_l |F_numerical(s_l) - F(s_l)| / |F(s_l)|

    Parameters
    ----------
    F : callable
        Original transfer function
    t : ndarray
        Time points
    f : ndarray
        Time-domain values
    a : float
        Bromwich shift
    T : float
        Half-period
    n_pts : int
        Number of check points on contour

    Returns
    -------
    residual : float
        Maximum relative residual
    """
    # Sample points on contour
    omega_pts = np.linspace(0, np.pi / T * len(t) / 2, n_pts)
    s_pts = a + 1j * omega_pts

    residuals = []
    dt = t[1] - t[0] if len(t) > 1 else 1.0

    for s in s_pts:
        # Numerical Laplace transform via trapezoidal rule
        integrand = f * np.exp(-s * t)
        F_numerical = np.trapz(integrand, t)

        # True value
        F_true = F(s)

        if abs(F_true) > 1e-20:
            rel_err = abs(F_numerical - F_true) / abs(F_true)
            residuals.append(rel_err)

    if not residuals:
        return np.inf

    return max(residuals)


def alias_error_estimate(
    f_short: np.ndarray,
    t_short: np.ndarray,
    f_long: np.ndarray,
    t_long: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate aliasing error by comparing solutions with different T.

    Uses a longer-period solution as reference to estimate
    the alias contribution in the shorter-period solution.

    Parameters
    ----------
    f_short : ndarray
        Solution with shorter period (more aliasing)
    t_short : ndarray
        Time points for short-period solution
    f_long : ndarray
        Solution with longer period (reference)
    t_long : ndarray
        Time points for long-period solution

    Returns
    -------
    t_common : ndarray
        Common time points
    alias_estimate : ndarray
        Estimated alias error at each time point
    """
    # Interpolate to common grid (use short time points within long range)
    t_max = min(t_short.max(), t_long.max())
    mask_short = t_short <= t_max
    t_common = t_short[mask_short]

    f_short_eval = f_short[mask_short]
    f_long_eval = np.interp(t_common, t_long, f_long)

    alias_estimate = np.abs(f_short_eval - f_long_eval)

    return t_common, alias_estimate


def bin_eps_im_results(
    eps_im_values: np.ndarray,
    rmse_values: np.ndarray,
    bins: list = None
) -> dict:
    """
    Bin results by eps_im value for calibration table.

    Parameters
    ----------
    eps_im_values : ndarray
        Imaginary leakage values for each configuration
    rmse_values : ndarray
        RMSE values for each configuration
    bins : list, optional
        Bin edges (default: [0, 1e-4, 1e-3, 1e-2, inf])

    Returns
    -------
    results : dict
        Binned statistics
    """
    if bins is None:
        bins = [0, 1e-4, 1e-3, 1e-2, np.inf]

    results = {
        "bins": [],
        "n_cases": [],
        "mean_rmse": [],
        "max_rmse": [],
        "min_rmse": []
    }

    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (eps_im_values >= lo) & (eps_im_values < hi)

        if np.any(mask):
            rmse_in_bin = rmse_values[mask]
            results["bins"].append(f"{lo:.0e} - {hi:.0e}" if hi < np.inf else f"> {lo:.0e}")
            results["n_cases"].append(int(np.sum(mask)))
            results["mean_rmse"].append(float(np.mean(rmse_in_bin)))
            results["max_rmse"].append(float(np.max(rmse_in_bin)))
            results["min_rmse"].append(float(np.min(rmse_in_bin)))

    return results
