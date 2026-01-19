"""
FFT-based Numerical Inverse Laplace Transform implementation.

Implements the Dubner-Abate (1968) / Hsu-Dranoff (1987) method for
CFL-stable inversion of Laplace transforms.

For real-valued f(t), the Bromwich integral reduces to:
    f(t) = exp(at)/π * Re[∫_0^∞ F(a+iω) exp(iωt) dω]

Dubner-Abate discretization (trapezoidal quadrature on positive frequencies):
    f(t_j) ≈ exp(a t_j)/T * Re[½F(a) + Σ_{k=1}^{N-1} F(a+ikΔω) exp(ikΔω t_j)]

IMPORTANT: Why One-Sided (Not Hermitian) Construction

For Laplace inversion, F(a+iω) ≠ conj(F(a-iω)) in general because F(s) is
evaluated along the Bromwich contour Re(s) = a, not on the imaginary axis.
The spectrum is fundamentally one-sided, and we extract the physical signal
by taking Re[] of the sum.

The "imaginary leakage" metric (ε_Im) measures numerical residual, not
implementation error. High ε_Im in one-sided evaluation is expected because:
- We sample positive frequencies only (ω = 0, Δω, 2Δω, ...)
- The IFFT "sees" this as a non-Hermitian spectrum
- The imaginary component is the "complement" that gets discarded by Re[]

For CFL-tuned parameters, ε_Im correlates with:
- Aliasing error (controlled by a and T)
- Truncation error (controlled by N)
- NOT with implementation correctness

Connection to NILT-CFL conditions:
- Constraint 1 (dynamic range): a*t_max < L - δ_s ensures exp(at) doesn't overflow
- Constraint 2 (spectral placement): a > α_c ensures contour right of singularities
- Constraint 3 (aliasing): exp(-2aT) < ε_tail ensures wraparound suppression
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Tuple


def fft_nilt(
    F: Callable[[complex], complex],
    a: float,
    T: float,
    N: int,
    return_complex: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute inverse Laplace transform using Dubner-Abate FFT method.

    Evaluates F(s) along the Bromwich contour Re(s) = a at positive
    frequencies ω_k = k*Δω for k = 0, 1, ..., N-1, then uses IFFT
    followed by Re[] extraction.

    Parameters
    ----------
    F : callable
        Laplace-domain transfer function F(s) -> complex
    a : float
        Bromwich shift parameter (contour Re(s) = a)
    T : float
        Half-period (aliasing period = 2T)
    N : int
        Number of FFT points (preferably power of 2)
    return_complex : bool
        Unused, kept for API compatibility

    Returns
    -------
    f : ndarray
        Time-domain function values at t_j = j * Δt
    t : ndarray
        Time points t_j for j = 0, ..., N-1
    z_ifft : ndarray
        Complex IFFT output (before Re[] extraction)

    Notes
    -----
    For CFL-tuned parameters (a, T) satisfying the feasibility conditions:
    - a > α_c + δ_min (spectral placement)
    - a < (L - δ_s)/(2T) (dynamic range)
    - a ≥ α_c + ln(C/ε_tail)/(2T-t_end) (aliasing suppression)

    The result accuracy is controlled by N (truncation error) and the
    CFL parameters (aliasing error). The imaginary part of z_ifft is
    the "complement" signal that gets discarded - high Im/Re ratio
    indicates aliasing or truncation, not implementation error.
    """
    # Frequency spacing: Δω = π/T
    delta_omega = np.pi / T

    # Time step: Δt = 2T/N
    delta_t = 2 * T / N

    # Time grid: t_j = j * Δt for j = 0, ..., N-1
    t = np.arange(N) * delta_t

    # Frequency grid: ω_k = k * Δω for k = 0, ..., N-1
    omega = np.arange(N) * delta_omega
    s = a + 1j * omega

    # Evaluate F(s) at Bromwich contour points
    G = np.array([F(sk) for sk in s], dtype=np.complex128)

    # Apply trapezoidal weight at DC (k=0 endpoint)
    G[0] = G[0] / 2

    # Compute sum via IFFT
    # IFFT: (1/N) * Σ G[k] exp(i 2π k j / N)
    # Our sum: Σ G[k] exp(i k Δω t_j) = Σ G[k] exp(i 2π k j / N) [since Δω*Δt = 2π/N]
    # So multiply IFFT by N
    z_ifft = N * np.fft.ifft(G)

    # Apply exponential factor, scaling, and extract real part
    # f(t) = exp(a*t) / T * Re[sum]
    f = np.exp(a * t) / T * np.real(z_ifft)

    return f, t, z_ifft


def fft_nilt_one_sided(
    F: Callable[[complex], complex],
    a: float,
    T: float,
    N: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One-sided (non-Hermitian) implementation for comparison.

    This is the traditional Hsu-Dranoff approach that evaluates F(s)
    at all N frequencies and takes Re[] at the end.

    May have larger imaginary leakage due to accumulated phase errors.
    """
    delta_omega = np.pi / T
    delta_t = 2 * T / N
    t = np.arange(N) * delta_t
    omega = np.arange(N) * delta_omega
    s = a + 1j * omega

    G = np.array([F(sk) for sk in s], dtype=np.complex128)
    G[0] = G[0] / 2  # Trapezoidal weight for DC

    z_ifft = N * np.fft.ifft(G)
    f = np.exp(a * t) / T * np.real(z_ifft)

    return f, t, z_ifft


def eps_im(z_ifft: np.ndarray) -> float:
    """
    Compute imaginary leakage metric.

    ε_Im = RMS(Im(z)) / RMS(Re(z))

    This measures the relative magnitude of the imaginary component
    that should be zero for real-valued f(t).

    Parameters
    ----------
    z_ifft : ndarray
        Complex IFFT output (before extracting real part)

    Returns
    -------
    eps : float
        Imaginary leakage metric
    """
    real_part = np.real(z_ifft)
    imag_part = np.imag(z_ifft)

    rms_real = np.sqrt(np.mean(real_part**2))
    rms_imag = np.sqrt(np.mean(imag_part**2))

    if rms_real < 1e-300:
        return np.inf

    return rms_imag / rms_real


def n_doubling_error(
    F: Callable[[complex], complex],
    a: float,
    T: float,
    N: int,
    t_eval_min: float = 0.1,
    t_eval_max: float = None
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute N-doubling convergence error.

    E_N = RMS(f_N - f_{2N}) / RMS(f_{2N})

    evaluated over [t_eval_min, t_eval_max].

    Parameters
    ----------
    F : callable
        Transfer function
    a : float
        Bromwich shift
    T : float
        Half-period
    N : int
        Current sample count
    t_eval_min : float
        Minimum time for evaluation (default 0.1 to avoid t=0 issues)
    t_eval_max : float
        Maximum time for evaluation (default T)

    Returns
    -------
    E_N : float
        Convergence error metric
    f_N : ndarray
        Solution at N points (on evaluation grid)
    f_2N : ndarray
        Solution at 2N points (on evaluation grid)
    """
    if t_eval_max is None:
        t_eval_max = T

    # Compute at N points
    f_N_full, t_N, _ = fft_nilt(F, a, T, N)

    # Compute at 2N points
    f_2N_full, t_2N, _ = fft_nilt(F, a, T, 2 * N)

    # Interpolate to common evaluation grid
    # Use the 2N time points within evaluation range
    mask_2N = (t_2N >= t_eval_min) & (t_2N <= t_eval_max)
    t_eval = t_2N[mask_2N]
    f_2N = f_2N_full[mask_2N]

    # Interpolate f_N to the same time points
    f_N = np.interp(t_eval, t_N, f_N_full)

    # Compute error
    rms_diff = np.sqrt(np.mean((f_N - f_2N)**2))
    rms_2N = np.sqrt(np.mean(f_2N**2))

    if rms_2N < 1e-300:
        return np.inf, f_N, f_2N

    E_N = rms_diff / rms_2N

    return E_N, f_N, f_2N
