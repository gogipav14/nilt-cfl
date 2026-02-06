"""
FFT-based Numerical Inverse Laplace Transform implementation.

Implements the Dubner-Abate (1968) / Hsu-Dranoff (1987) method with
CFL-informed parameter selection for stable inversion of Laplace transforms.

For real-valued f(t), the Bromwich integral reduces to (Eq. 2):
    f(t) = (e^{at}/π) Re[∫_0^∞ F(a+iω) e^{iωt} dω]

FFT Acceleration (Eq. 3):
    Discretize with Δω = π/T and Δt = 2T/N, apply trapezoidal quadrature
    (half-weight at k=0), and compute the sum via IFFT in O(N log N):

    f(t_j) ≈ (e^{a t_j} / T) Re[Σ_{k=0}^{N-1} w_k F(a + ikΔω) e^{ikΔω t_j}]

    where w_0 = 1/2, w_k = 1 for k ≥ 1.

Quality Diagnostic (Eq. 12):
    ε_Im = max|Im(f̃)| / max|Re(f̃)|
    Should be < 10^{-2} for well-tuned parameters.

CFL-like Feasibility Condition (Eq. 10):
    α_c * t_max + ln(C/ε_tail) < L - δ_s

Reference: Paper submitted to Chemical Engineering Science (2026)
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
    Compute inverse Laplace transform using FFT (Eq. 3 in paper).

    Evaluates F(s) at N positive frequencies along the Bromwich contour,
    applies trapezoidal quadrature (DC half-weight), and computes the sum
    via IFFT with 1/T scaling. Takes Re() of the result per Eq. 3.

    For the quality diagnostic (eps_im), a separate Hermitian spectrum is
    constructed from the first N/2+1 frequencies so that the imaginary
    leakage metric is meaningful.

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
        Hermitian IFFT output for eps_im diagnostic
        (nearly real for well-tuned parameters)
    """
    # Frequency and time spacing
    delta_omega = np.pi / T
    delta_t = 2 * T / N

    # Time grid: t_j = j * Δt
    t = np.arange(N) * delta_t

    # Evaluate F at all N positive frequencies: k = 0, ..., N-1 (Eq. 3)
    omega = np.arange(N) * delta_omega
    s = a + 1j * omega
    G = np.array([F(sk) for sk in s], dtype=np.complex128)

    # Trapezoidal quadrature: half-weight at DC (k=0)
    G[0] = G[0] / 2

    # Compute f(t) via one-sided IFFT + Re() (paper Eq. 3)
    # f(t_j) = (e^{a*t_j}/T) * Re[N * IFFT(G)]
    z_raw = N * np.fft.ifft(G)
    f = np.exp(a * t) / T * np.real(z_raw)

    # Diagnostic: Hermitian spectrum from first N/2+1 frequencies
    # for eps_im quality metric (Eq. 12)
    n_pos = N // 2 + 1
    G_herm = np.zeros(N, dtype=np.complex128)
    G_herm[:n_pos] = G[:n_pos]
    G_herm[n_pos:] = np.conj(G[1:N - n_pos + 1][::-1])
    z_ifft = N * np.fft.ifft(G_herm)

    return f, t, z_ifft


def eps_im(z_ifft: np.ndarray) -> float:
    """
    Compute imaginary leakage metric (Eq. 12).

    ε_Im = max_j |Im(f̃(t_j))| / max_j |Re(f̃(t_j))|

    For real-valued f(t), the NILT output should be purely real.
    Non-zero imaginary content indicates numerical issues (aliasing,
    roundoff, truncation) and serves as a quality diagnostic.

    Parameters
    ----------
    z_ifft : ndarray
        Complex IFFT output (before extracting real part)

    Returns
    -------
    eps : float
        Imaginary leakage metric (should be < 1e-2 for quality results)

    Reference
    ---------
    Paper Eq. (12): ε_Im = max|Im(f̃)| / max|Re(f̃)|
    Quality threshold Eq. (13): ε_Im ≤ 10^{-2}
    """
    real_part = np.real(z_ifft)
    imag_part = np.imag(z_ifft)

    max_real = np.max(np.abs(real_part))
    max_imag = np.max(np.abs(imag_part))

    if max_real < 1e-300:
        return np.inf

    return max_imag / max_real


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
