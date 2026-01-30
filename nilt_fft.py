"""
FFT-based Numerical Inverse Laplace Transform implementation.

Implements the Dubner-Abate (1968) / Hsu-Dranoff (1987) method with
CFL-informed parameter selection for stable inversion of Laplace transforms.

For real-valued f(t), the Bromwich integral is (Eq. 2):
    f(t) = (e^{at}/2π) ∫_{-∞}^{∞} F(a+iω) e^{iωt} dω

Using the conjugate symmetry F(a-iω) = conj(F(a+iω)) for real f(t), this
is equivalent to the one-sided form (Eq. 3):
    f(t) = (e^{at}/π) Re[∫_0^∞ F(a+iω) e^{iωt} dω]

FFT Acceleration (Half-Spectrum Hermitian Construction):
    The integral is discretized with Δω = π/T and Δt = 2T/N.
    To avoid numerical overflow at negative frequencies:
    1. Evaluate F only at positive frequencies: k = 0, 1, ..., N/2
    2. Construct negative frequency bins via conjugate symmetry:
       G[N-k] = conj(G[k]) for k = 1, ..., N/2-1
    3. Apply IFFT to get nearly-real output in O(N log N) operations

Quality Diagnostic (Eq. 12):
    ε_Im = max|Im(f̃)| / max|Re(f̃)|

For real f(t) with correct Hermitian construction, the IFFT output should be
nearly real. Small ε_Im (< 10^{-2}, Eq. 13) indicates good quality.
Large ε_Im indicates aliasing, truncation, or parameter issues.

CFL-like Feasibility Condition (Eq. 10):
    α_c * t_max + ln(C/ε_tail) < L - δ_s

where:
- Constraint 1 (dynamic range): a*t_max < L - δ_s prevents overflow
- Constraint 2 (spectral placement): a > α_c ensures contour right of singularities
- Constraint 3 (aliasing suppression): adequate (a-α_c) ensures wraparound decay

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
    Compute inverse Laplace transform using FFT with half-spectrum Hermitian construction.

    For real f(t), exploits conjugate symmetry F(a-iω) = conj(F(a+iω)):
    - Evaluates F only at positive frequencies k = 0, 1, ..., N/2
    - Constructs negative frequency bins via conjugate mirroring
    - Applies IFFT to get nearly-real output

    This avoids evaluating F at negative frequencies (which can overflow
    for certain transfer functions) while ensuring proper DFT structure.

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
        Complex IFFT output (should be nearly real for well-tuned parameters)

    Notes
    -----
    For CFL-tuned parameters (a, T) satisfying the feasibility conditions:
    - a > α_c + δ_min (spectral placement)
    - a < (L - δ_s)/(2T) (dynamic range)
    - a ≥ α_c + ln(C/ε_tail)/(2T-t_end) (aliasing suppression)

    The imaginary leakage ε_Im = max|Im|/max|Re| should be small (< 0.01)
    for well-tuned parameters, indicating the IFFT output is nearly real.
    """
    # Frequency spacing: Δω = π/T
    delta_omega = np.pi / T

    # Time step: Δt = 2T/N
    delta_t = 2 * T / N

    # Time grid: t_j = j * Δt for j = 0, ..., N-1
    t = np.arange(N) * delta_t

    # === Half-spectrum approach ===
    # Only evaluate F at positive frequencies k = 0, 1, ..., N/2
    n_pos = N // 2 + 1
    omega_pos = np.arange(n_pos) * delta_omega  # ω = 0, Δω, 2Δω, ..., N/2 * Δω
    s_pos = a + 1j * omega_pos

    # Evaluate F only at positive frequencies (avoids overflow at negative ω)
    G_pos = np.array([F(sk) for sk in s_pos], dtype=np.complex128)

    # Apply trapezoidal weight at DC (k=0 endpoint)
    G_pos[0] = G_pos[0] * 0.5

    # Construct full Hermitian spectrum via conjugate mirroring
    # For DFT: G[k] for k=0..N/2, then G[N-k] = conj(G[k]) for k=1..N/2-1
    if N % 2 == 0:
        # Even N: G = [G[0], G[1], ..., G[N/2], conj(G[N/2-1]), ..., conj(G[1])]
        G = np.concatenate([G_pos, np.conj(G_pos[-2:0:-1])])
    else:
        # Odd N: G = [G[0], G[1], ..., G[(N-1)/2], conj(G[(N-1)/2]), ..., conj(G[1])]
        G = np.concatenate([G_pos, np.conj(G_pos[-1:0:-1])])

    # Compute sum via IFFT
    # Hermitian spectrum → nearly real IFFT output
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
    DEPRECATED: One-sided positive-frequency implementation.

    This implementation has a frequency grid mapping error: it treats all
    N bins as positive frequencies, but IFFT interprets bins k>N/2 as
    negative frequencies. This causes high imaginary leakage by construction.

    Kept for backwards compatibility and comparison purposes only.
    Use fft_nilt() instead for correct DFT-consistent implementation.

    Parameters
    ----------
    F : callable
        Laplace-domain transfer function
    a : float
        Bromwich shift parameter
    T : float
        Half-period
    N : int
        Number of FFT points

    Returns
    -------
    f, t, z_ifft : tuple
        Same as fft_nilt(), but with higher imaginary leakage
    """
    delta_omega = np.pi / T
    delta_t = 2 * T / N
    t = np.arange(N) * delta_t

    # OLD (incorrect): All positive frequencies
    omega = np.arange(N) * delta_omega
    s = a + 1j * omega

    G = np.array([F(sk) for sk in s], dtype=np.complex128)
    G[0] = G[0] / 2  # Trapezoidal weight for DC

    z_ifft = N * np.fft.ifft(G)
    f = np.exp(a * t) / T * np.real(z_ifft)

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
