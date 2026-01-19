"""
FFT-NILT CFL Tuning Reproducibility Package.

This package provides all code to reproduce results from:
"CFL-Informed Parameter Tuning for FFT-Based Numerical Inverse Laplace Transform"
"""

from .nilt_fft import fft_nilt, eps_im, n_doubling_error
from .tuner import tune_params, refine_until_accept, check_cfl_feasibility, TunedParams
from .problems import Problem, get_problem, get_all_problems
from .metrics import rmse, relative_rmse, max_absolute_error, max_relative_error

__all__ = [
    'fft_nilt',
    'eps_im',
    'n_doubling_error',
    'tune_params',
    'refine_until_accept',
    'check_cfl_feasibility',
    'TunedParams',
    'Problem',
    'get_problem',
    'get_all_problems',
    'rmse',
    'relative_rmse',
    'max_absolute_error',
    'max_relative_error',
]

__version__ = '1.0.0'
