# CFL-Informed Parameter Tuning for FFT-Based Numerical Inverse Laplace Transform

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



## Overview

The FFT-based numerical inverse Laplace transform (NILT) offers O(N log N) efficiency but requires careful parameter tuning. This work develops a systematic parameter selection framework based on three CFL-like conditions:

1. **Dynamic-range feasibility**: Prevents floating-point overflow
2. **Spectral placement**: Ensures Bromwich contour validity
3. **Aliasing suppression**: Controls wraparound error

The framework includes:
- Deterministic parameter selection algorithm (Algorithm 1)
- Quality diagnostics (εIm + N-doubling convergence tests)
- Validation on distributed-parameter transfer functions

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete reproduction (figures + tables + case study)
python reproduce_paper.py

# Generate figures only
python reproduce_paper.py --figures

# Run case study only
python reproduce_paper.py --case-study
```

## Repository Structure

```
nilt-cfl/
├── Core Implementation
│   ├── nilt_fft.py          # FFT-based NILT (Eq. 3-6)
│   ├── tuner.py             # CFL-informed parameter selection (Algorithm 1)
│   ├── problems.py          # Benchmark transfer functions (Table 4)
│   └── metrics.py           # Error metrics and diagnostics
│
├── Case Study
│   └── case_study_adsorption.py  # Fixed-bed breakthrough (Section 5.5)
│
├── Benchmarks
│   ├── benchmark_pareto.py           # Accuracy vs runtime Pareto curves
│   ├── ablation_study.py             # CFL constraint ablation (Fig. 5)
│   └── benchmark_numpy_jax_torch.py  # GPU comparison (Table 8)
│
├── Reproduction
│   ├── reproduce_paper.py        # Master script for all results
│   └── generate_paper_figures.py # Publication-quality figures
│
├── Paper
│   ├── paper2_nilt_cfl.tex       # LaTeX manuscript
│   ├── paper2_nilt_cfl.pdf       # Compiled PDF
│   └── fig*.png                  # Paper figures
│
└── figures/                      # Generated figures
```

## Reproducing Paper Results

### All Figures (Figures 1-6)

```bash
python generate_paper_figures.py
```

Outputs to `figures/`:
- `fig01_feasible_region.png` - CFL feasibility region (Theorem 1)
- `fig02_pareto_curves.png` - Accuracy vs runtime Pareto frontiers
- `fig03_breakthrough_bc.png` - Breakthrough curves, BC comparison
- `fig04_param_estimation.png` - Parameter estimation demonstration
- `fig05_ablation.png` - CFL constraint ablation study
- `fig06_diagnostics.png` - Diagnostic metric behavior

### Table 5: Accuracy Comparison

```bash
python repro_nilt_cfl.py --all
```

### Table 6: FFT-NILT vs de Hoog

```bash
python benchmark_pareto.py
```

### Table 8: GPU Benchmark

```bash
python benchmark_numpy_jax_torch.py --n-runs 50 --pretty
```

Note: Requires NVIDIA GPU with CUDA for meaningful results.

### Case Study (Section 5.5)

```bash
python case_study_adsorption.py
```

Demonstrates:
- Breakthrough curve computation for three BC types
- Parameter estimation workflow (~8× speedup vs MOL)
- BC comparison at different Péclet numbers

## Benchmark Problems

| Problem | F(s) | αc | Description |
|---------|------|-----|-------------|
| `lag` | K/(τs + 1) | -1/τ | First-order lag |
| `fopdt` | K·exp(-θs)/(τs + 1) | -1/τ | First-order plus dead time |
| `secondorder` | ωn²/(s² + 2ζωn·s + ωn²) | -ζωn | Second-order underdamped |
| `diffusion` | exp(-x√(s/D))/s | 0 | Semi-infinite diffusion |
| `packedbed` | exp(Pe/2·(1-√(1+4s/Pe))) | 0 | Axial dispersion |

## Requirements

- Python 3.9+
- NumPy, SciPy, Matplotlib
- Optional: JAX, PyTorch (for GPU benchmarks)

Install via:
```bash
pip install -r requirements.txt
```

## API Usage

### Basic NILT Evaluation

```python
from nilt_fft import fft_nilt
from tuner import tune_params
from problems import get_problem

# Get a test problem
problem = get_problem('lag')

# Auto-tune parameters
tuned = tune_params(
    t_end=10.0,
    alpha_c=problem.alpha_c,
    C=problem.C,
    rho=problem.rho
)

# Compute inverse transform
f, t, z = fft_nilt(problem.F, tuned.a, tuned.T, tuned.N)
```

### Breakthrough Curve Computation

```python
from case_study_adsorption import ColumnParams, compute_breakthrough_nilt, BCType

# Define column parameters
params = ColumnParams(
    L=0.1,        # Column length [m]
    v=0.01,       # Velocity [m/s]
    D_L=1e-4,     # Dispersion coefficient [m²/s]
    epsilon=0.4,  # Porosity
    K_H=1.0,      # Henry constant
    C_0=1.0       # Inlet concentration
)

# Compute breakthrough curve
result = compute_breakthrough_nilt(
    params,
    t_end=75.0,
    bc_type=BCType.DANCKWERTS
)

print(f"Pe = {params.Pe:.1f}, τ = {params.tau:.1f} s")
```

## Citation

```bibtex
@article{pavlov2025nilt,
  title={Systematic Parameter Selection for FFT-based Numerical Inverse
         Laplace Transform: CFL-informed Tuning Rules and Quality Diagnostics},
  author={Pavlov, Gorgi},
}
```

## License

MIT License - see LICENSE file for details.

## Contact

Gorgi Pavlov - gop214@lehigh.edu
