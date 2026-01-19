"""
Generate all figures for Paper 2: FFT-NILT with CFL Tuning

Produces publication-quality figures:
- Fig 1: Feasible region (a_min, a_max vs t_max)
- Fig 2: Pareto curves (accuracy vs runtime)
- Fig 3: Case study breakthrough curves (BC comparison)
- Fig 4: Parameter estimation demonstration
- Fig 5: Ablation study (constraint violation effects)
- Fig 6: N-doubling convergence + ε_Im diagnostic

Author: Gorgi Pavlov
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from nilt_fft import fft_nilt, eps_im
from tuner import tune_params
from problems import get_problem, get_all_problems
from benchmark_pareto import benchmark_fft_nilt, benchmark_de_hoog
from case_study_adsorption import (
    BCType, ColumnParams, breakthrough_transfer_function,
    compute_breakthrough_nilt
)


def make_column_params(Pe: float, tau: float = 1.0) -> ColumnParams:
    """Create ColumnParams from dimensionless Pe and tau values.

    Sets physical parameters to achieve desired Pe and tau:
    - Pe = v*L/D_L  →  L=1, v=Pe, D_L=1 gives Pe
    - tau = R*L/v   →  With R=1 (no adsorption), tau = L/v = 1/Pe

    To get tau=1.0 with arbitrary Pe, we use:
    - L = 1.0
    - v = 1.0/tau
    - D_L = 1/(Pe*tau)
    - epsilon = 0.5, K_H = 0 (so R = 1)
    """
    return ColumnParams(
        L=1.0,
        v=1.0 / tau,
        D_L=1.0 / (Pe * tau),
        epsilon=0.5,
        K_H=0.0,  # No adsorption, R=1
        C_0=1.0
    )


from ablation_study import (
    test_remove_feasibility_bound,
    test_remove_aliasing_constraint,
    test_n_doubling_catches_failures
)

# Publication-quality plot settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.5,
    'axes.linewidth': 0.8,
    'grid.alpha': 0.3,
    'text.usetex': False,  # Set True if LaTeX available
})

# Color palette (colorblind-friendly)
COLORS = {
    'fft_nilt': '#1f77b4',   # Blue
    'de_hoog': '#ff7f0e',    # Orange
    'analytical': '#2ca02c',  # Green
    'error': '#d62728',       # Red
    'danckwerts': '#1f77b4',
    'robin': '#ff7f0e',
    'dirichlet': '#2ca02c',
}


def fig01_feasible_region(
    alpha_c_values: list = [-1.0, 0.0, 0.5],
    t_max_range: tuple = (1.0, 100.0),
    save_path: str = None
) -> plt.Figure:
    """
    Figure 1: Feasible region in (t_max, a) space.
    Shows CFL-like stability region for NILT parameter selection.
    """
    C = 1.0
    eps_tail = 1e-6
    delta_floor = 1e-3
    delta_s = 10.0
    L = 709.8  # ln(DBL_MAX) ≈ 709.8

    fig, ax = plt.subplots(figsize=(7, 5))
    t_max_vals = np.linspace(t_max_range[0], t_max_range[1], 200)

    linestyles = ['-', '--', ':']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for idx, alpha_c in enumerate(alpha_c_values):
        a_min_vals = []
        a_max_vals = []

        for t_max in t_max_vals:
            kappa = 1.0
            t_end = t_max / 2
            alias_factor = (2 * kappa - 1) * t_end
            if alias_factor > 0:
                a_alias = alpha_c + np.log(C / eps_tail) / alias_factor
            else:
                a_alias = alpha_c + delta_floor

            a_sing = alpha_c + delta_floor
            a_min_star = max(a_alias, a_sing)
            a_min = max(a_min_star, delta_floor)
            a_max = (L - delta_s) / t_max

            a_min_vals.append(a_min)
            a_max_vals.append(a_max)

        a_min_vals = np.array(a_min_vals)
        a_max_vals = np.array(a_max_vals)

        label_min = f'$a_{{\\min}}$ ($\\alpha_c={alpha_c}$)'
        ax.plot(t_max_vals, a_min_vals, color=colors[idx], linestyle=linestyles[idx],
                linewidth=2, label=label_min)

        if idx == 0:
            ax.plot(t_max_vals, a_max_vals, 'k-', linewidth=2, label='$a_{\\max}$')

            # Shade feasible region
            feasible_mask = a_min_vals <= a_max_vals
            if np.any(feasible_mask):
                t_feas = t_max_vals[feasible_mask]
                a_min_feas = a_min_vals[feasible_mask]
                a_max_feas = a_max_vals[feasible_mask]
                verts = list(zip(t_feas, a_min_feas)) + list(zip(t_feas[::-1], a_max_feas[::-1]))
                poly = Polygon(verts, alpha=0.15, facecolor='green', edgecolor='none',
                              label='Feasible region')
                ax.add_patch(poly)

    ax.axhline(y=delta_floor, color='red', linestyle='-.', alpha=0.7, linewidth=1.5)
    ax.annotate(f'$\\delta_{{floor}}$', xy=(t_max_range[1]*0.9, delta_floor*2),
                fontsize=9, color='red')

    ax.set_xlabel('$t_{\\max} = 2T$')
    ax.set_ylabel('Bromwich shift $a$')
    ax.set_title('CFL Feasibility Region (Theorem 1)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(t_max_range)
    ax.set_ylim(0, 50)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


def fig02_pareto_curves(
    problems: list = None,
    save_path: str = None
) -> plt.Figure:
    """
    Figure 2: Pareto curves (accuracy vs runtime) for multiple problems.
    Compares FFT-NILT vs de Hoog algorithm.
    """
    if problems is None:
        problems = ["lag", "secondorder", "diffusion"]

    N_values = [128, 256, 512, 1024, 2048, 4096]
    M_values = [10, 15, 20, 25, 30]

    fig, axes = plt.subplots(1, len(problems), figsize=(4*len(problems), 4))
    if len(problems) == 1:
        axes = [axes]

    for ax, prob_name in zip(axes, problems):
        # Benchmark both methods
        fft_results = benchmark_fft_nilt(prob_name, N_values, t_end=10.0, n_timing_runs=20)
        dehoog_results = benchmark_de_hoog(prob_name, M_values, t_end=10.0, n_timing_runs=5)

        # Extract data
        fft_times = [r.timing_us for r in fft_results]
        fft_rmse = [r.rmse for r in fft_results]
        fft_N = [r.N for r in fft_results]

        dehoog_times = [r.timing_us for r in dehoog_results]
        dehoog_rmse = [r.rmse for r in dehoog_results]
        dehoog_M = [r.N for r in dehoog_results]

        # Plot
        ax.loglog(fft_times, fft_rmse, 'o-', color=COLORS['fft_nilt'],
                  markersize=6, label='FFT-NILT')
        ax.loglog(dehoog_times, dehoog_rmse, 's--', color=COLORS['de_hoog'],
                  markersize=6, label='de Hoog')

        # Annotate some points
        for i, (t, e, n) in enumerate(zip(fft_times, fft_rmse, fft_N)):
            if i % 2 == 0:
                ax.annotate(f'N={n}', (t, e), textcoords="offset points",
                           xytext=(5, 5), fontsize=7, color=COLORS['fft_nilt'])

        ax.set_xlabel('Runtime (μs)')
        ax.set_ylabel('RMSE')
        ax.set_title(f'{prob_name.capitalize()}')
        ax.legend(loc='lower left', fontsize=8)
        ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


def fig03_breakthrough_comparison(
    Pe_values: list = None,
    save_path: str = None
) -> plt.Figure:
    """
    Figure 3: Breakthrough curves comparing three boundary condition formulations.
    """
    if Pe_values is None:
        Pe_values = [10, 50, 200]

    fig, axes = plt.subplots(1, len(Pe_values), figsize=(4*len(Pe_values), 4))
    if len(Pe_values) == 1:
        axes = [axes]

    bc_types = [BCType.DANCKWERTS, BCType.ROBIN_NEUMANN, BCType.DIRICHLET_NEUMANN]
    bc_labels = ['Danckwerts (closed)', 'Robin-Neumann', 'Dirichlet-Neumann']
    bc_colors = [COLORS['danckwerts'], COLORS['robin'], COLORS['dirichlet']]
    # Use distinct line styles and markers to distinguish overlapping curves
    bc_styles = ['-', '--', ':']
    bc_markers = ['o', 's', '^']

    for ax, Pe in zip(axes, Pe_values):
        params = make_column_params(Pe=Pe, tau=1.0)

        for bc_type, label, color, style, marker in zip(
            bc_types, bc_labels, bc_colors, bc_styles, bc_markers
        ):
            result = compute_breakthrough_nilt(
                params, bc_type=bc_type, t_end=3.0, N=1024
            )
            t_arr = result['t']
            C_arr = result['C']
            # Subsample markers to avoid clutter
            marker_every = max(1, len(t_arr) // 15)
            ax.plot(t_arr / params.tau, C_arr, linestyle=style, color=color,
                   linewidth=1.5, marker=marker, markersize=4,
                   markevery=marker_every, label=label)

        ax.set_xlabel('Dimensionless time $t/\\tau$')
        ax.set_ylabel('$C/C_0$')
        ax.set_title(f'Pe = {Pe}')
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='upper left', fontsize=7, framealpha=0.9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


def fig04_parameter_estimation(
    Pe_true: float = 50.0,
    noise_level: float = 0.02,
    save_path: str = None
) -> plt.Figure:
    """
    Figure 4: Parameter estimation demonstration with synthetic data.
    """
    from case_study_adsorption import estimate_peclet_nilt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Generate synthetic data with noise
    params_true = make_column_params(Pe=Pe_true, tau=1.0)
    bt_result = compute_breakthrough_nilt(
        params_true, bc_type=BCType.DANCKWERTS, t_end=3.0, N=1024
    )
    t_data = bt_result['t']
    C_true = bt_result['C']

    # Add noise
    np.random.seed(42)
    C_noisy = C_true + noise_level * np.random.randn(len(C_true))
    C_noisy = np.clip(C_noisy, 0, 1)

    # Estimate Pe
    est_result = estimate_peclet_nilt(t_data, C_noisy, params_true)
    Pe_estimated = est_result['Pe_est']

    # Reconstruct fit
    params_fit = make_column_params(Pe=Pe_estimated, tau=params_true.tau)
    fit_result = compute_breakthrough_nilt(
        params_fit, bc_type=BCType.DANCKWERTS, t_end=3.0, N=1024
    )
    C_fit = fit_result['C']

    # Compute RMSE
    rmse = np.sqrt(np.mean((C_noisy - np.interp(t_data, fit_result['t'], C_fit))**2))

    # Left plot: Data and fit
    ax1.plot(t_data / params_true.tau, C_noisy, 'o', color='gray',
             markersize=3, alpha=0.5, label='Noisy data')
    ax1.plot(t_data / params_true.tau, C_true, '-', color=COLORS['analytical'],
             linewidth=2, label=f'True (Pe={Pe_true})')
    ax1.plot(t_data / params_true.tau, C_fit, '--', color=COLORS['fft_nilt'],
             linewidth=2, label=f'NILT fit (Pe={Pe_estimated:.1f})')

    ax1.set_xlabel('Dimensionless time $t/\\tau$')
    ax1.set_ylabel('$C/C_0$')
    ax1.set_title('Parameter Estimation via NILT')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Right plot: Residual analysis
    residuals = C_noisy - np.interp(t_data, t_data, C_fit)
    ax2.plot(t_data / params_true.tau, residuals, 'o', color='gray',
             markersize=3, alpha=0.6)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
    ax2.axhline(y=noise_level, color='r', linestyle='--', alpha=0.7,
                label=f'±{noise_level} noise level')
    ax2.axhline(y=-noise_level, color='r', linestyle='--', alpha=0.7)

    ax2.set_xlabel('Dimensionless time $t/\\tau$')
    ax2.set_ylabel('Residual')
    ax2.set_title(f'Residuals (RMSE = {rmse:.4f})')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Add text box with results
    textstr = f'True Pe = {Pe_true}\nEstimated Pe = {Pe_estimated:.2f}\nError = {100*abs(Pe_estimated-Pe_true)/Pe_true:.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=props)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


def fig05_ablation_study(save_path: str = None) -> plt.Figure:
    """
    Figure 5: Ablation study showing what happens when CFL constraints are violated.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel A: Feasibility violation - show exponential growth risk
    ax1 = axes[0]
    result_feas = test_remove_feasibility_bound(verbose=False)

    # Show exponential growth at different a values
    t_max = 10.0
    a_values = [5, 20, 50, 100]
    for a in a_values:
        y = np.exp(a * np.linspace(0, t_max * 0.5, 50))
        y_clipped = np.clip(y, 1, 1e20)
        color = COLORS['fft_nilt'] if a <= 20 else COLORS['error']
        style = '-' if a <= 20 else '--'
        ax1.semilogy(np.linspace(0, t_max * 0.5, 50), y_clipped, style,
                     color=color, linewidth=1.5, alpha=0.7,
                     label=f'$a = {a}$')

    ax1.axhline(y=1e15, color='red', linestyle=':', linewidth=1.5)
    ax1.annotate('Practical overflow\n($a \\cdot t > 35$)', xy=(4, 5e15), fontsize=8, color='red')

    ax1.set_xlabel('Time $t$')
    ax1.set_ylabel('$\\exp(at)$')
    ax1.set_title('(a) Feasibility Bound: Overflow Risk')
    ax1.legend(loc='upper left', fontsize=7)
    ax1.set_ylim(1, 1e20)
    ax1.grid(True, alpha=0.3)

    # Panel B: Aliasing violation
    ax2 = axes[1]

    # Compute with and without aliasing margin
    problem = get_problem("lag")
    tuned = tune_params(t_end=10.0, alpha_c=problem.alpha_c, C=problem.C)

    # Good parameters
    f_good, t_good, _ = fft_nilt(problem.F, tuned.a, tuned.T, 512)
    f_ref = problem.f_ref(t_good)

    # Bad parameters (aliasing)
    a_bad = problem.alpha_c + 0.001
    f_bad, t_bad, _ = fft_nilt(problem.F, a_bad, tuned.T, 512)

    mask = (t_good > 0.1) & (t_good <= 10)
    ax2.plot(t_good[mask], f_ref[mask], '-', color=COLORS['analytical'],
             linewidth=2, label='Analytical')
    ax2.plot(t_good[mask], f_good[mask], '--', color=COLORS['fft_nilt'],
             linewidth=1.5, label='CFL-tuned')
    ax2.plot(t_bad[mask], f_bad[mask], ':', color=COLORS['error'],
             linewidth=1.5, label='Aliasing violated')

    ax2.set_xlabel('Time $t$')
    ax2.set_ylabel('$f(t)$')
    ax2.set_title('(b) Aliasing Constraint Violation')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel C: N-doubling convergence
    ax3 = axes[2]
    n_double_results = test_n_doubling_catches_failures(verbose=False)

    N_vals = [r['N'] for r in n_double_results]
    E_N_vals = [r['E_N'] for r in n_double_results]
    eps_im_vals = [r['eps_im'] for r in n_double_results]

    ax3.semilogy(N_vals, E_N_vals, 'o-', color=COLORS['fft_nilt'],
                 linewidth=2, markersize=6, label='$E_N$ (N-doubling)')
    ax3.semilogy(N_vals, eps_im_vals, 's--', color=COLORS['de_hoog'],
                 linewidth=2, markersize=6, label='$\\varepsilon_{Im}$')

    ax3.axhline(y=1e-2, color='orange', linestyle=':', linewidth=1.5,
                label='Acceptance threshold')

    ax3.set_xlabel('FFT size $N$')
    ax3.set_ylabel('Error metric')
    ax3.set_title('(c) Convergence Diagnostics')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


def fig06_eps_im_diagnostic(
    problem_name: str = "secondorder",
    save_path: str = None
) -> plt.Figure:
    """
    Figure 6: ε_Im diagnostic behavior across parameter space.
    """
    problem = get_problem(problem_name)
    tuned = tune_params(t_end=10.0, alpha_c=problem.alpha_c, C=problem.C)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: ε_Im vs N for fixed a
    N_values = [64, 128, 256, 512, 1024, 2048, 4096]
    eps_im_vs_N = []
    rmse_vs_N = []

    for N in N_values:
        f, t, z = fft_nilt(problem.F, tuned.a, tuned.T, N)
        mask = (t > 0.1) & (t <= 10)
        eps_im_vs_N.append(eps_im(z[mask]))
        if problem.f_ref is not None:
            f_ref = problem.f_ref(t)
            rmse_vs_N.append(np.sqrt(np.mean((f[mask] - f_ref[mask])**2)))
        else:
            rmse_vs_N.append(np.nan)

    ax1.semilogy(N_values, eps_im_vs_N, 'o-', color=COLORS['fft_nilt'],
                 linewidth=2, markersize=6, label='$\\varepsilon_{Im}$')
    ax1.semilogy(N_values, rmse_vs_N, 's--', color=COLORS['error'],
                 linewidth=2, markersize=6, label='RMSE')
    ax1.axhline(y=1e-2, color='orange', linestyle=':', linewidth=1.5,
                label='Threshold')

    ax1.set_xlabel('FFT size $N$')
    ax1.set_ylabel('Error metric')
    ax1.set_title(f'(a) Convergence: {problem_name}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, which='both')

    # Panel B: ε_Im vs a for fixed N
    a_values = np.linspace(tuned.a * 0.5, tuned.a * 2, 20)
    a_values = a_values[(a_values > problem.alpha_c + 0.01) & (a_values < tuned.a_max)]
    eps_im_vs_a = []
    rmse_vs_a = []

    N_fixed = 1024
    for a in a_values:
        T = tuned.T  # Keep T fixed
        f, t, z = fft_nilt(problem.F, a, T, N_fixed)
        mask = (t > 0.1) & (t <= 10)
        eps_im_vs_a.append(eps_im(z[mask]))
        if problem.f_ref is not None:
            f_ref = problem.f_ref(t)
            rmse_vs_a.append(np.sqrt(np.mean((f[mask] - f_ref[mask])**2)))
        else:
            rmse_vs_a.append(np.nan)

    ax2.semilogy(a_values, eps_im_vs_a, 'o-', color=COLORS['fft_nilt'],
                 linewidth=2, markersize=5, label='$\\varepsilon_{Im}$')
    ax2.semilogy(a_values, rmse_vs_a, 's--', color=COLORS['error'],
                 linewidth=2, markersize=5, label='RMSE')
    ax2.axvline(x=tuned.a, color='green', linestyle='-.', linewidth=1.5,
                alpha=0.7, label=f'$a^*$ = {tuned.a:.2f}')
    ax2.axhline(y=1e-2, color='orange', linestyle=':', linewidth=1.5)

    ax2.set_xlabel('Bromwich shift $a$')
    ax2.set_ylabel('Error metric')
    ax2.set_title(f'(b) Parameter sensitivity (N={N_fixed})')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


def generate_all_figures(output_dir: str = "figures") -> None:
    """
    Generate all paper figures to output directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print("=" * 60)
    print("Generating figures for Paper 2: FFT-NILT with CFL Tuning")
    print("=" * 60)

    # Figure 1: Feasible region
    print("\n[1/6] Generating feasible region plot...")
    fig01_feasible_region(save_path=str(output_path / "fig01_feasible_region.png"))
    plt.close()

    # Figure 2: Pareto curves
    print("[2/6] Generating Pareto curves...")
    fig02_pareto_curves(save_path=str(output_path / "fig02_pareto_curves.png"))
    plt.close()

    # Figure 3: Breakthrough comparison
    print("[3/6] Generating breakthrough curve comparison...")
    fig03_breakthrough_comparison(save_path=str(output_path / "fig03_breakthrough_bc.png"))
    plt.close()

    # Figure 4: Parameter estimation
    print("[4/6] Generating parameter estimation demo...")
    fig04_parameter_estimation(save_path=str(output_path / "fig04_param_estimation.png"))
    plt.close()

    # Figure 5: Ablation study
    print("[5/6] Generating ablation study...")
    fig05_ablation_study(save_path=str(output_path / "fig05_ablation.png"))
    plt.close()

    # Figure 6: ε_Im diagnostic
    print("[6/6] Generating diagnostic behavior plot...")
    fig06_eps_im_diagnostic(save_path=str(output_path / "fig06_diagnostics.png"))
    plt.close()

    print("\n" + "=" * 60)
    print(f"All figures saved to: {output_path.absolute()}")
    print("=" * 60)

    # List generated files
    print("\nGenerated files:")
    for f in sorted(output_path.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    generate_all_figures()
