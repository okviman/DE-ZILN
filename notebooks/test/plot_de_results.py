"""
Script to plot TPR and FPR from the CSV output of vishd_test_de_parallel.py.

Usage:
    python plot_de_results.py --input results.csv --output plot.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Use matplotlib's built-in math rendering (doesn't require LaTeX)
# This provides LaTeX-style formatting without needing LaTeX installed
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern style
matplotlib.rcParams['font.family'] = 'serif'


def rename_method(method):
    """Rename method names for display."""
    method_map = {
        'DELN': r"LN's $t$-test",
        'Scanpy t-test': r'$t$-test',
        'Scanpy wilcoxon': 'Wilcoxon'
    }
    return method_map.get(method, method)


def compute_log10_error_bars(mean, std):
    """
    Compute error bars for log10 scale.
    
    Parameters
    ----------
    mean : array-like
        Mean values
    std : array-like
        Standard deviation values
    
    Returns
    -------
    log_mean : array-like
        log10 of mean values
    log_err_lower : array-like
        Lower error bars (log_mean - log10(max(mean - std, small_value)))
    log_err_upper : array-like
        Upper error bars (log10(mean + std) - log_mean)
    """
    mean = np.asarray(mean)
    std = np.asarray(std)
    
    # Small value to avoid log(0) or log(negative)
    eps = 1e-10
    
    # Compute log10 of means
    log_mean = np.log10(np.maximum(mean, eps))
    
    # Compute upper and lower bounds
    lower_bound = np.maximum(mean - std, eps)
    upper_bound = mean + std
    
    log_err_lower = log_mean - np.log10(lower_bound)
    log_err_upper = np.log10(upper_bound) - log_mean
    
    return log_mean, log_err_lower, log_err_upper


def plot_de_results(csv_file, output_file, title_suffix=None):
    """
    Plot TPR and FPR from CSV results.
    
    Parameters
    ----------
    csv_file : str
        Path to input CSV file
    output_file : str
        Path to output plot file
    title_suffix : str, optional
        Text to append to all subplot titles (default: None)
    """
    # Load data
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Get unique methods and p values
    methods = df['method'].unique()
    p_values = sorted(df['p'].unique())
    
    print(f"Found {len(methods)} methods: {methods}")
    print(f"Found {len(p_values)} p values")
    
    # Rename methods
    method_renames = {method: rename_method(method) for method in methods}
    
    # Set up colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    method_colors = {method: colors[i] for i, method in enumerate(methods)}
    
    # Create figure with 2 subplots (side by side)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot TPR
    ax_tpr = axes[0]
    for method in methods:
        method_data = df[df['method'] == method].sort_values('p')
        tpr_means = method_data['tpr_mean'].values
        tpr_stds = method_data['tpr_std'].values
        p_vals = method_data['p'].values
        
        ax_tpr.errorbar(
            p_vals, tpr_means, yerr=tpr_stds,
            marker='o', capsize=5, capthick=2,
            label=method_renames[method],
            color=method_colors[method],
            linewidth=2, markersize=6
        )
    
    # Prepare title text
    tpr_title = 'TPR vs Downsample Ratio'
    fpr_title = 'FPR vs Downsample Ratio'
    if title_suffix:
        tpr_title = f'{tpr_title} {title_suffix}'
        fpr_title = f'{fpr_title} {title_suffix}'
    
    ax_tpr.set_xlabel(r'$p$', fontsize=14)
    ax_tpr.set_ylabel('True Positive Rate (TPR)', fontsize=14)
    ax_tpr.set_title(tpr_title, fontsize=16, fontweight='bold')
    ax_tpr.legend(loc='best', fontsize=11)
    ax_tpr.set_ylim([0, 1])
    
    # Plot FPR (log10 scale)
    ax_fpr = axes[1]
    for method in methods:
        method_data = df[df['method'] == method].sort_values('p')
        fpr_means = method_data['fpr_mean'].values
        fpr_stds = method_data['fpr_std'].values
        p_vals = method_data['p'].values
        
        # Compute log10 values and error bars
        log_fpr_mean, log_err_lower, log_err_upper = compute_log10_error_bars(
            fpr_means, fpr_stds
        )
        
        # Use asymmetric error bars
        ax_fpr.errorbar(
            p_vals, log_fpr_mean,
            yerr=[log_err_lower, log_err_upper],
            marker='o', capsize=5, capthick=2,
            label=method_renames[method],
            color=method_colors[method],
            linewidth=2, markersize=6
        )
    
    ax_fpr.set_xlabel(r'$p$', fontsize=14)
    ax_fpr.set_ylabel(r'$\log_{10}$(FPR)', fontsize=14)
    ax_fpr.set_title(fpr_title, fontsize=16, fontweight='bold')
    ax_fpr.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    # Ensure output file has .eps extension
    if not output_file.endswith('.eps'):
        output_file = output_file.rsplit('.', 1)[0] + '.eps'
    plt.savefig(output_file, format='eps', dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot TPR and FPR from DE test results CSV'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV file (output from vishd_test_de_parallel.py)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output plot file (will be saved as .eps)'
    )
    parser.add_argument(
        '--title_suffix',
        type=str,
        default=None,
        help='Text to append to all subplot titles (default: None)'
    )
    
    args = parser.parse_args()
    
    plot_de_results(args.input, args.output, title_suffix=args.title_suffix)

