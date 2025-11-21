import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Use matplotlib's built-in math rendering (doesn't require LaTeX)
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern style
plt.rcParams['font.family'] = 'serif'


def plot_fpr_results(csv_file, output_file, aspect_ratio=1.0, title_suffix=None):
    """
    Plot log10(FPR) vs p from CSV results file.
    
    Parameters
    ----------
    csv_file : str
        Path to input CSV file
    output_file : str
        Path to output plot file
    aspect_ratio : float
        Aspect ratio for the plot (default: 1.0)
    title_suffix : str, optional
        Text to append to the plot title (default: None)
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Map method names to LaTeX-friendly names
    method_mapping = {
        'DELN': r"LN's $t$-test",
        'Scanpy t-test': r'$t$-test',
        'Scanpy wilcoxon': 'Wilcoxon'
    }
    
    # Apply method name mapping
    df['method_display'] = df['method'].map(method_mapping).fillna(df['method'])
    
    # Get unique methods
    methods = df['method_display'].unique()
    p_values = sorted(df['p'].unique())
    
    # Create color map
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    method_colors = {method: colors[i] for i, method in enumerate(methods)}
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot each method
    # Use raw FPR values and plot on log scale to avoid epsilon capping issues
    for method in methods:
        method_data = df[df['method_display'] == method]
        method_data = method_data.sort_values('p')
        
        fpr_means = method_data['fpr_mean'].values
        fpr_stds = method_data['fpr_std'].values
        p_vals = method_data['p'].values
        
        # Handle zero FPR values by setting a minimum for plotting
        # Use asymmetric error bars: only show positive error bars to avoid going below log scale minimum
        epsilon = 1e-10
        fpr_means_plot = np.maximum(fpr_means, epsilon)  # Minimum for log scale
        
        # Compute error bar bounds in linear space
        fpr_lower = np.maximum(fpr_means - fpr_stds, epsilon)
        fpr_upper = np.maximum(fpr_means + fpr_stds, epsilon)  # Also cap upper bound
        
        # Convert to log10 for plotting
        fpr_means_log10 = np.log10(fpr_means_plot)
        fpr_lower_log10 = np.log10(fpr_lower)
        fpr_upper_log10 = np.log10(fpr_upper)
        
        # Asymmetric error bars: ensure non-negative
        # Lower error: how far down from mean (should be non-negative since fpr_lower <= fpr_means)
        yerr_lower = np.maximum(0, fpr_means_log10 - fpr_lower_log10)
        # Upper error: how far up from mean (should be non-negative since fpr_upper >= fpr_means)
        yerr_upper = np.maximum(0, fpr_upper_log10 - fpr_means_log10)
        
        ax.errorbar(p_vals, fpr_means_log10, 
                   yerr=[yerr_lower, yerr_upper], 
                   marker='o', capsize=5, capthick=2, label=method, 
                   color=method_colors[method], linewidth=2, markersize=6)
    
    # Prepare title text
    title = r'FPR vs Sampling probability'
    if title_suffix:
        title = f'{title} {title_suffix}'
    
    ax.set_xlabel(r'Sampling probability $p$', fontsize=18)
    ax.set_ylabel(r'$\log_{10}(\mathrm{FPR})$', fontsize=18)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_aspect(aspect_ratio, adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_file, format='eps', dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot FPR results from CSV file')
    parser.add_argument('--csv_file', type=str, required=True,
                        help='Input CSV file with FPR results')
    parser.add_argument('--output', type=str, default='fpr_plot.eps',
                        help='Output file for the plot (default: fpr_plot.eps)')
    parser.add_argument('--aspect_ratio', type=float, default=1.0,
                        help='Aspect ratio for the plot (default: 1.0)')
    parser.add_argument('--title_suffix', type=str, default=None,
                        help='Text to append to the plot title (default: None)')
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        raise FileNotFoundError(f"CSV file not found: {args.csv_file}")
    
    print(f"Reading results from {args.csv_file}...")
    plot_fpr_results(args.csv_file, args.output, args.aspect_ratio, title_suffix=args.title_suffix)
    print("Done!")

