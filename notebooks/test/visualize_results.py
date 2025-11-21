import os
import sys
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import argparse
from matplotlib.lines import Line2D

def get_method_display_name(method):
    """Convert method name to display name with LaTeX formatting."""
    if method == "DELN":
        return "LN's-$t$-test"
    elif method == "Scanpy t-test":
        return "$t$-test"
    elif method == "Scanpy wilcoxon":
        return "Wilcoxon"
    else:
        return method


def plot_tpr_fpr_results(csv_file, output_file):
    """Plot TPR and FPR vs p for all methods."""
    df = pd.read_csv(csv_file)
    
    # Get unique methods and p values
    methods = df['method'].unique()
    p_values = sorted(df['p'].unique())
    
    # Use same colors as original script (tab10 colormap)
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    method_to_color = {method: colors[i] for i, method in enumerate(methods)}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot TPR
    for method in methods:
        method_df = df[df['method'] == method]
        method_df = method_df.sort_values('p')
        p_vals = method_df['p'].values
        tpr_means = method_df['tpr_mean'].values
        tpr_stds = method_df['tpr_std'].values
        
        display_name = get_method_display_name(method)
        axes[0].errorbar(p_vals, tpr_means, yerr=tpr_stds,
                        marker='o', capsize=5, capthick=2, label=display_name,
                        color=method_to_color[method], linewidth=2, markersize=6)
    
    axes[0].set_xlabel(r'Split probability $p$', fontsize=12)
    axes[0].set_ylabel('True Positive Rate (TPR)', fontsize=12)
    axes[0].set_title('TPR vs Split Probability', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=10)
    axes[0].set_ylim([0, 1])
    
    # Plot FPR
    for method in methods:
        method_df = df[df['method'] == method]
        method_df = method_df.sort_values('p')
        p_vals = method_df['p'].values
        fpr_means = method_df['fpr_mean'].values
        fpr_stds = method_df['fpr_std'].values
        
        display_name = get_method_display_name(method)
        axes[1].errorbar(p_vals, fpr_means, yerr=fpr_stds,
                        marker='o', capsize=5, capthick=2, label=display_name,
                        color=method_to_color[method], linewidth=2, markersize=6)
    
    axes[1].set_xlabel(r'Split probability $p$', fontsize=12)
    axes[1].set_ylabel('False Positive Rate (FPR)', fontsize=12)
    axes[1].set_title('FPR vs Split Probability', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=10)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_file, format='eps', dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


def plot_ap_pr_auc_results(csv_file, output_file):
    """Plot AP and PR-AUC vs p for all methods."""
    df = pd.read_csv(csv_file)
    
    # Get unique methods and p values
    methods = df['method'].unique()
    p_values = sorted(df['p'].unique())
    
    # Use same colors as original script (tab10 colormap)
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    method_to_color = {method: colors[i] for i, method in enumerate(methods)}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot AP
    for method in methods:
        method_df = df[df['method'] == method]
        method_df = method_df.sort_values('p')
        p_vals = method_df['p'].values
        ap_means = method_df['ap_mean'].values
        ap_stds = method_df['ap_std'].values
        
        display_name = get_method_display_name(method)
        axes[0].errorbar(p_vals, ap_means, yerr=ap_stds,
                         marker='o', capsize=5, capthick=2, label=display_name,
                         color=method_to_color[method], linewidth=2, markersize=6)
    axes[0].set_xlabel(r'Split probability $p$', fontsize=12)
    axes[0].set_ylabel('Average Precision (AP)', fontsize=12)
    axes[0].set_title('AP vs Split Probability', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=10)
    axes[0].set_ylim([0, 1])
    
    # Plot PR-AUC
    for method in methods:
        method_df = df[df['method'] == method]
        method_df = method_df.sort_values('p')
        p_vals = method_df['p'].values
        pr_auc_means = method_df['pr_auc_mean'].values
        pr_auc_stds = method_df['pr_auc_std'].values
        
        display_name = get_method_display_name(method)
        axes[1].errorbar(p_vals, pr_auc_means, yerr=pr_auc_stds,
                         marker='o', capsize=5, capthick=2, label=display_name,
                         color=method_to_color[method], linewidth=2, markersize=6)
    axes[1].set_xlabel(r'Split probability $p$', fontsize=12)
    axes[1].set_ylabel('PR-AUC (Trapezoidal)', fontsize=12)
    axes[1].set_title('PR-AUC vs Split Probability', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=10)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_file, format='eps', dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


def plot_pr_curves_all_p(json_file, output_file, q, lfc):
    """Plot Precision vs Recall curves for each p value (original style)."""
    with open(json_file, 'r') as f:
        curve_data = json.load(f)
    
    p_values = np.array(curve_data['p_values'])
    curves = curve_data['curves']
    
    # Get methods from first p value
    first_p_key = list(curves.keys())[0]
    methods = list(curves[first_p_key].keys())
    
    # Use same colors as original script (tab10 colormap)
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    method_to_color = {method: colors[i] for i, method in enumerate(methods)}
    
    # Create subplots: one for each p value
    n_p = len(p_values)
    n_cols = min(3, n_p)
    n_rows = (n_p + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_p == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    for p_idx, p in enumerate(p_values):
        ax = axes[p_idx]
        p_key = f"p_{p:.6f}"
        
        if p_key in curves:
            for method in methods:
                if method in curves[p_key]:
                    method_data = curves[p_key][method]
                    recall = np.array(method_data['recall'])
                    precision_mean = np.array(method_data['precision_mean'])
                    precision_std = method_data.get('precision_std')
                    pr_auc_curve_mean = method_data.get('pr_auc_curve_mean', np.nan)
                    
                    display_name = get_method_display_name(method)
                    label = f'{display_name} (AUC={pr_auc_curve_mean:.3f})' if not np.isnan(pr_auc_curve_mean) else display_name
                    
                    # Plot mean curve
                    ax.plot(recall, precision_mean,
                           label=label,
                           color=method_to_color[method], linewidth=2)
                    
                    # Plot std as shaded region (lighter shade)
                    if precision_std is not None:
                        precision_std_arr = np.array(precision_std)
                        ax.fill_between(recall,
                                      precision_mean - precision_std_arr,
                                      precision_mean + precision_std_arr,
                                      alpha=0.1, color=method_to_color[method])
        
        ax.set_xlabel('Recall', fontsize=11)
        ax.set_ylabel('Precision', fontsize=11)
        ax.set_title(f'Precision vs Recall (p={p:.3f}, q={q}, lfc={lfc})',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    # Hide unused subplots
    for idx in range(n_p, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    # Use PNG format for the subplot version
    plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


def plot_pr_curves_selected_p(json_file, output_file, q, lfc):
    """Plot Precision vs Recall curves for selected p values (smallest, median, largest) in one plot."""
    with open(json_file, 'r') as f:
        curve_data = json.load(f)
    
    p_values = np.array(curve_data['p_values'])
    curves = curve_data['curves']
    
    # Get methods from first p value
    first_p_key = list(curves.keys())[0]
    methods = list(curves[first_p_key].keys())
    
    # Use same colors as original script (tab10 colormap)
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    method_to_color = {method: colors[i] for i, method in enumerate(methods)}
    
    # Select p values: smallest, largest, and closest to median
    p_sorted = sorted(p_values)
    p_min = p_sorted[0]
    p_max = p_sorted[-1]
    p_median_val = np.median(p_sorted)
    # Find p value closest to median
    p_median_idx = np.argmin(np.abs(p_sorted - p_median_val))
    p_median = p_sorted[p_median_idx]
    
    selected_p = [p_min, p_median, p_max]
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Define line styles: dot for smallest, dash for median, solid for largest
    linestyle_map = {p_min: ':', p_median: '--', p_max: '-'}
    
    for method in methods:
        for p in selected_p:
            p_key = f"p_{p:.6f}"
            
            if p_key in curves and method in curves[p_key]:
                method_data = curves[p_key][method]
                recall = np.array(method_data['recall'])
                precision_mean = np.array(method_data['precision_mean'])
                precision_std = method_data.get('precision_std')
                
                display_name = get_method_display_name(method)
                label = f'{display_name} (p={p:.3f})'
                
                # Plot mean curve (no variance shades for selected p plot)
                # Use dot (:) for smallest, dash (--) for median, solid (-) for largest
                ax.plot(recall, precision_mean,
                       label=label,
                       color=method_to_color[method], linewidth=2, linestyle=linestyle_map[p])
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision vs Recall (q={q}, lfc={lfc})',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_file, format='eps', dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


def plot_summary_figure(csv_file, json_file, output_file, q, lfc):
    """
    Create a summary figure with:
      Top row:    [TPR vs p]  [FPR vs p]
      Bottom row: [PR-AUC vs p]  [Selected PR curves (min/median/max p)]
    Global legend (colors) indicates methods; PR-curves subplot legend shows p-pattern mapping.
    """
    # Load CSV and JSON
    df = pd.read_csv(csv_file)
    with open(json_file, 'r') as f:
        curve_data = json.load(f)
    p_values_json = np.array(curve_data['p_values'])
    curves = curve_data['curves']
    # Methods from CSV (ensures alignment with DE summaries)
    methods = df['method'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    method_to_color = {method: colors[i] for i, method in enumerate(methods)}
    # Select p values: min, median (closest), max from JSON (where PR curves exist)
    p_sorted = np.sort(p_values_json)
    p_min = p_sorted[0]
    p_max = p_sorted[-1]
    p_median_val = np.median(p_sorted)
    p_median = p_sorted[np.argmin(np.abs(p_sorted - p_median_val))]
    selected_p = [p_min, p_median, p_max]
    linestyle_map = {p_min: ':', p_median: '--', p_max: '-'}
    # Build figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_tpr, ax_fpr = axes[0, 0], axes[0, 1]
    ax_prauc, ax_prcurves = axes[1, 0], axes[1, 1]
    # TPR vs p
    for method in methods:
        mdf = df[df['method'] == method].sort_values('p')
        ax_tpr.errorbar(mdf['p'].values, mdf['tpr_mean'].values, yerr=mdf['tpr_std'].values,
                        marker='o', capsize=5, capthick=2,
                        color=method_to_color[method], linewidth=2, markersize=6,
                        label=get_method_display_name(method))
    ax_tpr.set_xlabel(r'Split probability $p$', fontsize=22)
    ax_tpr.set_ylabel('True Positive Rate (TPR)', fontsize=22)
    ax_tpr.grid(True, alpha=0.3)
    ax_tpr.set_ylim([0, 1])
    ax_tpr.tick_params(axis='both', which='major', labelsize=20)
    # FPR vs p
    for method in methods:
        mdf = df[df['method'] == method].sort_values('p')
        ax_fpr.errorbar(mdf['p'].values, mdf['fpr_mean'].values, yerr=mdf['fpr_std'].values,
                        marker='o', capsize=5, capthick=2,
                        color=method_to_color[method], linewidth=2, markersize=6)
    ax_fpr.set_xlabel(r'Split probability $p$', fontsize=22)
    ax_fpr.set_ylabel('False Positive Rate (FPR)', fontsize=22)
    ax_fpr.grid(True, alpha=0.3)
    ax_fpr.set_ylim([0, 1])
    ax_fpr.tick_params(axis='both', which='major', labelsize=20)
    # PR-AUC vs p (recomputed from JSON PR curves)
    # Use JSON p grid to ensure availability of curves
    p_values_for_auc = np.sort(p_values_json)
    for method in methods:
        pr_auc_list = []
        pr_auc_err = []
        p_valid = []
        for p in p_values_for_auc:
            p_key = f"p_{p:.6f}"
            if p_key in curves and method in curves[p_key]:
                md = curves[p_key][method]
                recall = np.array(md.get('recall', []))
                precision_mean = np.array(md.get('precision_mean', []))
                if recall.size > 1 and precision_mean.size == recall.size:
                    # Recompute AUC from mean curve
                    auc_val = np.trapz(precision_mean, recall)
                    pr_auc_list.append(auc_val)
                    # If std available, use it as yerr; else no error bar
                    pr_auc_std = md.get('pr_auc_curve_std', None)
                    if pr_auc_std is None:
                        pr_auc_err.append(np.nan)
                    else:
                        pr_auc_err.append(pr_auc_std)
                    p_valid.append(p)
        if len(p_valid) > 0:
            p_valid = np.array(p_valid)
            pr_auc_arr = np.array(pr_auc_list, dtype=float)
            pr_auc_err_arr = np.array(pr_auc_err, dtype=float)
            # Plot with error bars where available
            # If all yerr are NaN, omit yerr to avoid warnings
            if np.all(np.isnan(pr_auc_err_arr)):
                ax_prauc.errorbar(p_valid, pr_auc_arr,
                                  marker='o', capsize=5, capthick=2,
                                  color=method_to_color[method], linewidth=2, markersize=6)
            else:
                ax_prauc.errorbar(p_valid, pr_auc_arr, yerr=pr_auc_err_arr,
                                  marker='o', capsize=5, capthick=2,
                                  color=method_to_color[method], linewidth=2, markersize=6)
    ax_prauc.set_xlabel(r'Split probability $p$', fontsize=22)
    ax_prauc.set_ylabel('PR-AUC', fontsize=22)
    ax_prauc.grid(True, alpha=0.3)
    ax_prauc.set_ylim([0, 1])
    ax_prauc.tick_params(axis='both', which='major', labelsize=20)
    # Selected PR curves (min, median, max p) with p-pattern legend
    # Methods legend will be global; avoid method labels here
    for method in methods:
        color = method_to_color[method]
        for p in selected_p:
            p_key = f"p_{p:.6f}"
            if p_key in curves and method in curves[p_key]:
                md = curves[p_key][method]
                recall = np.array(md['recall'])
                precision_mean = np.array(md['precision_mean'])
                ax_prcurves.plot(recall, precision_mean,
                                 color=color, linewidth=2, linestyle=linestyle_map[p])
    ax_prcurves.set_xlabel('Recall', fontsize=22)
    ax_prcurves.set_ylabel('Precision', fontsize=22)
    ax_prcurves.grid(True, alpha=0.3)
    ax_prcurves.set_xlim([0, 1])
    ax_prcurves.set_ylim([0, 1])
    ax_prcurves.tick_params(axis='both', which='major', labelsize=20)
    # Global legend for methods (color -> method)
    method_handles = [Line2D([0], [0], color=method_to_color[m], lw=3)
                      for m in methods]
    method_labels = [get_method_display_name(m) for m in methods]
    fig.legend(method_handles, method_labels, loc='upper center',
               bbox_to_anchor=(0.5, 1.02), ncol=min(len(methods), 3),
               frameon=False, fontsize=20)
    # Subplot legend for p-patterns (linestyle -> p)
    pattern_handles = [Line2D([0], [0], color='black', lw=2, linestyle=linestyle_map[p])
                       for p in selected_p]
    pattern_labels = [f"p={p:.3f}" for p in selected_p]
    ax_prcurves.legend(pattern_handles, pattern_labels, loc='lower left', fontsize=20, frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for global legend
    plt.savefig(output_file, format='eps', dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize results from vishd_test_shape_split_shared.py')
    parser.add_argument('--csv_file', type=str, required=True,
                        help='Input CSV file with results')
    parser.add_argument('--json_file', type=str, required=True,
                        help='Input JSON file with PR curve data')
    parser.add_argument('--output_prefix', type=str, default='results',
                        help='Output file prefix for plots (default: results)')
    args = parser.parse_args()
    
    csv_file = args.csv_file
    json_file = args.json_file
    output_prefix = args.output_prefix
    
    # Read q and lfc from CSV (should be same for all rows)
    df = pd.read_csv(csv_file)
    q = df['q'].iloc[0]
    lfc = df['lfc'].iloc[0]
    
    print(f"Visualizing results with q={q}, lfc={lfc}")
    
    # Plot 1: TPR and FPR
    print("Plotting TPR and FPR...")
    plot_tpr_fpr_results(csv_file, f"{output_prefix}_tpr_fpr.eps")
    
    # Plot 2: AP and PR-AUC
    print("Plotting AP and PR-AUC...")
    plot_ap_pr_auc_results(csv_file, f"{output_prefix}_ap_pr_auc.eps")
    
    # Plot 3: PR curves for all p values
    print("Plotting PR curves for all p values...")
    plot_pr_curves_all_p(json_file, f"{output_prefix}_pr_curves_all_p.eps", q, lfc)
    
    # Plot 4: PR curves for selected p values (min, median, max)
    print("Plotting PR curves for selected p values...")
    plot_pr_curves_selected_p(json_file, f"{output_prefix}_pr_curves_selected_p.eps", q, lfc)
    
    # Plot 5: Summary figure (TPR/FPR/PR-AUC + selected PR curves)
    print("Plotting summary figure...")
    plot_summary_figure(csv_file, json_file, f"{output_prefix}_summary.eps", q, lfc)
    
    print("Done!")

