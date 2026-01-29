import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import argparse

# Ensure project root (DE-ZILN) is on sys.path so local modules resolve
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from de_test_with_scores import (
    de_test_with_scores, 
    run_gsea_for_all_methods, 
    save_gsea_results,
    format_gsea_results
)


def aggregate_by_cell_id_and_cluster(adata):
    """
    Aggregate reads for spots that share the same cell_id and Cluster.
    
    Optimized version using groupby operations for faster aggregation.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object with adata.obs.cell_id and adata.obs.Cluster
        
    Returns:
    --------
    adata_agg : AnnData
        Aggregated data with one row per unique (cell_id, Cluster) combination
    """
    if 'cell_id' not in adata.obs.columns:
        raise ValueError("adata.obs must contain 'cell_id' column")
    if 'Cluster' not in adata.obs.columns:
        raise ValueError("adata.obs must contain 'Cluster' column")
    
    # Create a unique identifier for each (cell_id, Cluster) combination
    adata.obs['cell_cluster_id'] = (
        adata.obs['cell_id'].astype(str) + '_' + 
        adata.obs['Cluster'].astype(str)
    )
    
    # Get unique combinations and their indices
    unique_combinations = adata.obs['cell_cluster_id'].unique()
    n_combinations = len(unique_combinations)
    n_genes = adata.shape[1]
    
    print(f"  Found {n_combinations} unique (cell_id, Cluster) combinations")
    
    # Use groupby to get indices for each combination (much faster)
    grouped = adata.obs.groupby('cell_cluster_id', sort=False)
    
    # Collect aggregated rows (more efficient than pre-allocating sparse matrix)
    aggregated_rows = []
    aggregated_obs_list = []
    
    for idx, (combo, group_indices) in enumerate(grouped):
        indices = group_indices.index.values
        idx_array = adata.obs.index.get_indexer(indices)
        
        # Sum counts across spots with same cell_id and Cluster
        if sp.issparse(adata.X):
            combo_counts = adata.X[idx_array, :].sum(axis=0)
            # Convert to 1D array (handles both sparse and dense)
            if hasattr(combo_counts, 'A1'):
                combo_counts = combo_counts.A1
            elif hasattr(combo_counts, 'toarray'):
                combo_counts = np.asarray(combo_counts).flatten()
            else:
                combo_counts = np.asarray(combo_counts).flatten()
        else:
            combo_counts = adata.X[idx_array, :].sum(axis=0)
        
        aggregated_rows.append(combo_counts)
        
        # Store metadata (take first row's metadata, but update cell_cluster_id)
        obs_row = group_indices.iloc[0].copy()
        obs_row['cell_cluster_id'] = combo
        obs_row['n_spots_aggregated'] = len(group_indices)
        aggregated_obs_list.append(obs_row)
    
    # Stack all rows into final array
    if sp.issparse(adata.X):
        # If input was sparse, try to keep output sparse if possible
        # But for aggregation (sums), dense is often more efficient
        aggregated_counts = np.vstack(aggregated_rows)
    else:
        aggregated_counts = np.vstack(aggregated_rows)
    
    # Create aggregated obs DataFrame
    aggregated_obs = pd.DataFrame(aggregated_obs_list)
    
    # Create new AnnData object
    adata_agg = sc.AnnData(aggregated_counts)
    adata_agg.obs = aggregated_obs
    adata_agg.var = adata.var.copy()
    
    return adata_agg


def run_de_between_clusters(
    adata,
    cluster_col='Cluster',
    gene_sets=None,
    n_permutations=1000,
    gsea_ranks_only=False,
    min_expr_fraction=0.0
):
    """
    Run DE analysis between two clusters and perform GSEA.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object (should be aggregated by cell_id and Cluster)
    cluster_col : str
        Name of column in adata.obs containing cluster labels
    gene_sets : dict, optional
        Dictionary where keys are gene set names and values are sets/lists of genes.
        If None, GSEA will be skipped.
    n_permutations : int
        Number of permutations for GSEA (default: 1000)
        
    Returns:
    --------
    de_results_merged : pandas.DataFrame
        Merged DE results table for all three methods
    gsea_results : dict, optional
        GSEA results if gene_sets provided
    """
    # Check number of clusters
    unique_clusters = adata.obs[cluster_col].unique()
    if len(unique_clusters) != 2:
        raise ValueError(
            f"Expected exactly 2 clusters, found {len(unique_clusters)}: {unique_clusters}"
        )
    
    cluster1, cluster2 = sorted(unique_clusters)  # Sort for consistent ordering
    print(f"Comparing clusters: {cluster1} vs {cluster2}")
    
    # Convert to dense before splitting (more efficient for DE tests)
    if sp.issparse(adata.X):
        print("  Converting sparse matrix to dense for faster DE tests...")
        adata.X = adata.X.toarray()
        print("  Conversion complete.")
    
    # Filter genes expressed in at least min_expr_fraction of cells (across all cells)
    if min_expr_fraction and min_expr_fraction > 0:
        expressed_fraction = (adata.X > 0).mean(axis=0)
        keep_genes_mask = np.asarray(expressed_fraction).flatten() > float(min_expr_fraction)
        n_removed = np.sum(~keep_genes_mask)
        if n_removed > 0:
            print(
                f"  Filtering genes expressed in <= {min_expr_fraction:.4g} of cells: "
                f"removing {n_removed} genes"
            )
            adata = adata[:, keep_genes_mask].copy()
        else:
            print(f"  No genes removed by min_expr_fraction={min_expr_fraction:.4g}")

    # Split data by cluster
    mask_cluster1 = adata.obs[cluster_col] == cluster1
    mask_cluster2 = adata.obs[cluster_col] == cluster2
    
    X = adata[mask_cluster1].X  # Cluster 1 (control)
    Y = adata[mask_cluster2].X  # Cluster 2 (treatment)
    
    # Ensure arrays are numpy arrays (should already be dense after conversion above)
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    print(f"Cluster {cluster1}: {X.shape[0]} cells, {X.shape[1]} genes")
    print(f"Cluster {cluster2}: {Y.shape[0]} cells, {Y.shape[1]} genes")
    
    # Get gene names from adata.var before running DE
    gene_names_original = adata.var.index.values
    # Also get gene_ids if available
    gene_ids_original = adata.var['gene_ids'].values if 'gene_ids' in adata.var.columns else None
    
    # Run DE tests with gene names
    print("Running DE tests...")
    all_de_results = de_test_with_scores(X, Y, gene_names=gene_names_original)
    
    # Create merged DE results table
    de_results_list = []
    for method_name, method_results in all_de_results.items():
        n_genes_filtered = len(method_results['lfc'])
        
        # Get gene names after filtering
        if 'gene_names' in method_results:
            gene_names_filtered = method_results['gene_names']
        else:
            # Fallback to indices if gene names not available
            gene_names_filtered = np.arange(n_genes_filtered)
        
        # Get gene_ids if available (need to filter using the same mask)
        if gene_ids_original is not None and 'gene_mask' in method_results:
            gene_mask = method_results['gene_mask']
            gene_ids_filtered = gene_ids_original[gene_mask]
        else:
            gene_ids_filtered = None
        
        # Create DataFrame for this method
        df = pd.DataFrame({
            'gene_name': gene_names_filtered,
            'method': method_name,
            'lfc': method_results['lfc'],
            'test_statistic': method_results['test_statistic'],
            'p_val': method_results['p_vals'],
            'adj_p_val': method_results['adj_pvals']
        })
        
        # Add gene_ids if available
        if gene_ids_filtered is not None:
            df['gene_id'] = gene_ids_filtered
        
        # Add log_abs_statistic for LN if available
        if 'log_abs_statistic' in method_results:
            df['log_abs_statistic'] = method_results['log_abs_statistic']
        
        de_results_list.append(df)
    
    de_results_merged = pd.concat(de_results_list, ignore_index=True)
    
    # Return DE results and all_de_results (for potential GSEA later)
    # GSEA will be run separately in main() after writing DE results
    return de_results_merged, all_de_results, cluster1, cluster2


def load_gmt_gene_sets(gmt_path):
    """
    Load GMT file into dict of set_name -> set(genes).
    """
    gene_sets = {}
    with open(gmt_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            set_name = parts[0]
            genes = [g for g in parts[2:] if g]
            if genes:
                gene_sets[set_name] = set(genes)
    return gene_sets


def main():
    parser = argparse.ArgumentParser(
        description='Run DE analysis between two clusters and perform GSEA'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input h5ad file path')
    parser.add_argument('--output_de', type=str, required=True,
                        help='Output CSV file for merged DE results')
    parser.add_argument('--output_gsea', type=str, default=None,
                        help='Output CSV file for GSEA results (optional)')
    parser.add_argument('--gene_sets_file', type=str, default=None,
                        help='JSON file containing gene sets (optional, format: {"set_name": ["gene1", "gene2", ...]}). '
                             'If not provided, GSEA will be skipped. For standard gene sets, consider using gseapy or '
                             'downloading MSigDB gene sets.')
    parser.add_argument('--use_gseapy', action='store_true',
                        help='Use gseapy library to load standard gene sets (requires gseapy package)')
    parser.add_argument('--gene_set_name', type=str, default='KEGG_2019_Human',
                        help='Gene set name to use with gseapy (default: KEGG_2019_Human). '
                             'Common options: KEGG_2019_Human, GO_Biological_Process_2021, etc.')
    parser.add_argument('--use-gmt-file', action='store_true',
                        help='Use a local GMT file for gene sets. '
                             'Overrides --use_gseapy and --gene_sets_file.')
    parser.add_argument('--gmt-file', type=str, default=None,
                        help='Path to GMT file containing gene sets. '
                             'Used only when --use-gmt-file is set.')
    parser.add_argument('--n_permutations', type=int, default=1000,
                        help='Number of permutations for GSEA (default: 1000)')
    parser.add_argument('--gsea-ranks-only', action='store_true',
                        help='Use ranks only for GSEA (ignore score magnitudes). '
                             'If False, uses test_statistic scores (default: False)')
    parser.add_argument('--output_aggregated', type=str, default=None,
                        help='Output h5ad file path for aggregated data (optional). '
                             'If not provided, defaults to input filename with "_aggregated" suffix')
    parser.add_argument('--min-expr-fraction', type=float, default=0.0,
                        help='Minimum fraction of cells expressing a gene to keep it (default: 0, no filter).')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    adata = sc.read_h5ad(args.input)
    
    # Aggregate by cell_id and Cluster
    print("Aggregating reads by cell_id and Cluster...")
    adata_agg = aggregate_by_cell_id_and_cluster(adata)
    print(f"Aggregated data: {adata_agg.shape[0]} unique (cell_id, Cluster) combinations, {adata_agg.shape[1]} genes")
    
    # Save aggregated data if output path specified or use default
    if args.output_aggregated is None:
        # Generate default filename: input_filename_aggregated.h5ad
        import os
        input_basename = os.path.splitext(os.path.basename(args.input))[0]
        input_dir = os.path.dirname(args.input) if os.path.dirname(args.input) else '.'
        args.output_aggregated = os.path.join(input_dir, f"{input_basename}_aggregated.h5ad")
    
    print(f"\nSaving aggregated data to {args.output_aggregated}...")
    adata_agg.write(args.output_aggregated)
    print(f"Aggregated data saved. Shape: {adata_agg.shape}")
    
    # Load gene sets
    gene_sets = None
    if args.use_gmt_file or args.gmt_file:
        gmt_path = args.gmt_file
        if not gmt_path:
            raise ValueError("Missing --gmt-file. Provide a path to a GMT file.")
        if not os.path.exists(gmt_path):
            raise FileNotFoundError(f"GMT file not found: {gmt_path}.")
        print(f"Loading gene sets from GMT: {gmt_path}...")
        gene_sets = load_gmt_gene_sets(gmt_path)
        print(f"Loaded {len(gene_sets)} gene sets from GMT")
        # Ensure other gene set options are ignored
        args.use_gseapy = False
        args.gene_sets_file = None

    if args.use_gseapy:
        try:
            import gseapy as gp
            print(f"Loading gene sets from gseapy: {args.gene_set_name}...")
            print("Note: First run may download gene sets from MSigDB (this can take a few minutes)")
            
            # Get gene sets from gseapy
            # get_library_name() returns a list of available library names, not the actual gene sets
            # We need to use get_library() to get the actual gene sets
            try:
                # Try to get the gene sets dictionary using get_library()
                # This downloads and loads the gene sets
                gene_sets_dict = gp.get_library(name=args.gene_set_name, organism='Human')
                
                # Check if it's a dict (what we want) or list (library names)
                if isinstance(gene_sets_dict, list):
                    # If it's a list, it's probably the list of available library names
                    print(f"Warning: '{args.gene_set_name}' appears to be a library name list, not gene sets.")
                    print(f"Available gene set libraries: {gene_sets_dict[:10]}..." if len(gene_sets_dict) > 10 else f"Available: {gene_sets_dict}")
                    print("Please use a specific gene set library name like 'KEGG_2019_Human'")
                    gene_sets_dict = None
                elif not isinstance(gene_sets_dict, dict):
                    print(f"Warning: Unexpected format returned by gseapy: {type(gene_sets_dict)}")
                    gene_sets_dict = None
                
                if gene_sets_dict is None or len(gene_sets_dict) == 0:
                    print(f"Warning: Gene set '{args.gene_set_name}' not found or empty.")
                    print("  Use gseapy.get_library_name() to see available options")
                    print("  Common options: 'KEGG_2019_Human', 'GO_Biological_Process_2021', 'Reactome_2016', etc.")
                else:
                    # Convert to our format (dict of set_name -> set of genes)
                    gene_sets = {}
                    for set_name, genes in gene_sets_dict.items():
                        # genes might be a list, set, or string
                        if isinstance(genes, str):
                            genes = genes.split('\t') if '\t' in genes else [genes]
                        elif not isinstance(genes, (list, set)):
                            genes = [genes]
                        gene_sets[set_name] = set(genes)
                    print(f"Loaded {len(gene_sets)} gene sets from {args.gene_set_name}")
            except Exception as e:
                print(f"Error loading gene sets from gseapy: {e}")
                print("Falling back to gene_sets_file if provided...")
                args.use_gseapy = False
        except ImportError:
            print("Warning: gseapy not installed. Install with: pip install gseapy")
            print("Falling back to gene_sets_file if provided...")
            args.use_gseapy = False
    
    if not args.use_gseapy and args.gene_sets_file:
        import json
        print(f"Loading gene sets from {args.gene_sets_file}...")
        with open(args.gene_sets_file, 'r') as f:
            gene_sets = json.load(f)
        # Convert lists to sets for faster lookup
        gene_sets = {k: set(v) if isinstance(v, list) else v for k, v in gene_sets.items()}
        print(f"Loaded {len(gene_sets)} gene sets from file")
    
    if gene_sets is None:
        print("No gene sets provided. GSEA will be skipped.")
        print("To use gene sets:")
        print("  1. Use --use_gseapy to load standard gene sets (requires: pip install gseapy)")
        print("  2. Provide --gene_sets_file with a JSON file containing your gene sets")
    
    # Run DE analysis first (without GSEA)
    print("Running DE analysis...")
    de_results, all_de_results, cluster1, cluster2 = run_de_between_clusters(
        adata_agg, 
        gene_sets=None,  # Skip GSEA for now - will run separately
        n_permutations=args.n_permutations,
        gsea_ranks_only=args.gsea_ranks_only,
        min_expr_fraction=args.min_expr_fraction
    )
    
    # Add cluster information to DE results
    de_results['cluster1'] = cluster1
    de_results['cluster2'] = cluster2
    de_results['comparison'] = f"{cluster1}_vs_{cluster2}"
    
    # Save DE results BEFORE running GSEA
    print(f"\nSaving DE results to {args.output_de}...")
    de_results.to_csv(args.output_de, index=False)
    print(f"DE results saved. Shape: {de_results.shape}")
    
    # Run GSEA if gene sets provided
    gsea_results = None
    use_gseapy_prerank = False  # Initialize variable
    if gene_sets is not None:
        print("\nRunning GSEA...")
        from de_test_with_scores import run_gsea_for_all_methods
        
        # Get gene names after filtering (from DE results)
        if 'gene_names' in all_de_results['LN_test']:
            gene_names_for_gsea = all_de_results['LN_test']['gene_names']
        else:
            n_genes_filtered = len(all_de_results['LN_test']['lfc'])
            gene_names_for_gsea = np.arange(n_genes_filtered)
        
        # Map gene sets to gene names (not indices)
        # GSEA will use gene names for matching
        gene_sets_mapped = {}
        if len(gene_sets) > 0:
            # Create a set of all available gene names for faster lookup
            available_genes = set(gene_names_for_gsea)
            
            for set_name, gene_set in gene_sets.items():
                # Convert gene set to set if it's a list
                if isinstance(gene_set, list):
                    gene_set = set(gene_set)
                elif not isinstance(gene_set, set):
                    gene_set = {gene_set}
                
                # Find intersection: genes in the set that are also in our filtered genes
                gene_set_filtered = gene_set.intersection(available_genes)
                
                if len(gene_set_filtered) > 0:
                    gene_sets_mapped[set_name] = gene_set_filtered
                else:
                    print(f"Warning: Gene set '{set_name}' has no genes matching the filtered gene list")
        
        if len(gene_sets_mapped) > 0:
            print(f"\n{'='*60}")
            print("GSEA Configuration:")
            print(f"{'='*60}")
            print(f"  Number of gene sets: {len(gene_sets_mapped)}")
            print(f"  Number of permutations: {args.n_permutations}")
            print(f"  Ranking mode: {'RANKS ONLY' if args.gsea_ranks_only else 'SCORES (test_statistic values)'}")
            
            # Use gseapy prerank if available, otherwise fall back to custom implementation
            use_gseapy_prerank = True
            try:
                import gseapy as gp
                print(f"  Implementation: gseapy.prerank")
            except ImportError:
                use_gseapy_prerank = False
                print(f"  Implementation: Custom GSEA (gseapy not available)")
            
            print(f"  Method: {'Prerank' if use_gseapy_prerank else 'Simple'}")
            print(f"{'='*60}\n")
            
            print(f"Running GSEA on {len(gene_sets_mapped)} gene sets...")
            gsea_results = run_gsea_for_all_methods(
                all_de_results, 
                gene_names_for_gsea, 
                gene_sets_mapped,
                n_permutations=args.n_permutations,
                use_gseapy_prerank=use_gseapy_prerank,
                gsea_ranks_only=args.gsea_ranks_only
            )
        else:
            print("Warning: No valid gene sets found after filtering. Skipping GSEA.")
    
    # Save GSEA results if available
    if gsea_results is not None and args.output_gsea:
        print(f"\nSaving GSEA results to {args.output_gsea}...")
        gsea_df = format_gsea_results(gsea_results)
        gsea_df['cluster1'] = cluster1
        gsea_df['cluster2'] = cluster2
        gsea_df['comparison'] = f"{cluster1}_vs_{cluster2}"
        # Add metadata columns about GSEA configuration
        gsea_df['gsea_mode'] = 'ranks_only' if args.gsea_ranks_only else 'scores'
        gsea_df['gsea_implementation'] = 'gseapy_prerank' if use_gseapy_prerank else 'custom'
        gsea_df['n_permutations'] = args.n_permutations
        gsea_df.to_csv(args.output_gsea, index=False)
        print(f"GSEA results saved. Shape: {gsea_df.shape}")
        print(f"  GSEA mode: {'ranks_only' if args.gsea_ranks_only else 'scores'}")
        print(f"  Implementation: {'gseapy.prerank' if use_gseapy_prerank else 'custom'}")
    
    print("Done!")


if __name__ == '__main__':
    main()

