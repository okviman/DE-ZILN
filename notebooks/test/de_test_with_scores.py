import os
import sys
import numpy as np
import statsmodels.stats.multitest as smm
import scanpy as sc
import pandas as pd

# Ensure project root (DE-ZILN) is on sys.path so local modules resolve
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import get_LN_lfcs as get_DELN_lfcs


def scanpy_sig_test_with_scores(X, Y, method='t-test', normalization='CP10K', corr_method="bonferroni"):
    """
    Run scanpy DE test and return log fold changes, test statistics (scores), p-values, and adjusted p-values.
    
    Parameters:
    -----------
    X : array (n_cells_x, n_genes)
        Control group counts
    Y : array (n_cells_y, n_genes)
        Treatment group counts
    method : str
        't-test' or 'wilcoxon'
    normalization : str
        'CP10K' or 'median-of-ratios'
    corr_method : str
        Multiple testing correction method
        
    Returns:
    --------
    lfcs : array (n_genes,)
        Log fold changes
    test_scores : array (n_genes,)
        Test statistics (scores) - signed, can be used for GSEA ranking
    p_vals : array (n_genes,)
        Unadjusted p-values
    adj_pvals : array (n_genes,)
        Adjusted p-values
    """
    nx = X.shape[0]
    ny = Y.shape[0]
    n_genes = Y.shape[1]
    # Create Scanpy AnnData Object
    X_group = np.repeat("X", nx)  # Labels for group X
    Y_group = np.repeat("Y", ny)  # Labels for group Y

    if normalization == 'median-of-ratios':
        X = X.astype(float)
        Y = Y.astype(float)
        X_ = X.copy()
        Y_ = Y.copy()
        X_[X_ <= 0] = np.nan
        Y_[Y_ == 0] = np.nan

        denom_Y = np.exp(np.nanmean(np.log(Y_), 0))
        c_Y = np.nanmedian(Y_ / denom_Y, 1, keepdims=True)
        Y /= c_Y

        denom_X = np.exp(np.nanmean(np.log(X_), 0))
        c_X = np.nanmedian(X_ / denom_X, 1, keepdims=True)
        X /= c_X

        Y[Y == np.nan] = 0.
        X[X == np.nan] = 0.

    adata = sc.AnnData(np.vstack([X, Y]))  # Combine X and Y
    adata.var_names = [f"Gene{i}" for i in range(n_genes)]  # Gene names
    adata.obs["group"] = np.concatenate([X_group, Y_group])  # Assign group labels

    adata.layers["counts"] = adata.X.copy()
    if normalization == 'CP10K':
        sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Run Differential Expression Analysis using Scanpy
    sc.tl.rank_genes_groups(adata, groupby="group", method=method, reference="X", corr_method=corr_method)

    # Extract DE Results including test scores
    de_results = pd.DataFrame({
        "gene": adata.uns['rank_genes_groups']['names']['Y'],
        "log2_fc": adata.uns["rank_genes_groups"]["logfoldchanges"]["Y"],
        "p_value": adata.uns["rank_genes_groups"]["pvals"]["Y"],
        "p_adj": adata.uns["rank_genes_groups"]["pvals_adj"]["Y"]
    })
    
    # Extract test scores if available (scores field contains test statistics)
    if 'scores' in adata.uns['rank_genes_groups']:
        de_results["scores"] = adata.uns["rank_genes_groups"]["scores"]["Y"]
    else:
        # If scores not available, use signed log fold change as proxy
        # (multiply by -log10(p) to incorporate significance)
        de_results["scores"] = de_results["log2_fc"] * (-np.log10(de_results["p_value"] + 1e-300))
    
    de_results["gene"] = [int(gene.replace("Gene", "")) for gene in de_results["gene"]]
    gene_idx_sorted = np.argsort(de_results["gene"])

    lfcs = de_results["log2_fc"][gene_idx_sorted].values
    test_scores = de_results["scores"][gene_idx_sorted].values
    p_vals = de_results["p_value"][gene_idx_sorted].values
    adj_pvals = de_results["p_adj"][gene_idx_sorted].values
    
    return lfcs, test_scores, p_vals, adj_pvals


def de_test_with_scores(X, Y, gene_names=None):
    """
    Run DE test on X and Y, return log fold changes, test statistics, and p-values for all methods.
    
    This function is designed for GSEA analysis, which requires test statistics (scores) 
    that are signed and can be used for ranking genes.
    
    Parameters:
    -----------
    X : array (n_cells_x, n_genes)
        Control group counts
    Y : array (n_cells_y, n_genes)
        Treatment group counts
    gene_names : array-like (n_genes,), optional
        Gene names/identifiers corresponding to the genes in X and Y.
        If provided, will be filtered and returned in results.
        
    Returns:
    --------
    results : dict
        Dictionary with keys for each method ("LN_test", "Scanpy t-test", "Scanpy wilcoxon").
        Each method's value is a dict with:
        - 'lfc': array of log fold changes
        - 'test_statistic': array of test statistics (signed, for GSEA ranking)
        - 'p_vals': array of unadjusted p-values
        - 'adj_pvals': array of adjusted p-values (Bonferroni for DELN, method-specific for scanpy)
        - 'gene_names': array of gene names after filtering (if gene_names provided)
        - 'gene_mask': boolean mask indicating which original genes were kept
    """
    n_genes = X.shape[-1]

    # Remove genes with zero counts in both groups
    idx_X = set(np.arange(n_genes)[X.sum(0) == 0])
    idx_Y = set(np.arange(n_genes)[Y.sum(0) == 0])
    union_unexpressed_gene_set = idx_X.union(idx_Y)
    cols_to_remove = np.array(list(union_unexpressed_gene_set), dtype=int)
    mask = np.ones(n_genes, dtype=bool)
    if cols_to_remove.size > 0:
        mask[cols_to_remove] = False
    X = X[:, mask].copy()
    Y = Y[:, mask].copy()
    
    # Filter gene names if provided
    gene_names_filtered = None
    if gene_names is not None:
        gene_names_filtered = np.array(gene_names)[mask]

    n_genes_filtered = X.shape[-1]
    results = {}

    for method in ["DELN", "t-test", 'wilcoxon']:
        if method == "DELN":
            method_key = "LN_test"  # Rename DELN to LN_test in output
        else:
            method_key = "Scanpy " + method
            
        if method == "DELN":
            # Get test statistics for DELN (returns lfc, statistic, log_abs_statistic, p_vals)
            lfcs, test_statistics, log_abs_statistic, p_vals = get_DELN_lfcs(
                Y, X, test='t', return_log_abs_statistic=True
            )
            # Apply Bonferroni correction
            adj_pvals = smm.multipletests(p_vals, alpha=0.05, method='bonferroni')[1]
            
            # Store log_abs_statistic for GSEA ranking
            # For now, we'll use test_statistics as before, but store log_abs_statistic separately
            # The GSEA ranking function will use log_abs_statistic for LN
            
        else:
            # Get test statistics for scanpy methods
            lfcs, test_scores, p_vals, adj_pvals = scanpy_sig_test_with_scores(X, Y, method=method)
            
            # For scanpy, test_scores are already signed and can be used directly
            test_statistics = test_scores
        
        # Convert to numpy arrays and ensure correct shape
        lfcs = np.asarray(lfcs).flatten()
        test_statistics = np.asarray(test_statistics).flatten()
        p_vals = np.asarray(p_vals).flatten()
        adj_pvals = np.asarray(adj_pvals).flatten()
        
        # Ensure arrays have the correct length
        if len(lfcs) != n_genes_filtered or len(test_statistics) != n_genes_filtered:
            raise ValueError(f"Array length mismatch: lfcs={len(lfcs)}, test_statistics={len(test_statistics)}, expected={n_genes_filtered}")
        
        results[method_key] = {
            "lfc": lfcs,
            "test_statistic": test_statistics,
            "p_vals": p_vals,
            "adj_pvals": adj_pvals
        }
        
        # Store log_abs_statistic for LN method (for GSEA ranking)
        if method == "DELN":
            results[method_key]["log_abs_statistic"] = log_abs_statistic
    
    # Add gene names and mask to all methods (same for all)
    if gene_names_filtered is not None:
        for method_key in results.keys():
            results[method_key]["gene_names"] = gene_names_filtered
            results[method_key]["gene_mask"] = mask
    
    return results


def compute_gsea_ranking_scores(lfcs, test_statistics, p_vals, method, log_abs_statistic=None):
    """
    Compute GSEA ranking scores for genes.
    
    For GSEA, genes are ranked by test_statistic directly for all methods.
    This provides a consistent ranking approach across all three methods.
    
    Ranking approach:
    - All methods: use test_statistic directly to sort
    
    Parameters:
    -----------
    lfcs : array (n_genes,)
        Log fold changes (not used for ranking, but kept for compatibility)
    test_statistics : array (n_genes,)
        Test statistics (used for ranking)
    p_vals : array (n_genes,)
        P-values (not used for ranking, but kept for compatibility)
    method : str
        Method name ("LN_test", "Scanpy t-test", "Scanpy wilcoxon")
    log_abs_statistic : array (n_genes,), optional
        Log of absolute statistic for LN method (not used, but kept for compatibility)
        
    Returns:
    --------
    gsea_scores : array (n_genes,)
        GSEA ranking scores (test_statistic values, sorted by these)
    """
    # Use test_statistic directly for all methods
    gsea_scores = test_statistics
    
    return gsea_scores


def run_gsea_simple(ranked_genes, gene_sets, n_permutations=1000):
    """
    Run simple GSEA (Gene Set Enrichment Analysis) on ranked genes.
    
    This is a simplified implementation of GSEA that computes enrichment scores
    for gene sets based on a ranked gene list.
    
    Parameters:
    -----------
    ranked_genes : array-like (n_genes,)
        Gene names or indices, ranked by their GSEA score (highest to lowest)
    gene_sets : dict
        Dictionary where keys are gene set names and values are sets/lists of genes
        that belong to that gene set
    n_permutations : int
        Number of permutations for p-value calculation (default: 1000)
        
    Returns:
    --------
    results : dict
        Dictionary with gene set names as keys and dicts containing:
        - 'es': Enrichment score
        - 'nes': Normalized enrichment score
        - 'p_value': P-value from permutation test
        - 'fdr': False discovery rate (if multiple gene sets tested)
    """
    n_genes = len(ranked_genes)
    ranked_genes_list = list(ranked_genes)
    
    results = {}
    
    # Compute enrichment scores for each gene set
    for gene_set_name, gene_set in gene_sets.items():
        # Convert gene set to set for faster lookup
        gene_set_set = set(gene_set) if not isinstance(gene_set, set) else gene_set
        
        # Find positions of genes in the ranked list
        hits = [i for i, gene in enumerate(ranked_genes_list) if gene in gene_set_set]
        
        if len(hits) == 0:
            # No genes from this set in the ranked list
            results[gene_set_name] = {
                'es': 0.0,
                'nes': 0.0,
                'p_value': 1.0,
                'fdr': 1.0
            }
            continue
        
        # Compute enrichment score (ES)
        # ES is the maximum deviation from zero of a running sum statistic
        # Positive ES: genes in set are enriched at top of ranked list
        # Negative ES: genes in set are enriched at bottom of ranked list
        
        # Create running sum
        running_sum = np.zeros(n_genes + 1)
        hit_indices = np.array(hits)
        
        # Compute hit and miss contributions
        # Hit: +1/sum(hit_weights) when we hit a gene in the set
        # Miss: -1/(N - |S|) when we miss a gene in the set
        n_hits = len(hits)
        n_misses = n_genes - n_hits
        
        if n_misses == 0:
            # All genes are in the set
            es = 1.0
        else:
            hit_weight = 1.0 / n_hits if n_hits > 0 else 0.0
            miss_weight = 1.0 / n_misses if n_misses > 0 else 0.0
            
            # Build running sum
            for i in range(1, n_genes + 1):
                if i - 1 in hit_indices:
                    running_sum[i] = running_sum[i-1] + hit_weight
                else:
                    running_sum[i] = running_sum[i-1] - miss_weight
            
            # ES is the maximum absolute deviation from zero
            es = running_sum[np.argmax(np.abs(running_sum))]
        
        # Permutation test to compute p-value
        # Randomly permute gene labels and compute ES
        permuted_es = []
        for _ in range(n_permutations):
            # Random permutation of gene positions
            permuted_hits = np.random.choice(n_genes, size=n_hits, replace=False)
            permuted_running_sum = np.zeros(n_genes + 1)
            
            if n_misses == 0:
                permuted_es.append(1.0)
            else:
                hit_weight = 1.0 / n_hits if n_hits > 0 else 0.0
                miss_weight = 1.0 / n_misses if n_misses > 0 else 0.0
                
                for i in range(1, n_genes + 1):
                    if i - 1 in permuted_hits:
                        permuted_running_sum[i] = permuted_running_sum[i-1] + hit_weight
                    else:
                        permuted_running_sum[i] = permuted_running_sum[i-1] - miss_weight
                
                perm_es = permuted_running_sum[np.argmax(np.abs(permuted_running_sum))]
                permuted_es.append(perm_es)
        
        permuted_es = np.array(permuted_es)
        
        # Compute p-value: fraction of permutations with |ES| >= |observed ES|
        p_value = np.mean(np.abs(permuted_es) >= np.abs(es))
        
        # Normalize ES by mean of permuted ES (NES = ES / mean(|permuted_ES|))
        mean_abs_perm_es = np.mean(np.abs(permuted_es))
        if mean_abs_perm_es > 0:
            nes = es / mean_abs_perm_es
        else:
            nes = es
        
        results[gene_set_name] = {
            'es': es,
            'nes': nes,
            'p_value': p_value,
            'n_hits': n_hits,
            'n_genes_in_set': len(gene_set_set),
            'n_genes_total': n_genes
        }
    
    # Compute FDR if multiple gene sets
    if len(results) > 1:
        p_values = [results[gs]['p_value'] for gs in results.keys()]
        _, fdr_values, _, _ = smm.multipletests(p_values, method='fdr_bh')
        for i, gene_set_name in enumerate(results.keys()):
            results[gene_set_name]['fdr'] = fdr_values[i]
    else:
        for gene_set_name in results.keys():
            results[gene_set_name]['fdr'] = results[gene_set_name]['p_value']
    
    return results


def run_gsea_on_de_results(de_results, gene_names, gene_sets, method_name):
    """
    Run GSEA on DE test results.
    
    Parameters:
    -----------
    de_results : dict
        Results from de_test_with_scores, containing 'lfc', 'test_statistic', 'p_vals', 'adj_pvals'
        and optionally 'log_abs_statistic' for LN_test
    gene_names : array-like (n_genes,)
        Gene names or identifiers corresponding to the genes in de_results
    gene_sets : dict
        Dictionary where keys are gene set names and values are sets/lists of genes
    method_name : str
        Name of the DE method used
        
    Returns:
    --------
    gsea_results : dict
        GSEA results for each gene set
    ranked_genes : array
        Genes ranked by GSEA score (highest to lowest)
    gsea_scores : array
        GSEA scores for each gene
    """
    lfcs = de_results['lfc']
    test_statistics = de_results['test_statistic']
    p_vals = de_results['p_vals']
    log_abs_statistic = de_results.get('log_abs_statistic', None)
    
    # Compute GSEA ranking scores
    gsea_scores = compute_gsea_ranking_scores(lfcs, test_statistics, p_vals, method_name, log_abs_statistic)
    
    # Rank genes by GSEA score (descending: highest score first)
    ranked_indices = np.argsort(gsea_scores)[::-1]
    ranked_genes = np.array(gene_names)[ranked_indices]
    
    # Run GSEA
    gsea_results = run_gsea_simple(ranked_genes, gene_sets)
    
    return gsea_results, ranked_genes, gsea_scores


def run_gsea_for_all_methods(all_de_results, gene_names, gene_sets, n_permutations=1000, use_gseapy_prerank=True, gsea_ranks_only=False):
    """
    Run GSEA for all DE methods.
    
    Parameters:
    -----------
    all_de_results : dict
        Results from de_test_with_scores, with keys for each method
    gene_names : array-like (n_genes,)
        Gene names or identifiers
    gene_sets : dict
        Dictionary where keys are gene set names and values are sets/lists of genes
    n_permutations : int
        Number of permutations for GSEA (default: 1000)
    use_gseapy_prerank : bool
        If True, use gseapy's prerank function (recommended). If False, use custom implementation.
    gsea_ranks_only : bool
        If True, use ranks only (ignore score magnitudes). If False, use test_statistic scores.
        
    Returns:
    --------
    all_gsea_results : dict
        Dictionary with method names as keys, each containing:
        - 'gsea_results': GSEA results for each gene set
        - 'ranked_genes': Genes ranked by GSEA score
        - 'gsea_scores': GSEA scores for each gene
    """
    all_gsea_results = {}
    
    # Print GSEA mode information (all methods use same mode)
    first_method = list(all_de_results.keys())[0]
    print(f"  Running GSEA for {len(all_de_results)} methods: {', '.join(all_de_results.keys())}")
    if gsea_ranks_only:
        print(f"    Mode: RANKS ONLY (scores ignored, using rank positions)")
    else:
        print(f"    Mode: SCORES (using test_statistic values)")
    print(f"    Implementation: {'gseapy.prerank' if use_gseapy_prerank else 'Custom GSEA'}")
    
    for method_idx, (method_name, de_results) in enumerate(all_de_results.items(), 1):
        print(f"\n  [{method_idx}/{len(all_de_results)}] Processing method: {method_name}...")
        # Get GSEA ranking scores (test_statistic for all methods)
        lfcs = de_results['lfc']
        test_statistics = de_results['test_statistic']
        p_vals = de_results['p_vals']
        log_abs_statistic = de_results.get('log_abs_statistic', None)
        
        gsea_scores = compute_gsea_ranking_scores(
            lfcs, test_statistics, p_vals, method_name, log_abs_statistic
        )
        
        # If ranks_only mode, we need to create a deterministic ranking first
        # Sort by test_statistic (desc), then lfc (desc), then gene name (asc)
        if gsea_ranks_only:
            gene_names_str = np.asarray(gene_names, dtype=str)
            sort_keys = (gene_names_str, -lfcs, -test_statistics)
            ranked_indices = np.lexsort(sort_keys)
            ranked_genes = gene_names_str[ranked_indices]
            
            # Assign rank positions (1 = highest rank, n = lowest rank)
            # Higher test_statistic and lfc get lower rank numbers (better rank)
            n_genes = len(ranked_genes)
            ranked_scores = np.arange(n_genes, 0, -1, dtype=float)  # [n, n-1, ..., 2, 1] for descending order
            
            if method_name == first_method:
                print(f"    Ranks-only mode: Sorted by test_statistic (desc), lfc (desc), gene name (asc)")
                print(f"    Assigned rank positions 1-{n_genes} to {n_genes} genes (lower rank = higher test_statistic/lfc)")
        else:
            # Normal mode: sort by test_statistic (desc), then lfc (desc), then gene name (asc)
            gene_names_str = np.asarray(gene_names, dtype=str)
            sort_keys = (gene_names_str, -lfcs, -gsea_scores)
            ranked_indices = np.lexsort(sort_keys)
            ranked_genes = gene_names_str[ranked_indices]
            ranked_scores = gsea_scores[ranked_indices]
            
            # Check for duplicate scores and report
            unique_scores, counts = np.unique(ranked_scores, return_counts=True)
            duplicate_mask = counts > 1
            if np.any(duplicate_mask):
                n_duplicate_scores = np.sum(counts[duplicate_mask])
                n_genes_with_duplicates = np.sum(duplicate_mask)
                if method_name == first_method:
                    print(f"    Note: {n_genes_with_duplicates} unique score values appear multiple times")
                    print(f"    ({n_duplicate_scores} genes share these scores, {n_duplicate_scores/len(ranked_scores)*100:.2f}% of genes)")
                    print(f"    Using stable sort to ensure consistent ordering for tied genes")
        
        # Ensure scores are numeric (float) and genes are strings
        ranked_scores = np.asarray(ranked_scores, dtype=float)
        ranked_genes = np.asarray(ranked_genes, dtype=str)
        
        # Run GSEA using gseapy prerank or custom implementation
        if use_gseapy_prerank:
            try:
                import gseapy as gp
                
                # Create a pandas Series with gene names as index and scores/ranks as values
                # This is the format expected by gseapy prerank
                # Ensure gene names are strings and scores are numeric
                # Convert gene names to strings explicitly and remove any NaN/inf values
                ranked_genes_str = []
                ranked_scores_clean = []
                for g, s in zip(ranked_genes, ranked_scores):
                    g_str = str(g)
                    # Skip NaN or inf scores
                    if np.isfinite(s):
                        ranked_genes_str.append(g_str)
                        ranked_scores_clean.append(float(s))
                
                if len(ranked_genes_str) == 0:
                    raise ValueError("No valid (finite) scores found for GSEA")
                
                # Ensure scores are float64
                ranked_scores_float = np.asarray(ranked_scores_clean, dtype=np.float64)
                
                rnk = pd.Series(ranked_scores_float, index=ranked_genes_str)
                
                # Verify types before passing to gseapy
                if rnk.dtype not in [np.float64, np.float32]:
                    rnk = rnk.astype(np.float64)
                
                # Ensure gene sets have string keys and values
                # Also filter to only include genes that are in our ranked list
                ranked_genes_set = set(ranked_genes_str)
                gene_sets_str = {}
                for set_name, gene_set in gene_sets.items():
                    # Convert gene set members to strings and filter to only those in ranked list
                    gene_set_str = {str(g) for g in gene_set if str(g) in ranked_genes_set}
                    if len(gene_set_str) > 0:  # Only include non-empty gene sets
                        gene_sets_str[str(set_name)] = gene_set_str
                
                if len(gene_sets_str) == 0:
                    raise ValueError("No valid gene sets found after filtering")
                
                # Run prerank GSEA
                # gseapy prerank expects gene_sets as a dict or GMT file path
                # Create a temporary directory for output (gseapy requires outdir)
                import tempfile
                import os
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Try to catch and provide more detailed error information
                    try:
                        prerank_res = gp.prerank(
                            rnk=rnk,
                            gene_sets=gene_sets_str,
                            processes=1,  # Single process to avoid issues
                            permutation_num=n_permutations,
                            outdir=tmpdir,
                            format='png',
                            seed=42,
                            verbose=False,
                            no_plot=True  # Don't generate plots
                        )
                    except Exception as e:
                        # Provide more context about the error
                        error_msg = str(e)
                        if method_name == first_method:
                            print(f"    Error details: {error_msg}")
                            print(f"    Ranked genes type: {type(ranked_genes_str[0]) if len(ranked_genes_str) > 0 else 'N/A'}")
                            print(f"    Scores type: {rnk.dtype}")
                            print(f"    Number of ranked genes: {len(rnk)}")
                            print(f"    Number of gene sets: {len(gene_sets_str)}")
                            print(f"    Sample gene name: {ranked_genes_str[0] if len(ranked_genes_str) > 0 else 'N/A'}")
                            print(f"    Sample score: {ranked_scores_float[0] if len(ranked_scores_float) > 0 else 'N/A'}")
                        raise
                    
                    # Extract results from gseapy output
                    gsea_results = {}
                    if prerank_res.res2d is not None and len(prerank_res.res2d) > 0:
                        for idx, row in prerank_res.res2d.iterrows():
                            # Safely extract and convert Tag % to a number
                            tag_pct = row.get('Tag %', 0)
                            if pd.notna(tag_pct):
                                # Handle different formats:
                                # 1. String percentages (e.g., "50.0%")
                                # 2. Fractions (e.g., "21/43")
                                # 3. Numeric values
                                if isinstance(tag_pct, str):
                                    # Check if it's a fraction (contains '/')
                                    if '/' in tag_pct:
                                        # Parse fraction like "21/43"
                                        parts = tag_pct.split('/')
                                        if len(parts) == 2:
                                            try:
                                                numerator = float(parts[0])
                                                denominator = float(parts[1])
                                                tag_pct = (numerator / denominator) * 100 if denominator > 0 else 0
                                            except (ValueError, ZeroDivisionError):
                                                tag_pct = 0
                                        else:
                                            tag_pct = 0
                                    else:
                                        # Remove % sign and convert to float
                                        tag_pct = float(tag_pct.replace('%', ''))
                                else:
                                    tag_pct = float(tag_pct)
                                n_genes_in_set = int(tag_pct * len(ranked_genes) / 100)
                            else:
                                n_genes_in_set = 0
                            
                            # Safely extract Lead_genes count
                            lead_genes = row.get('Lead_genes', '')
                            if pd.notna(lead_genes) and str(lead_genes) != '':
                                n_hits = len(str(lead_genes).split(','))
                            else:
                                n_hits = 0
                            
                            gsea_results[row['Term']] = {
                                'es': float(row.get('ES', np.nan)) if pd.notna(row.get('ES', np.nan)) else np.nan,
                                'nes': float(row.get('NES', np.nan)) if pd.notna(row.get('NES', np.nan)) else np.nan,
                                'p_value': float(row.get('NOM p-val', np.nan)) if pd.notna(row.get('NOM p-val', np.nan)) else np.nan,
                                'fdr': float(row.get('FDR q-val', np.nan)) if pd.notna(row.get('FDR q-val', np.nan)) else np.nan,
                                'n_hits': n_hits,
                                'n_genes_in_set': n_genes_in_set,
                                'n_genes_total': len(ranked_genes)
                            }
                    else:
                        # If no results, create empty results for each gene set
                        for gene_set_name in gene_sets.keys():
                            gsea_results[gene_set_name] = {
                                'es': 0.0,
                                'nes': 0.0,
                                'p_value': 1.0,
                                'fdr': 1.0,
                                'n_hits': 0,
                                'n_genes_in_set': len(gene_sets[gene_set_name]),
                                'n_genes_total': len(ranked_genes)
                            }
                
            except ImportError:
                if method_name == first_method:
                    print(f"    Warning: gseapy not available. Falling back to custom GSEA implementation.")
                use_gseapy_prerank = False
            except Exception as e:
                # More detailed error reporting for debugging
                error_msg = str(e)
                if method_name == first_method:
                    print(f"    Warning: Error using gseapy prerank: {error_msg}")
                    print(f"    Error type: {type(e).__name__}")
                    import traceback
                    print(f"    Traceback (first few lines):")
                    tb_lines = traceback.format_exc().split('\n')[:5]
                    for line in tb_lines:
                        if line.strip():
                            print(f"      {line}")
                    print(f"    Falling back to custom GSEA implementation.")
                # Disable gseapy for this method, but continue with custom implementation
                use_gseapy_prerank = False
        
        if not use_gseapy_prerank:
            # Use custom GSEA implementation
            if method_name == first_method:
                print(f"    Using custom GSEA implementation (simple algorithm)")
            gsea_results = run_gsea_simple(ranked_genes, gene_sets, n_permutations=n_permutations)
        
        all_gsea_results[method_name] = {
            'gsea_results': gsea_results,
            'ranked_genes': ranked_genes,
            'gsea_scores': gsea_scores
        }
        
        print(f"  [{method_idx}/{len(all_de_results)}] Completed GSEA for {method_name}: {len(gsea_results)} gene sets analyzed")
    
    print(f"\n  GSEA completed for all {len(all_gsea_results)} methods")
    return all_gsea_results


def format_gsea_results(all_gsea_results):
    """
    Format GSEA results into a pandas DataFrame for easy viewing.
    
    Parameters:
    -----------
    all_gsea_results : dict
        Results from run_gsea_for_all_methods
        
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with columns: method, gene_set, es, nes, p_value, fdr, n_hits, n_genes_in_set
    """
    rows = []
    for method_name, method_results in all_gsea_results.items():
        gsea_results = method_results['gsea_results']
        for gene_set_name, gs_results in gsea_results.items():
            rows.append({
                'method': method_name,
                'gene_set': gene_set_name,
                'es': gs_results['es'],
                'nes': gs_results['nes'],
                'p_value': gs_results['p_value'],
                'fdr': gs_results.get('fdr', gs_results['p_value']),
                'n_hits': gs_results.get('n_hits', 0),
                'n_genes_in_set': gs_results.get('n_genes_in_set', 0),
                'n_genes_total': gs_results.get('n_genes_total', 0)
            })
    
    df = pd.DataFrame(rows)
    return df


def save_gsea_results(all_gsea_results, output_file):
    """
    Save GSEA results to a CSV file.
    
    Parameters:
    -----------
    all_gsea_results : dict
        Results from run_gsea_for_all_methods
    output_file : str
        Path to output CSV file
    """
    df = format_gsea_results(all_gsea_results)
    df.to_csv(output_file, index=False)
    print(f"GSEA results saved to {output_file}")
    return df

