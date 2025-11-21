rm(list=ls())
library(data.table)
library(ggplot2)
library(glue)
library(Seurat)
library(tidyverse)
library(xtable)

# Run Seurat to get LFC estimates.
replicates <- 0:99
lfc_results <- list()
for (rep_no in replicates)
{
  data_path <- glue("10X_PBMC_10K/replicates/rep{rep_no}/")
  X <- read.csv(glue("{data_path}/X.csv"), header=F)
  Y <- read.csv(glue("{data_path}/Y.csv"), header=F)
  true_lfcs <- read.csv(glue("{data_path}/true_lfcs.csv"), header=F)
  gene_names_df <- read.csv(glue("{data_path}/gene_names.csv"), header = FALSE)
  cts <- cbind(t(X), t(Y))
  rownames(cts) <- gene_names_df$V1
  seurat_obj <- CreateSeuratObject(counts = cts)
  seurat_obj$group <- c(rep("A", nrow(X)), rep("B", nrow(Y)))
  seurat_obj <- NormalizeData(seurat_obj)
  Idents(seurat_obj) <- seurat_obj$group
  results <- FindMarkers(seurat_obj, 
                         ident.1 = "A", 
                         ident.2 = "B", 
                         logfc.threshold = 0, 
                         #pseudocount.use = 1e-9,
                         test.use = "wilcox")
  true_lfcs_dt <- data.table(gene_names = gene_names_df$V1, true_lfcs=true_lfcs$V1)
  results_dt <- data.table(gene_names = rownames(results), 
                           seurat_lfc = results$avg_log2FC)
  merged <- merge(true_lfcs_dt, results_dt, by="gene_names")
  pl <- ggplot(merged, aes(true_lfcs, seurat_lfc)) + 
    geom_point() + 
    geom_abline(slope=1, intercept=0, color="red", linetype="dashed")
  ggsave(glue("{data_path}/seurat_lfc_plot.png"), pl)
  saveRDS(merged, glue("{data_path}/seurat_lfc_results.rds"))
  lfc_results[[(rep_no+1)]] <- merged
}

lfc_results_dt <- rbindlist(lfc_results, idcol = TRUE)
lfc_results_dt$rep <- lfc_results_dt$.id - 1
saveRDS(lfc_results_dt, "10X_PBMC_10K/seurat_lfc_results.rds")

pl <- ggplot(lfc_results_dt, aes(true_lfcs, seurat_lfc)) + 
  geom_point() +
  geom_abline(slope=1, intercept=0, color="red", linetype="dashed")
pl
ggsave(glue("10X_PBMC_10K/CITE_seq_seurat_lfc_plot.png"), pl)

# Look at one specific replicate to see why Seurat's LFC is worse.
rep_no <- 2
data_path <- glue("10X_PBMC_10K/replicates/rep{rep_no}/")
X <- read.csv(glue("{data_path}/X.csv"), header=F)
Y <- read.csv(glue("{data_path}/Y.csv"), header=F)
gene_names <- read.csv(glue("{data_path}/gene_names.csv"), header = FALSE)
colnames(X) <- colnames(Y) <- gene_names$V1
lfc_results_rep2 <- readRDS(glue("{data_path}/seurat_lfc_results.rds"))
genes_to_investigate <- lfc_results_rep2[abs(seurat_lfc - true_lfcs) > 2 & true_lfcs != 0,gene_names]
colMeans(X[,genes_to_investigate] == 0)
colMeans(Y[,genes_to_investigate] == 0)

colMeans(X[,genes_to_investigate])
colMeans(Y[,genes_to_investigate])

