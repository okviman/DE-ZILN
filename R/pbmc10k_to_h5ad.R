rm(list=ls())
library(anndata)
library(basilisk)
library(data.table)
library(dplyr)
library(ggplot2)
library(reticulate)
library(Seurat)
library(splatter)
library(zellkonverter)

use_python("/Users/sjun6/opt/anaconda3/envs/nbsr/bin/python")

methods <- c("wilcox", "wilcox_limma", "t", "MAST", "negbinom")
output_path <- "10X_PBMC_10K/"
if (!dir.exists(output_path)) {
  dir.create(output_path, recursive = T)
}

MIN_READS <- 3
MIN_CELLS <- 5

dat <- readRDS("10X_PBMC_10K/pbmc10k_cd4_memory.rds")

cts <- LayerData(dat, assay = "RNA", layer="counts")
dim(cts)
row_idxs <- which(rowSums(cts > MIN_READS) >= MIN_CELLS)
length(row_idxs)

sub_dat <- dat[row_idxs,]
out_path <- paste0("10X_PBMC_10K/memory_CD4.h5ad")
sce <- as.SingleCellExperiment(sub_dat)
writeH5AD(sce, file = out_path)

