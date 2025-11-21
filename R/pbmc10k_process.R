rm(list=ls())
library(data.table)
library(mclust)
library(tidyverse)
library(Seurat)

library(Seurat)
MIN_CELLS <- 3
seurat <- Read10X_h5("sc5p_v2_hs_PBMC_10k_filtered_feature_bc_matrix.h5")
output_path <- "10X_PBMC_10K/"
if (!dir.exists(output_path)) {
  dir.create(output_path, recursive = T)
}

adt_counts <- seurat$`Antibody Capture`
rownames(adt_counts) <- stringr::str_replace(string = rownames(adt_counts), pattern = "_TotalC", replacement = "")
pbmc10k <- SeuratObject::CreateSeuratObject(counts = seurat$`Gene Expression`, project = "PBMC10k")
pbmc10k[["ADT"]] <- SeuratObject::CreateAssayObject(counts = adt_counts)
# Filtering.
pbmc10k[["percent.mt"]] <- PercentageFeatureSet(pbmc10k, pattern = "^MT-", assay = "RNA")
pbmc10k$ADT_nCount   <- Matrix::colSums(GetAssayData(pbmc10k, assay = "ADT", layer = "counts"))
pbmc10k$ADT_nFeature <- Matrix::colSums(GetAssayData(pbmc10k, assay = "ADT", layer = "counts") > 0)
keep_cells <- with(pbmc10k@meta.data,
                   nFeature_RNA >= 200 & 
                     nFeature_RNA <= 6000 &
                     percent.mt   <= 10 &
                     ADT_nCount   >= 500)
pbmc10k <- subset(pbmc10k, cells = colnames(pbmc10k)[keep_cells])


# Identify cell types.
# Normalize ADT data using CLR.
pbmc10k <- NormalizeData(pbmc10k, assay = "ADT", normalization.method = "CLR")
adt_clr_transformed <- t(GetAssayData(pbmc10k, assay = "ADT", slot = "data"))
# Fit Gaussian mixture model to identify CD3+ and CD4++ cells.
fit_mclust <- function(x, num_clusters) {
  fit <- mclust::Mclust(x, G = num_clusters)
  # Order the clusters.
  labels <- rep("", num_clusters)
  labels[1] <- "-"
  for (i in 1:(num_clusters - 1))
  {
    labels[i+1] <- paste(rep("+", i), collapse = "")
  }
  cl_id_ordered <- order(fit$parameters$mean, decreasing = FALSE)
  reorder_map <- integer(num_clusters)
  reorder_map[cl_id_ordered] <- 1:num_clusters
  cluster_id_reordered <- reorder_map[fit$classification]
  final_labels <- labels[cluster_id_reordered]
  result_df <- data.frame(
    x = x,
    cluster_id_reordered = cluster_id_reordered,
    label = final_labels
  )
  return(result_df)
}
adt_clr_transformed_dt <- data.table(barcode = rownames(adt_clr_transformed), adt_clr_transformed)
adt_clr_transformed_dt <- adt_clr_transformed_dt %>% 
  mutate(
    CD3_cluster = fit_mclust(CD3, num_clusters = 2)$label,
    CD4_cluster = fit_mclust(CD4, num_clusters = 3)$label,
    CD45RA_cluster = fit_mclust(CD45RA, num_clusters = 2)$label
  )

# Identify CD3+ cells.
pl <- ggplot(adt_clr_transformed_dt, aes(CD3, fill=CD3_cluster)) + 
  geom_density() +
  theme_bw() + 
  ylab("Density") +
  ggtitle("CD3 CLR distribution with GMM clustering") + 
  labs(fill = "CD3")
ggsave("10X_PBMC_10K/CD3_density_plot.pdf", plot = pl)

pl <- ggplot(adt_clr_transformed_dt, aes(CD4, fill=CD4_cluster)) + 
  geom_density() +
  theme_bw() + 
  ylab("Density") +
  ggtitle("CD4 CLR distribution with GMM clustering") + 
  labs(fill = "CD4")
ggsave("10X_PBMC_10K/CD4_density_plot.pdf", plot = pl)

# Take a subset of CD3+ and CD4++ cells and identify CD45RA- cells.
adt_clr_transformed_dt %>% 
  filter(CD3_cluster == "+", CD4_cluster == "++") %>% 
  ggplot(aes(CD45RA, fill=CD45RA_cluster)) + 
  geom_density() +
  theme_bw() + 
  ylab("Density") +
  ggtitle("CD45RA CLR distribution with GMM clustering") + 
  labs(fill = "CD45RA") -> pl
pl
ggsave("10X_PBMC_10K/CD45RA_density_plot.pdf", plot = pl)

adt_clr_transformed_dt %>% 
  filter(CD3_cluster == "+" & CD4_cluster == "++" & CD45RA_cluster == "-") %>% 
  select(barcode) -> cd4_memory_cells

pbmc10k_cd4_memory <- subset(pbmc10k, cells = cd4_memory_cells$barcode)
dim(pbmc10k_cd4_memory)
saveRDS(pbmc10k_cd4_memory, file = "10X_PBMC_10K/pbmc10k_cd4_memory.rds")

# Save all of the data.
saveRDS(pbmc10k, file = "10X_PBMC_10K/pbmc10k.rds")

