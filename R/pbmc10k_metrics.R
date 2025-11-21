rm(list=ls())
library(data.table)
library(ggplot2)
library(glue)
library(Seurat)
library(tidyverse)
library(xtable)

results <- fread("10X_PBMC_10K/CITE_seq_results.csv")
metrics <- fread("10X_PBMC_10K/CITE_seq_metrics.csv")
names(metrics)[1] <- "Metric"

mean(subset(metrics, Metric == "accuracy")$LN)

tbl <- metrics %>% 
  mutate(Metric = if_else(Metric %in% c("fpr", "tpr", "fnr", "tnr"), 
                          str_to_upper(Metric), 
                          str_to_title(Metric))) %>% 
  pivot_longer(cols = c("LN", "Wilcoxon", "t-test"), names_to = "Method", values_to = "value")
results_tbl <- tbl %>% 
  group_by(Method, Metric) %>% 
  summarise(n = n(), 
            mean = mean(value, na.rm=TRUE), 
            se = sd(value, na.rm=TRUE)/sqrt(n),
            .groups = "drop")

metric_order <- c("Accuracy", "Precision", "Recall", "TPR", "TNR", "FNR", "FPR", "F1")
xt <- results_tbl %>% 
  #mutate(cell = sprintf("%.3f ± %.3f", mean, se)) %>%
  mutate(
    Metric = factor(Metric, metric_order),
    Method = factor(Method, c("LN", "t-test", "Wilcoxon")),
    cell = sprintf("%.3f", mean)) %>%
  select(Method, Metric, cell) %>%
  pivot_wider(names_from = Method, values_from = cell) %>%
  arrange(Metric) %>% 
  xtable()
print(xt, include.rownames = FALSE, type="latex", file = "10X_PBMC_10K/CITE_seq_metrics.tex")

results[is_signal_gene == TRUE,.(mean((ln_lfc - true_lfc)^2),
                                 mean((scanpy_lfc - true_lfc)^2))] 
results_long <- results[is_signal_gene == TRUE] %>% 
  select(ln_lfc, scanpy_lfc, true_lfc) %>% 
  pivot_longer(cols = c("ln_lfc", "scanpy_lfc"), names_to = "Method", values_to = "LFC") %>% 
  mutate(Method = recode(Method, 
                         "ln_lfc" = "LN's test", 
                         "scanpy_lfc" = "Scanpy"))

pl <- results_long %>% 
  ggplot(aes(x = true_lfc, y = LFC)) +
  geom_point(alpha=0.5, size=0.5) + 
  geom_abline(slope=1, intercept=0, color="red", linetype="dashed") +
  theme_bw() +
  xlab("True LFC") +
  ylab("Estimated LFC") +
  facet_grid(~ Method) 
pl
ggsave("10X_PBMC_10K/CITE_seq_lfc_plot.png", pl)

pl <- results_long %>% 
  ggplot(aes(x = true_lfc, y = true_lfc - LFC)) +
  geom_point(alpha=0.5, size=0.5) + 
  geom_abline(slope=0, intercept=0, color="red", linetype="dashed") +
  theme_bw() +
  xlab("True LFC") +
  ylab("Bias") +
  facet_grid(~ Method) 
pl
ggsave("10X_PBMC_10K/CITE_seq_lfc_error_plot.png", pl)

# Load Seurat results
seurat_lfc_results <- readRDS("10X_PBMC_10K/seurat_lfc_results.rds")
seurat_lfc_results %>% 
  filter(true_lfcs != 0) %>% 
  ggplot(aes(true_lfcs, seurat_lfc)) + 
  geom_point() +
  geom_abline(slope=1, intercept=0, color="red", linetype="dashed")

# Merge `results` with `seurat_lfc_results`
# Rename V1
seurat_results_dt <- seurat_lfc_results %>% 
  select(gene_names, seurat_lfc, rep)
dim(seurat_results_dt)
results_dt <- results %>% 
  mutate(gene_names = V1, rep = replicate) %>% 
  select(gene_names, ln_lfc, scanpy_lfc, true_lfc, rep)
dim(results_dt)

# Join the two tables by gene_names and rep.
merged_results <- inner_join(results_dt, seurat_results_dt, by = c("gene_names", "rep"))

merged_results_long <- merged_results[true_lfc != 0] %>% 
  select(ln_lfc, scanpy_lfc, seurat_lfc, true_lfc) %>% 
  pivot_longer(cols = c("ln_lfc", "scanpy_lfc", "seurat_lfc"), names_to = "Method", values_to = "LFC") %>% 
  mutate(Method = recode(Method, 
                         "ln_lfc" = "LN's test", 
                         "scanpy_lfc" = "Scanpy",
                         "seurat_lfc" = "Seurat"))
pl <- merged_results_long %>% 
  ggplot(aes(x = true_lfc, y = LFC)) +
  geom_point(alpha=0.5) + 
  geom_abline(slope=1, intercept=0, color="red", linetype="dashed") +
  theme_bw() +
  xlab("True LFC") +
  ylab("Estimated LFC") +
  facet_grid(~ Method) 
ggsave(glue("10X_PBMC_10K/CITE_seq_lfc_plot_all_methods.png"), pl)

pl <- merged_results_long %>% 
  ggplot(aes(x = true_lfc, y = true_lfc - LFC)) +
  geom_point(alpha=0.5) + 
  geom_abline(slope=0, intercept=0, color="red", linetype="dashed") +
  theme_bw() +
  xlab("True LFC") +
  ylab("Bias") +
  facet_grid(~ Method) 
pl
ggsave("10X_PBMC_10K/CITE_seq_lfc_error_plot_all.png", pl)

