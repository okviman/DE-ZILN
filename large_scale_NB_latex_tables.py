import pandas as pd

df1 = pd.read_csv("/home/oskar/phd/DE-ZILN/simul/test/NB_test_results/d1_vs_d2_01_nde_mu_100.csv")
df2 = pd.read_csv("/home/oskar/phd/DE-ZILN/simul/test/NB_test_results/results.csv")

df1.drop(columns=['recall'], inplace=True)

df2.rename(columns={'acc': 'accuracy', 'prec': 'precision'}, inplace=True)
df2['method'] = df2['method'].apply(lambda x: 'Seurat ' + x)

df = pd.concat([df1, df2], ignore_index=True)
df_avg = df.groupby(['method', 'dispersion']).mean().reset_index().drop(columns=['rep_no'])
df_avg['method'] = df_avg['method'].apply(lambda x: x.replace('_', ' '))
df_avg['method'] = df_avg['method'].apply(lambda x: x.replace('Seurat t', 'Seurat t-test'))
df_avg['method'] = df_avg['method'].apply(lambda x: x.replace('Seurat wilcox', 'Seurat wilcoxon'))
df_avg['method'] = df_avg['method'].apply(lambda x: x.replace('wilcoxon', 'Wilcoxon'))
df_avg['method'] = df_avg['method'].apply(lambda x: x.replace('t-test', '$t$-test'))
df_avg['dispersion'] = df_avg['dispersion'].apply(lambda x: str(round(x, 1)))
df_avg = df_avg[df_avg.method != 'Seurat negbinom']
df_avg = df_avg[df_avg.method != 'Scanpy $t$-test overestim var']

print(df_avg.sort_values(by=["dispersion", 'method']).to_latex(index=False, float_format='%.3f'))
# df.to_csv("/home/oskar/phd/DE-ZILN/simul/test/NB_test_results/avg_results.csv")

