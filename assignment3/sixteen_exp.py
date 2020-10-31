cluster_algs = ['kmeans', 'GMM']
dimm_red_technique = ['LDA', 'PCA', 'ICA', 'RP']
dataset = ['1', '2']

def run_exp(exp_name,dim_tech,ds):

    pass

for c in cluster_algs:
    for d in dimm_red_technique:
        for d1 in dataset:
            run_exp(c,d,d1)
