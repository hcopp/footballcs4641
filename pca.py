from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bioinfokit.visuz import cluster

def main(num_components):
    print("Reading...")
    df = pd.read_csv('scaled.csv')
    print("Done Reading...")
    print("Scaling...")
    df_st =  StandardScaler().fit_transform(df)
    print("Done Scaling...")
    '''print("Performing PCA...")
    pca = PCA(n_components=num_components, svd_solver='full').fit(df_st)
    pca_out = pca.transform(df_st)
    print("Done Performing PCA...")
    pc_components = pca.components_
    pc_list = [f'PC{str(i)}' for i in list(range(1, len(pc_components)+1))]
    pc_components_df = pd.DataFrame.from_dict(dict(zip(pc_list, pc_components)))
    pc_components_df['variable'] = df.columns.values
    pc_components_df = pc_components_df.set_index('variable')
    print("Writing PCs to CSV...")
    pd.DataFrame(pca_out).to_csv('PCA.csv')
    print("Done Writing PCs to CSV...")
    cluster.screeplot(obj=[pc_list, pca.explained_variance_ratio_])
    print(get_important_features(pca, df.columns))'''
    # Heat Map doesn't really work becuase so many features...
    corr = df.corr()
    heatmap = sns.heatmap(corr,
        vmin=-1, vmax=1, center=0, cmap='Spectral', square=True, xticklabels=1, yticklabels=1)
    heatmap.set_xticklabels(
        heatmap.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    heatmap.set_yticklabels(
        heatmap.get_yticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    plt.show()

def get_important_features(pca, initial_feature_names):
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(pca.components_.shape[0])]
    most_important_names = [initial_feature_names[most_important[i]] for i in range(pca.components_.shape[0])]
    return most_important_names

if __name__ == '__main__':
    main(.99)
