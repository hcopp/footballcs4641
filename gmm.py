import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import numpy as np


def sel_best(arr: np.ndarray, x: int) -> list:
    """
    returns the set of X configurations with shortest distances
    """
    dx = np.argsort(arr)[:x]
    return arr[dx]


def get_num_clusters():
    print('Reading...')
    data = pd.read_csv('./data/scaled.csv').values
    print('Done Reading...')
    n_clusters = np.arange(2, 20)
    sils = []
    sils_err = []
    iterations = 5
    print('Getting Silohuette Scores...')
    for n in n_clusters:
        print(f'{n} Clusters...')
        tmp_sil = []
        for _ in range(iterations):
            print(f'Iteration #{_+1}')
            gmm = GaussianMixture(n, n_init=2).fit(data)
            labels = gmm.predict(data)
            sil = metrics.silhouette_score(data, labels, metric='euclidean')
            tmp_sil.append(sil)
        val = np.mean(tmp_sil)
        err = np.std(tmp_sil)
        sils.append(val)
        sils_err.append(err)
    plt.errorbar(n_clusters, sils, yerr=sils_err)
    plt.title("Silhouette Scores", fontsize=20)
    plt.xticks(n_clusters)
    plt.xlabel("N. of clusters")
    plt.ylabel("Score")


if __name__ == '__main__':
    get_num_clusters()
    # print('Making GMM...')
    # gmm = GaussianMixture(16).fit(data)
    # print('Done making GMM...')
    # print(gmm.score(data))
