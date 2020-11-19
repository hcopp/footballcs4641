import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pandas import DataFrame
from plotly.offline import iplot
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.mixture import GaussianMixture
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler


def find_num_clusters(max_clusters):
    # SETUP #
    data = pd.read_csv('./data/scaled.csv')
    x = data.values
    range_n_clusters = range(2, max_clusters + 1)
    all_silhouette_scores = []

    # CLUSTER ITERATION #
    for n_clusters in range_n_clusters:
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.2, 1]
        plt.xlim([-0.2, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        plt.ylim([0, len(x) + (n_clusters + 1) * 10])
        # Initialize the clusterer with n_clusters value
        clusterer = GaussianMixture(n_clusters)
        cluster_labels = clusterer.fit_predict(x)
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed clusters
        silhouette_avg = silhouette_score(x, cluster_labels)
        all_silhouette_scores.append(silhouette_avg)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(x, cluster_labels)
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.get_cmap("Spectral")(float(i) / n_clusters)
            plt.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
            # Label the silhouette plots with their cluster numbers at the middle
            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
        plt.title(("Silhouette analysis for GMM clustering with %d clusters"
                   % n_clusters), fontsize=10, fontweight='bold')
        plt.xlabel("Silhouette coefficient values")
        plt.ylabel("Cluster label")
        # The vertical line for average silhouette score of all the values
        plt.axvline(x=silhouette_avg, color="red", linestyle="--")
        plt.yticks([])  # Clear the yaxis labels / ticks
        plt.xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.savefig(f'./graphs/silhouette_{n_clusters}_clusters.png')
        plt.show()
    plt.scatter(range_n_clusters, all_silhouette_scores)
    plt.plot(range_n_clusters, all_silhouette_scores)
    plt.xticks(range_n_clusters)
    plt.title("Silhouette Scores of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Scores")
    plt.savefig(f'./graphs/silhouette_scores.png')
    plt.show()


def make_visualization(viz_type, dim, n_clusters, plot_x, cluster_labels, colors):
    data = []
    for i in range(n_clusters):
        cluster = plot_x[cluster_labels == i]
        if dim < 3:
            data.append(
                go.Scatter(
                    x=cluster[f"{viz_type}C1_{dim}d"],
                    y=cluster['zero'] if dim == 1 else cluster[f"{viz_type}C2_{dim}d"],
                    mode="markers",
                    name=f"Cluster {i + 1}",
                    marker=dict(color=colors[i]),
                    text=None
                )
            )
        else:
            data.append(
                go.Scatter3d(
                    x=cluster[f"{viz_type}C1_{dim}d"],
                    y=cluster[f"{viz_type}C2_{dim}d"],
                    z=cluster[f"{viz_type}C3_{dim}d"],
                    mode="markers",
                    name=f"Cluster {i + 1}",
                    marker=dict(color=colors[i]),
                    text=None
                )
            )
    dim_dict = {1: "One", 2: "Two", 3: "Three"}
    title = f"Visualizing Clusters in {dim_dict[dim]} Dimension{'s' if dim > 1 else ''} Using " \
            f"{'PCA' if viz_type == 'P' else 'T-SNE'}"
    layout = dict(title=title,
                  xaxis=dict(title=f'{viz_type}C1', ticklen=5, zeroline=False),
                  yaxis=dict(title=f'{viz_type}C2' if dim > 1 else '', ticklen=5, zeroline=False)
                  )
    fig = dict(data=data, layout=layout)
    iplot(fig)


def visualize_clusters(n_clusters, dim):
    data = pd.read_csv('./data/scaled.csv')
    x = data.values
    clusterer = GaussianMixture(n_clusters)
    cluster_labels = clusterer.fit_predict(x)
    scaler = StandardScaler().fit(pd.read_csv('./data/cleaned.csv'))
    cluster_label_means = []
    cluster_label_stds = []
    for n in range(n_clusters):
        print(f'Number of Plays in Cluster {n + 1}: {len(x[cluster_labels == n])}')
        means = np.average(scaler.inverse_transform(x)[cluster_labels == n], axis=0).round(4)
        stds = np.std(scaler.inverse_transform(x)[cluster_labels == n], axis=0).round(4)
        cluster_label_means.append(means)
        cluster_label_stds.append(stds)
    DataFrame(cluster_label_means, columns=data.columns)\
        .to_csv("./data/Cluster_Means.csv")
    DataFrame(cluster_label_stds, columns=data.columns)\
        .to_csv("./data/Cluster_STDs.csv")
    # One dimension
    if dim == 1:
        tsne_1d = TSNE(n_components=1)
        pca_1d = PCA(n_components=1)
        tcs_1d = pd.DataFrame(tsne_1d.fit_transform(x))
        pcs_1d = pd.DataFrame(pca_1d.fit_transform(x))
        tcs_1d.columns = ["TC1_1d"]
        pcs_1d.columns = ["PC1_1d"]
        plot_x_tsne = pd.concat([data, tcs_1d], axis=1, join='inner')
        plot_x_pca = pd.concat([data, pcs_1d], axis=1, join='inner')
        plot_x_tsne["zero"] = 0
        plot_x_pca["zero"] = 0
    # Two dimensions
    elif dim == 2:
        tsne_2d = TSNE(n_components=2)
        pca_2d = PCA(n_components=2)
        tcs_2d = pd.DataFrame(tsne_2d.fit_transform(x))
        pcs_2d = pd.DataFrame(pca_2d.fit_transform(x))
        tcs_2d.columns = ["TC1_2d", "TC2_2d"]
        pcs_2d.columns = ["PC1_2d", "PC2_2d"]
        plot_x_tsne = pd.concat([data, tcs_2d], axis=1, join='inner')
        plot_x_pca = pd.concat([data, pcs_2d], axis=1, join='inner')
    # Three dimensions
    elif dim == 3:
        tsne_3d = TSNE(n_components=3)
        pca_3d = PCA(n_components=3)
        tcs_3d = pd.DataFrame(tsne_3d.fit_transform(x))
        pcs_3d = pd.DataFrame(pca_3d.fit_transform(x))
        tcs_3d.columns = ["TC1_3d", "TC2_3d", "TC3_3d"]
        pcs_3d.columns = ["PC1_3d", "PC2_3d", "PC3_3d"]
        plot_x_tsne = pd.concat([data, tcs_3d], axis=1, join='inner')
        plot_x_pca = pd.concat([data, pcs_3d], axis=1, join='inner')
    else:
        print("Invalid Dimension...")
        return
    graph_colors = ['red', 'blue', 'green', 'yellow', 'pink', 'purple',
                    'black', 'lightskyblue', 'orange', 'darkred',
                    'salmon', 'cyan', 'lime', 'slategray', 'teal',
                    'peru', 'orchid', 'crimson', 'thistle', 'lavender']
    make_visualization('T', dim, n_clusters, plot_x_tsne, cluster_labels, graph_colors)
    make_visualization('P', dim, n_clusters, plot_x_pca, cluster_labels, graph_colors)


def plot_all_points():
    data = pd.read_csv('./data/scaled.csv')
    x = data.values
    x_index = np.argmax(np.var(data, axis=0))
    y_index = np.argmax(np.var(data.drop(columns=[data.columns[x_index]]), axis=0))
    print(np.var(data.drop(columns=[data.columns[x_index]]), axis=0))
    column_name = data.drop(columns=[data.columns[x_index]]).columns[y_index]
    y_index = data.columns.tolist().index(column_name)
    plt.plot(x[:3000, x_index], x[:3000, y_index], '.b')
    plt.xlabel(data.columns[x_index])
    plt.ylabel(data.columns[y_index])
    plt.show()


def dbscan():
    data = pd.read_csv('./data/scaled.csv')
    x = data.values
    db = DBSCAN(eps=3).fit(x)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(x, labels))
    unique_labels = set(labels)
    colors = [cm.get_cmap("Spectral")(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = x[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], '.b', markerfacecolor=tuple(col))
        xy = x[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], '.b', markerfacecolor=tuple(col))
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


if __name__ == '__main__':
    plot_all_points()
    dbscan()
    find_num_clusters(20)
    visualize_clusters(3, 2)
