"""
This script contains helper functions for machine learning, including pre-analysis
steps e.g., using screeplot and silhouette score to determine number of cluster
for k-means.
"""
import gc

import cupy as cp
from cuml.cluster import KMeans
from cuml.metrics.cluster import silhouette_score

import seaborn as sns
import matplotlib.pyplot as plt

from common_utils import plot_utils
from common_utils.style import *

def _fit_kmeans(data, n_clusters, n_init=100):
    """
    Fit data to gpu k-means.

    params
    ===
    data: cupy ndarray, data on gpu.

    n_clusters: int, number of clusters.

    return
    ===
    km_clf
    
    labels:
    """
    # initiate kmeans
    km_clf = KMeans(
        n_clusters=n_clusters,
        init="random",
        n_init=n_init,
    )

    # fit data to kmeans
    labels = km_clf.fit_predict(data)

    return km_clf, labels


def compute_metrics(data, n_clusters, n_init=100):
    km, labels = _fit_kmeans(data, n_clusters)

    # get inertia
    inertia = km.inertia_
    # tear down model to free gpu memory
    del km
    gc.collect()

    # get silhouette_score
    sil_score = silhouette_score(data, labels)

    return float(inertia), float(sil_score)

def find_optimal_k(data, fig_dir) -> float:
    """
    determine number of clusters by making screeplot.

    NOTE to parallelise in cpu:
    from joblib import Parallel, delayed
    n_jobs = -2 # use all cpu cores but 2

    silhouette_scores = Parallel(n_jobs=n_jobs)(
        delayed(_compute_silhouette)(n, data) for n in n_clusters
    )

    params
    ===
    data: ndarry.
        y-axis(0): items/samples/things-you-want-to-cluster, e.g., channels
        x-axis(1): features/variables/datapoints, e.g., datapoints in time
    """
    ks = range(2, 10)

    # sum of squared distance of samples to their closest cluster centre
    iners = []
    sil_scores = []

    logging.info("\n> transfer data to gpu")
    gpu_data = cp.asarray(data)

    for k in ks:
        logging.info(f"\n> get k-means metrics with {k} clusters")
        iner, sil_score = compute_metrics(gpu_data, k)
        iners.append(iner)
        sil_scores.append(sil_score)

    fig, axes = plt.subplots(
        ncols=2,
        sharey=False,
    )

    from kneed import KneeLocator
    kn = KneeLocator(
        ks,
        iners,
        curve="convex", 
        direction="decreasing",
    )
    elbow_point = kn.knee
    logging.info(f"\nOptimal number of clusters according to elbow method: "
                f"{elbow_point}")

    sns.lineplot(
        x=ks,
        y=iners,
        ax=axes[0],
        marker="o",
    )
    axes[0].set_xlabel("Number of Clusters")
    axes[0].set_ylabel("Sum of Squares")

    sns.lineplot(
        x=ks,
        y=sil_scores,
        ax=axes[1],
        marker="o",
    )
    axes[1].set_xlabel("Number of Clusters")
    axes[1].set_ylabel("Silhouette Scores")
    plot_utils.save(path=fig_dir + "_inertia_sil_scores.pdf")

    optimal_k = ks[sil_scores.index(max(sil_scores))]

    logging.info(f"\nOptimal number of clusters according to "
                f"silhouette scores: {optimal_k}")

    return optimal_k


def k_means_clustering(data, fig_dir, n_clusters=None, repeats=1000) -> list:
    """
    Use k-means to cluster channels
    
    params
    ===
    n_clusters: int
        number of clusters

    data: ndarray
        y-axis(0): items/samples/things-you-want-to-cluster, e.g., channels
        x-axis(1): features/variables/datapoints, e.g., datapoints in time

    """
    if n_clusters == None:
        n_clusters = find_optimal_k(data, fig_dir)

    logging.info(f"\n> Do k-means clustering with k = {n_clusters}.")

    km, labels = _fit_kmeans(
        data=data,
        n_clusters=n_clusters,
        n_init=repeats,
    )
    #logging.info(f"\n> cluster_centers {km.cluster_centers_}")
    logging.info(f"\n> num of iteration {km.n_iter_}")
    # tear down model to free gpu memory
    del km
    gc.collect()

    # convert to cpu
    Y_pred = cp.asnumpy(labels)

    return Y_pred
