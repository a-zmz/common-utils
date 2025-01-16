"""
This script contains helper functions for machine learning, including pre-analysis
steps e.g., using screeplot to determine number of cluster. 
"""

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import silhouette_score
from numba import vectorize, jit, cuda
import numpy as np
import pandas as pd 

import multiprocessing as mp

import seaborn as sns
import matplotlib.pyplot as plt

from analysis.base import fig_dir
from common_utils import plot_utils

#@vectorize(["int16"("int16")], target="cuda")
def screeplot(data, name) -> None:
    """
    determine number of clusters by making screeplot.

    params
    ===
    data: ndarry.
        y-axis(0): items/samples/things-you-want-to-cluster, e.g., channels
        x-axis(1): features/variables/datapoints, e.g., datapoints in time
    """
    iners = [] # sum of squared distance of samples to their closest cluster centre
    ks = range(2, 10)
    for k in ks:
        km_clf = KMeans(
            n_clusters=k,
            init="random",
        )
        km = km_clf.fit(
            X=data,
        )
        iner = km_clf.inertia_
        iners.append(iner)

    from kneed import KneeLocator
    kn = KneeLocator(
        ks,
        iners,
        curve="convex", 
        direction="decreasing",
    )
    elbow_point = kn.knee
    print(f"Optimal number of clusters according to elbow method: {elbow_point}")

    sns.lineplot(
        x=ks,
        y=iners,
    )
    plt.xlabel("Number of Clusters")
    plt.ylabel("Sum of Squares")
    plot_utils.save(path=fig_dir + name + "_screeplot.pdf")

    return None


def _compute_silhouette(n_clusters, data):
    kmeans = KMeans(n_clusters=n_clusters, init="random")
    labels = kmeans.fit_predict(data)
    score = silhouette_score(data, labels)

    return score

def get_silhouette(data, name):
    n_clusters = range(2, 10)

    from joblib import Parallel, delayed
    n_jobs = -2 # use all cpu cores but 2

    silhouette_scores = Parallel(n_jobs=n_jobs)(
        delayed(_compute_silhouette)(n, data) for n in n_clusters
    )

    sns.lineplot(
        x=n_clusters,
        y=silhouette_scores,
    )
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Scores")
    plot_utils.save(path=fig_dir + name + "_silhouette_scores.pdf")

    optimal_k = n_clusters[
        silhouette_scores.index(max(silhouette_scores))
    ]

    print(f"Optimal number of clusters according to silhouette scores: "
        f"{optimal_k}")

    return silhouette_scores, optimal_k


#@cuda.jit(device=True)
def k_means_clustering(data, name, n_clusters=None, repeats=1000) -> list:
    #TODO: use GPU to conduct ml computations using joblib
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
        _, n_clusters = get_silhouette(data, name)

    km_clf = KMeans(
        n_clusters=n_clusters,
        init="random",
        #init="k-means++",#TODO: use default "k-means++" as its faster?
        n_init=repeats,
        #random_state=repeats, # make randomness deterministic, i.e., reproducible
        algorithm="lloyd",
    )

    km = km_clf.fit(
        X=data, # no y label
        sample_weight=None, # can pass array (n_samples, n_features) weight to
    )

    Y_pred = km.predict(
        X=data,
    )

    print(f"\n> cluster_centers {km.cluster_centers_}")
    print(f"\n> num of iteration {km.n_iter_}")

    return Y_pred
