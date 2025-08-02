
#!/usr/bin/env python3
"""
Clustering module
"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Function that performs agglomerative clustering on a dataset
    Args:
        X is a numpy.ndarray of shape (n, d)
        dist is the maximum cophenetic distance for all clusters
    Returns:
        clss, a numpy.ndarray of shape (n,)
    """

    Z = scipy.cluster.hierarchy.linkage(X, method='ward')
    fig = plt.figure(figsize=(25, 10))
    dn = scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
    plt.show()
    clss = scipy.cluster.hierarchy.fcluster(Z, t=dist, criterion='distance')

    return clss
