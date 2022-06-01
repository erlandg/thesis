import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsTransformer
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import SpectralClustering
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment



DATASET = "edema_dataset"


def ordered_cmat(labels, pred):
    """
    Compute the confusion matrix and accuracy corresponding to the best cluster-to-class assignment.

    :param labels: Label array
    :type labels: np.array
    :param pred: Predictions array
    :type pred: np.array
    :return: Accuracy and confusion matrix
    :rtype: Tuple[float, np.array]
    """
    cmat = confusion_matrix(labels, pred)
    ri, ci = linear_sum_assignment(-cmat)
    ordered = cmat[np.ix_(ri, ci)]
    acc = np.sum(np.diag(ordered))/np.sum(ordered)
    return acc, ordered


def plot(hist_dict, fname, title=None, **plot_kwargs):
    if title is not None: plt.title(title)
    plt.bar(hist_dict.keys(), hist_dict.values(), **plot_kwargs)
    plt.ylabel("Accuracy")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.clf()


def cluster_acc(A, labels, **kmeans_kwargs):
    embedding = SpectralEmbedding(affinity="precomputed").fit_transform(A)
    pred = KMeans(n_clusters = 2, **kmeans_kwargs).fit_predict(embedding)
    return ordered_cmat(labels, pred)[0]


def knn(A, labels, n_neighbour_range=(50,400), N=20, **plot_kwargs):
    accs = {}
    best_knn = None
    best_acc = 0
    best_n = None
    for n_ in np.linspace(*n_neighbour_range, N):
        n = round(n_)
        knn_ = np.asarray(KNeighborsTransformer(
            n_neighbors = round(n),
            metric="precomputed"
        ).fit_transform(A).todense())
        knn_ = (knn_ + knn_.T) / 2
        acc_ = cluster_acc(np.asarray(knn_), labels)
        accs[n] = acc_
        if acc_ > best_acc:
            best_knn = knn_.copy()
            best_acc = acc_
            best_n = n
    print(f"k-NN accuracy :")
    for k, v in accs.items():
        print(f"{v} at {k} neighbours.")
    plot(accs, "knn_hist.png", title="k-nearest neighbours accuracy", **plot_kwargs)
    return best_knn


def threshold(A, labels, threshold_range=(.1,.95), N=20, **plot_kwargs):
    A = A/A.max()
    accs = {}
    best_thresh_graph = None
    best_acc = 0
    best_thresh = None
    for thresh_ in np.linspace(*threshold_range, N):
        A_ = A.copy()
        A_[A_ < thresh_] = 0.
        acc_ = cluster_acc(A_, labels)
        accs[thresh_] = acc_
        if acc_ > best_acc:
            best_thresh_graph = A_.copy()
            best_acc = acc_
            best_thresh = thresh_
    print(f"Thresholding accuracy :")
    for k, v in accs.items():
        print(f"{v} at threshold {k}.")
    plot(accs, "thresh_hist.png", title="thresholding neighbours accuracy", **plot_kwargs)
    return best_thresh_graph


if __name__ == "__main__":
    labels = np.load(f"data/processed/{DATASET}.npz")["labels"]
    graph = np.load(f"data/processed/{DATASET}.npz")["graph"]

    # labels = np.load("data/processed/graph_supervised.npz", allow_pickle=True)["labels"]
    # idx_ = labels == np.unique(labels)[0]
    # labels[idx_] = 1
    # labels[~idx_] = 0
    # labels = labels.astype(int)
    # graph = np.load("data/processed/graph_supervised.npz", allow_pickle=True)["graph"]

    # graph_knn = knn(graph, labels, width = 10)
    # graph_thresh = threshold(graph, labels, width = .02)

    knn_final = knn_ = np.asarray(KNeighborsTransformer(
        n_neighbors = 140,
        metric="precomputed"
    ).fit_transform(graph).todense())
    knn_final = (knn_final + knn_final.T) / 2

    print(cluster_acc(knn_final, np.load(f"data/processed/{DATASET}.npz")["labels"]))

    np.savez(
        f"data/processed/knn_{DATASET}",
        n_views = np.load(f"data/processed/{DATASET}.npz")["n_views"],
        labels = np.load(f"data/processed/{DATASET}.npz")["labels"],
        graph = knn_final,
        view_0 = np.load(f"data/processed/{DATASET}.npz")["view_0"],
        view_1 = np.load(f"data/processed/{DATASET}.npz")["view_1"],
        view_2 = np.load(f"data/processed/{DATASET}.npz")["view_2"]
    )
    np.savez(
        f"data/processed/knn_{DATASET}_images",
        n_views = np.load(f"data/processed/{DATASET}.npz")["n_views"],
        labels = np.load(f"data/processed/{DATASET}.npz")["labels"],
        graph = knn_final,
        view_0 = np.load(f"data/processed/{DATASET}.npz")["view_0"],
    )
    np.savez(
        f"data/processed/knn_{DATASET}_no_images",
        n_views = np.load(f"data/processed/{DATASET}.npz")["n_views"],
        labels = np.load(f"data/processed/{DATASET}.npz")["labels"],
        graph = knn_final,
        view_0 = np.load(f"data/processed/{DATASET}.npz")["view_1"],
        view_1 = np.load(f"data/processed/{DATASET}.npz")["view_2"]
    )

    print()
