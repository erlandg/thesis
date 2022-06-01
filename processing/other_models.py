import numpy as np
from sklearn import cluster
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score, adjusted_rand_score, f1_score
from scipy.optimize import linear_sum_assignment

import click
from collections import OrderedDict



DATASET = "knn_edema_dataset"


def correct_predictions(pred, ri, ci):
    idxs = [np.where(pred == _) for _ in ri]
    for idx, cor_val in zip(idxs, ci):
        pred[idx] = cor_val
    return pred


def get_metrics(pred, labels):
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
    pred = correct_predictions(pred, ri, ci)
    nmi_ = normalized_mutual_info_score(labels, pred, average_method="geometric")
    if np.abs(nmi_) >= 1.: nmi_ = 0. # Solve unstable nmi values
    metrics = {
        "acc": np.sum(np.diag(ordered))/np.sum(ordered),
        "cmat": ordered,
        "nmi": nmi_,
        "ari": adjusted_rand_score(labels, pred),
        "f1": f1_score(labels, pred, average="macro")
    }
    return metrics


def load_data(fname):
    out = [np.load(fname)["labels"], np.load(fname)["graph"]]
    out.extend(OrderedDict({k: v for k, v in np.load(fname).items() if "view_" in k}).values())
    return tuple(out)


def kmeans(data, labels, N=1, **kmeans_kwargs):
    metrics = {}
    for _ in range(N):
        pred = KMeans(
            n_clusters = len(np.unique(labels)),
            **kmeans_kwargs
        ).fit_predict(data)
        mets = get_metrics(pred, labels)
        for k, v in mets.items():
            if k == "cmat": continue
            if k not in metrics:
                metrics[k] = v
            else:
                if v > metrics[k]:
                    metrics[k] = v
    return metrics


def spectral_clustering(data, labels, N=1, **spectral_clustering_kwargs):
    metrics = {}
    for _ in range(N):
        pred = SpectralClustering(
            n_clusters = len(np.unique(labels)),
            affinity = "precomputed",
            **spectral_clustering_kwargs
        ).fit_predict(data)
        mets = get_metrics(pred, labels)
        for k, v in mets.items():
            if k == "cmat": continue
            if k not in metrics:
                metrics[k] = v
            else:
                if v > metrics[k]:
                    metrics[k] = v
    return metrics


@click.command()
@click.option("-d", "--dataset")
def main(dataset=None):
    print(f"Analysing {dataset}")
    if dataset is None: dataset = DATASET
    labels, graph, *views = load_data(f"data/processed/{dataset}.npz")
    graph = graph / graph.max()
    # print(f"{dataset} k-means accuracy : {kmeans(views[0], labels, N=100)}")
    print(f"{dataset} spectral clustering accuracy : {spectral_clustering(graph, labels, N=100)}")
    print()



if __name__ == "__main__":
    main()