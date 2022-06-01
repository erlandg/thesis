import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, spectral_embedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as FDA
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

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
    return acc


def plot(X, labels, title, save_path):
    fig = sns.scatterplot(
        x=X[:,0],
        y=X[:,1],
        hue=labels
    ).set_title(title)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def reduce_plot(X, save_path, method='pca', y=None):
    methods = {
        'pca': PCA,
        'tsne': TSNE,
        'fda': FDA,
    }
    n_components = 2
    if method == 'fda':
        n_components = min(len(np.unique(y)) -  1, X.shape[1])
        assert n_components < 2, "Less than 2 components"
    met = methods[method](n_components=n_components)
    new_X = met.fit_transform(X, y)
    shuffle_idx = np.random.choice(range(new_X.shape[0]), size=new_X.shape[0], replace=False)
    
    plot(new_X[shuffle_idx], labels, "PCA", save_path)


def box_plots(X, names, labels, save_path):
    for i in range(X.shape[1]):
        name = names[i,0]
        X_ = {
            'label 0' : X[:,i][labels == 0],
            'label 1' : X[:,i][labels == 1],
        }
        fig, ax = plt.subplots()
        ax.boxplot(x=X_.values())
        ax.set_xticklabels(X_.keys())
        plt.savefig(save_path+'/'+name.replace("/","-")+'.png', bbox_inches='tight')
        plt.close()
        plt.clf()



chart_cols = pd.read_csv("data/processed/chartevents_cols.csv", header=None)
lab_cols = pd.read_csv("data/processed/labevents_cols.csv", header=None)

npz = np.load("data/processed/graph_dataset.npz")
labels = npz["labels"].astype(int)
static = np.c_[npz["view_1"], npz["view_2"]]
graph = npz["graph"]
names = chart_cols.append(lab_cols).values
# box_plots(static, names, labels, "processing/box-plots")
# eig = np.linalg.eig(graph)[1][:,:2] # N x m
# G_transformed = graph @ eig
# plot(G_transformed, labels, "PCA on graph", "processing/graph_pca.png")
G_transformed = spectral_embedding(graph, n_components=2)
plot(G_transformed, labels, "Spectral embedding of graph", "processing/graph_laplacian.png")


lasso = Lasso(1e-2).fit(static, labels)
coefs = lasso.coef_
red_static = static[:,np.abs(coefs) > 1e-12]
red_names = names[np.abs(coefs) > 1e-12]
dict_ = dict(zip(names[np.abs(coefs) > 1e-12][:,0], np.abs(coefs[np.abs(coefs) > 1e-12])))
sorted_dict_ = {k: v for k, v in sorted(dict_.items(), key=lambda item: item[1], reverse=True)}
# for i, (k, v) in enumerate(sorted_dict_.items()):
#     print(f"Feature #{i+1}: {k} - {v}")


spec_pred = SpectralClustering(n_clusters=len(np.unique(labels)), affinity="precomputed").fit_predict(graph)
acc, _ = ordered_cmat(labels, spec_pred)
print(f"Spectral clustering accuracy: {acc}")


names = ["Decision tree", "Random forest", "AdaBoost", "AdaBoost of Random forest"]
best_accs = [0, 0, 0, 0]
for _ in tqdm(range(20)):
    train_idx = np.random.choice(
        range(red_static.shape[0]),
        size=int(.8*red_static.shape[0]),
        replace=False,
    )
    test_idx = np.random.choice(
        [_ for _ in list(range(red_static.shape[0])) if _ not in train_idx],
        size=red_static.shape[0]-len(train_idx),
        replace=False,
    )

    trees = [
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        AdaBoostClassifier(RandomForestClassifier())
    ]
    [tree.fit(red_static[train_idx], labels[train_idx]) for tree in trees]
    preds = [tree.predict(red_static[test_idx]) for tree in trees]
    for i, pred in enumerate(preds):
        acc = (pred==labels[test_idx]).mean()
        if acc > best_accs[i]:
            best_accs[i] = acc

for i, acc in enumerate(best_accs):
    print(f"Best {names[i]} test accuracy : {acc}")

