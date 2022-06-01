import enum
import numpy as np
import torch
import torch_geometric as thg
import scipy.sparse as sp
import networkx as nx
import pickle as pkl
from scipy.fftpack import fft

import sys
from pathlib import Path
from glob import glob

import config


DATASETS = {
    "Amazon": [thg.datasets.Amazon, ["Computers", "Photo"]],
    "Planetoid": [thg.datasets.Planetoid, ["Cora", "CiteSeer", "PubMed"]],
}


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def download_data():
    for key, thg_funcs in DATASETS.items():
        path = config.PROJECT_ROOT / "data" / "processed"
        if type(thg_funcs[-1]) == list:
            for dataset_name in thg_funcs[-1]:
                curr_path = path / f"{key}_{dataset_name}"
                curr_path.mkdir(exist_ok=True, parents=True)
                thg_funcs[0](root = curr_path, name = dataset_name)
        else:
            curr_path = path / f"{key}"
            curr_path.mkdir(exist_ok=True, parents=True)
            thg_funcs[0](root = curr_path)


def save_npz(save_fname, force = False, n_views = None, labels = None, graph = None, **views):
    for _, __ in {"n_views": n_views, "labels": labels, "graph": graph}.items():
        assert __ is not None, f"{_} not provided."
    assert len(views) == n_views

    if (not save_fname.exists()) or force:
        np.savez(save_fname, n_views = n_views, labels = labels, graph = graph, **views)
    else:
        print(f"Found existing dataset {save_fname}.")


def process_npz(dataset_str, npz):
    out = {"n_views": 2}
    out["labels"] = npz["labels"]
    out["graph"] = sp.csr_matrix((npz["adj_data"], npz["adj_indices"], npz["adj_indptr"]), shape = npz["adj_shape"]).todense()
    out["view_0"] = sp.csr_matrix((npz["attr_data"], npz["attr_indices"], npz["attr_indptr"]), shape = npz["attr_shape"]).todense()
    out["view_1"] = fft(out["view_0"])
    print(f"{dataset_str} shapes :")
    print(f"Number of classes : {len(np.unique(out['labels']))}")
    print(f"Adjacency matrix : {out['graph'].shape}")
    print(f"Features shape : {out['view_0'].shape}")
    return out


def save_data(dataset_str, save_filename, force=False): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    path = config.DATA_DIR / 'raw' / dataset_str
    objects = []
    for i in range(len(names)):
        with open(f"{path / f'ind.{dataset_str}.{names[i]}'}", 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(f"{str(path)}/ind.{dataset_str}.test.index")
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    nx_graph = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(nx_graph)
    edges = nx_graph.edges()

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    print(f"{dataset_str} shapes :")
    print(f"Number of classes : {labels.shape[1]}")
    print(f"Adjacency matrix : {adj.shape}")
    print(f"Features shape : {features.shape}")

    if (not save_filename.exists()) or force:
        np.savez(
            save_filename,
            n_views = 2,
            labels = labels.argmax(1),
            graph = adj.todense(),
            view_0 = features.todense(),
            view_1 = fft(features.todense()),
        )
    else:
        print(f"Found existing dataset {save_filename}.")
    # return sp.coo_matrix(adj), features.todense(), labels, idx_train, idx_val, idx_test


def main():
    npzs = glob(str(config.DATA_DIR / "raw" / "**" / "*.npz"))
    for npz_fname in npzs:
        dataset = npz_fname.split("/")[-1].replace(".npz","")
        with np.load(npz_fname) as npz:
            save_npz(
                save_fname = config.DATA_DIR / "processed" / f"{dataset}.npz",
                **process_npz(dataset, npz)
            )
    for dataset in {'pubmed', 'citeseer'}:
        save_data(
            dataset,
            save_filename = config.DATA_DIR / "processed" / f"{dataset}.npz"
        )
    


if __name__ == "__main__":
    main()