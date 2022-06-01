import enum
import numpy as np
from sklearn import semi_supervised
import torch as th
import torch_geometric as thgeo
import networkx as nx
import pandas as pd
import torch.nn.functional as F
from torchvision import transforms
from torch_geometric.loader import GraphSAINTSampler


from typing import Optional
import config
from lib.graph_operations import normalise_graph
from helpers import string2int


class GraphSampler(GraphSAINTSampler):
    r"""The GraphSAINT random walk sampler class (see
    :class:`~torch_geometric.loader.GraphSAINTSampler`).

    Args:
        walk_length (int): The length of each random walk.
    """
    def __init__(self, data, batch_size: int, walk_length: int,
                 num_steps: int = 1, sample_coverage: int = 0,
                 save_dir: Optional[str] = None, log: bool = True,
                 semi_supervised_N: int = None, **kwargs):
        self.labels = data.y
        self.walk_length = walk_length
        self.semi_supervised_N = semi_supervised_N
        self.update_index
        super().__init__(data, batch_size, num_steps, sample_coverage,
                         save_dir, log, **kwargs)

    @property
    def __filename__(self):
        return (f'{self.__class__.__name__.lower()}_{self.walk_length}_'
                f'{self.sample_coverage}.pt')

    @property
    def update_index(self):
        self.semi_supervised = -th.ones_like(self.labels).to(th.int32)
        if self.semi_supervised_N is not None:
            semi_supervised_idx = np.random.choice(range(len(self.labels)), size = self.semi_supervised_N, replace = False)
            self.semi_supervised[semi_supervised_idx] = self.labels[semi_supervised_idx]

    def __getitem__(self, idx):
        node_idx = self.__sample_nodes__(self.__batch_size__).unique()
        adj, _ = self.adj.saint_subgraph(node_idx)
        return node_idx, adj, self.semi_supervised[node_idx]

    def __collate__(self, data_list):
        assert len(data_list) == 1
        node_idx, adj, labels = data_list[0]

        data = self.data.__class__()
        data.num_nodes = node_idx.size(0)
        row, col, edge_idx = adj.coo()
        data.edge_index = th.stack([row, col], dim=0)

        for key, item in self.data:
            if key in ['edge_index', 'num_nodes']:
                continue
            if isinstance(item, th.Tensor) and item.size(0) == self.N:
                data[key] = item[node_idx]
            elif isinstance(item, th.Tensor) and item.size(0) == self.E:
                data[key] = item[edge_idx]
            else:
                data[key] = item

        if self.sample_coverage > 0:
            data.node_norm = self.node_norm[node_idx]
            data.edge_norm = self.edge_norm[edge_idx]

        return data, labels.to(config.DEVICE)

    def __sample_nodes__(self, batch_size):
        start = th.randint(0, self.N, (batch_size, ), dtype=th.long)
        node_idx = self.adj.random_walk(start.flatten(), self.walk_length)
        return node_idx.view(-1)


class TensorDataset(th.utils.data.TensorDataset):
    def __init__(self, *args, graph=None, inference_device = None, **kwargs) -> None:
        super(TensorDataset, self).__init__(*args, **kwargs)
        self.graph = graph
        if inference_device is not None:
            self.device = inference_device
        else:
            self.device = config.DEVICE
       
    def __getitem__(self, index):
        if self.graph is not None:
            graph_ = th.cat([row[index].reshape(1,-1) for row in self.graph[index]], axis=0)
            out = (tuple(tensor[index].to(self.device) for tensor in self.tensors), graph_.to(self.device))
        else:
            out = (tensor[index].to(self.device) for tensor in self.tensors)
        return out


def sampler(dataset, **kwargs):
    return th.utils.data.sampler.BatchSampler(
        th.utils.data.sampler.RandomSampler(dataset), **kwargs
    )


def get_graph_sampler(dataset, loader = thgeo.loader.DataLoader, **loader_kwargs):
    graph, *views, labels = dataset.graph.to_sparse(), *dataset.tensors
    data = thgeo.data.Data(y = labels.to(th.int), edge_index = graph.indices(), edge_attr = graph.values(), num_nodes = len(labels))
    for i, view in enumerate(views):
        data[f"view_{i}"] = view
    if loader == thgeo.loader.DataLoader:
        return(loader([data], loader_kwargs["batch_size"]))
    return loader(data, **loader_kwargs)


def _load_npz(name):
    return np.load(config.DATA_DIR / "processed" / f"{name}.npz")


def _fix_labels(l):
    uniq = np.unique(l)[None, :]
    new = (l[:, None] == uniq).argmax(axis=1)
    return new


def alter_G(G, buffer=None):
    N = G.shape[0]
    eye = np.eye(N)
    if np.diag(G).all():
        G = G / np.diag(G).reshape(-1,1) # divide rows by diagonal
    else:
        G = G + eye
    if buffer is not None:
        G_min, G_diff = G.min(), G.max() - G.min()
        buffer *= G_diff
        G = ((G - G_min) / G_diff) * (G_diff - buffer) + G_min + buffer # alter range(G) from [min, max] to [min + buffer, max]
    return normalise_graph(G)


def normalise_image(image, mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225]), downscale=False):
    if downscale: pp = [transforms.CenterCrop(224)]
    else: pp = []
    preprocess = transforms.Compose(pp + [
        transforms.Normalize(mean=mean, std=std)
    ])

    if image.shape[1] == 1:
        image = th.cat(
            tuple(th.Tensor(image) for _ in range(3)),
            dim=1
        ).type(th.float)
    else:
        image = th.Tensor(image).to(th.float)
    # return transforms.Normalize(mean=mean, std=std)(image).numpy()
    return preprocess(image).numpy()


def normalise_vector(vector, mean=0, std=1):
    v_std = vector.std(0)
    idx = v_std == 0
    if idx.any():
        diff = vector[:,idx].max() - vector[:,idx].min()
        if diff.any():
            v_std[idx] = diff
        else:
            v_std[idx] = 1
    return ((vector - vector.mean(0)) / v_std) * std + mean


def load_dataset(
    name,
    n_samples=None,
    eval_sample_proportion=None,
    select_views=None,
    select_labels=None,
    label_counts=None,
    noise_sd=None,
    noise_views=None,
    load_graph = False,
    graph_tol=None,
    to_dataset=True,
    return_diagnoses=False,
    seed=None,
    normalise_images=False,
    downscale_244=False,
    normalise_data=True,
    **kwargs
):
    # To save space
    labels = _load_npz(name)["labels"]
    views = [_load_npz(name)[f"view_{i}"] for i in range(_load_npz(name)["n_views"])]
    if load_graph:
        graph = _load_npz(name)["graph"]
        graph = graph / graph.max()
        graph_bool = True
    else:
        graph = None
        graph_bool = False

    if normalise_images:
        for i, view in enumerate(views):
            if len(view.shape) >= 3:
                views[i] = normalise_image(view, downscale = downscale_244)
    if normalise_data:
        for i, view in enumerate(views):
            if len(view.shape) < 3:
                views[i] = normalise_vector(view)

    imported_diags = False
    if return_diagnoses:
        try:
            diagnoses = pd.read_csv(
                "../data/processed/diagnoses.csv",
                header=0,
                index_col=0
            )
            assert len(diagnoses) == len(labels), "length of diagnoses.csv not equal to length of views"
            imported_diags = True
        except (FileNotFoundError, AssertionError):
            print("Failed to import diagnoses.csv")
            diagnoses = None
            eval_diagnoses = None

    if select_labels is not None:
        mask = np.isin(labels, select_labels)
        labels = labels[mask]
        views = [v[mask] for v in views]
        if graph_bool: graph = graph[np.ix_(mask, mask)]
        if imported_diags: diagnoses = diagnoses.iloc[mask]
        labels = _fix_labels(labels)

    if label_counts is not None:
        idx = []
        unique_labels = np.unique(labels)
        assert len(unique_labels) == len(label_counts)
        for l, n in zip(unique_labels, label_counts):
            rng = np.random.default_rng(seed=string2int(seed))
            _idx = rng.choice(np.where(labels == l)[0], size=n, replace=False)
            idx.append(_idx)

        idx = np.concatenate(idx, axis=0)
        labels = labels[idx]
        views = [v[idx] for v in views]
        if graph_bool: graph = graph[np.ix_(idx, idx)]
        if imported_diags: diagnoses = diagnoses.iloc[idx]

    if n_samples is not None:
        rng = np.random.default_rng(seed=string2int(seed))
        idx = rng.choice(labels.shape[0], size=min(labels.shape[0], int(n_samples)), replace=False)
        labels = labels[idx]
        views = [v[idx] for v in views]
        if graph_bool: graph = graph[np.ix_(idx, idx)]
        if imported_diags: diagnoses = diagnoses.iloc[idx]

    if select_views is not None:
        if not isinstance(select_views, (list, tuple)):
            select_views = [select_views]
        views = [views[i] for i in select_views]

    if noise_sd is not None:
        assert noise_views is not None, "'noise_views' has to be specified when 'noise_sd' is not None."
        if not isinstance(noise_views, (list, tuple)):
            noise_views = [int(noise_views)]
        for v in noise_views:
            rng = np.random.default_rng(seed=string2int(seed))
            views[v] += rng.normal(loc=0, scale=float(noise_sd), size=views[v].shape)

    views = [v.astype(np.float32) for v in views]
    if graph_bool:
        # graph = alter_G(graph, buffer=graph_tol)
        graph = th.Tensor(graph)

    if eval_sample_proportion is not None:
        rng = np.random.default_rng(seed=string2int(seed))
        idx = rng.choice(
            labels.shape[0],
            size=int((1 - eval_sample_proportion) * graph.shape[0]),
            replace=False,
        )
        eval_idx = ~np.isin(np.arange(graph.shape[0]), idx)

        eval_views = [v[eval_idx] for v in views]
        views = [v[idx] for v in views]
        eval_labels = labels[eval_idx]
        labels = labels[idx]
        if graph_bool:
            eval_graph = graph[np.ix_(eval_idx, eval_idx)]
            graph = graph[np.ix_(idx, idx)]
        if imported_diags:
            eval_diagnoses = diagnoses.iloc[eval_idx]
            diagnoses = diagnoses.iloc[idx]
        elif return_diagnoses:
            eval_diagnoses = None

    if to_dataset:
        if graph_bool:
            dataset = TensorDataset(*[th.Tensor(v) for v in views],
                                    th.Tensor(labels),
                                    graph=graph,
                                    inference_device=config.DEVICE)
            if eval_sample_proportion is not None:
                eval_dataset = TensorDataset(*[th.Tensor(v) for v in eval_views],
                                             th.Tensor(eval_labels),
                                             graph=eval_graph,
                                             inference_device=config.DEVICE)
        else:
            dataset = TensorDataset(*[th.Tensor(v) for v in views],
                                    th.Tensor(labels),
                                    inference_device=config.DEVICE)
            if eval_sample_proportion is not None:
                eval_dataset = TensorDataset(*[th.Tensor(v) for v in eval_views],
                                             th.Tensor(eval_labels),
                                             inference_device=config.DEVICE)

    else:
        dataset = (views, labels, graph)
        if eval_sample_proportion is not None:
            eval_dataset = (eval_views, eval_labels, eval_graph)


    if eval_sample_proportion is not None:
        out = [(dataset, graph_bool), eval_dataset]
    else:
        out = [dataset, graph_bool]

    if return_diagnoses:
        if eval_sample_proportion is not None:
            out.append((diagnoses, eval_diagnoses))
        else:
            out.append((diagnoses,))
    
    return (*out,)