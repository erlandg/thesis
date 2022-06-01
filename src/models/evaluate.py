import enum
import os
import argparse
import numpy as np
import pandas as pd
import torch as th
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score
from math import ceil
from torch_geometric.loader import GraphSAINTRandomWalkSampler

import config
import helpers
import analysis
from models.build_model import from_file
from data.load import TensorDataset, load_dataset, sampler, get_graph_sampler, GraphSampler

IGNORE_IN_TOTAL = ("contrast", "semi_supervised")
LABELS_STRING = {
    0: "Negative",
    1: "Positive"
}


def calc_metrics(labels, pred):
    """
    Compute metrics.

    :param labels: Label tensor
    :type labels: th.Tensor
    :param pred: Predictions tensor
    :type pred: th.Tensor
    :return: Dictionary containing calculated metrics
    :rtype: dict
    """
    acc, cmat, (ri, ci) = helpers.ordered_cmat(labels, pred, return_ri_ci = True)
    pred = helpers.correct_predictions(pred, ri, ci)
    nmi_ = normalized_mutual_info_score(labels, pred, average_method="geometric")
    if np.abs(nmi_) > 1: nmi_ = 0. # Solve unstable nmi values
    metrics = {
        "acc": acc,
        "cmat": cmat,
        "nmi": nmi_,
        "ari": adjusted_rand_score(labels, pred),
        "f1": f1_score(labels, pred, average="macro")
    }
    return metrics


def get_log_params(net):
    """
    Get the network parameters we want to log.

    :param net: Model
    :type net:
    :return:
    :rtype:
    """
    params_dict = {}
    weights = []
    if getattr(net, "fusion", None) is not None:
        with th.no_grad():
            weights = net.fusion.get_weights(softmax=True)

    elif hasattr(net, "attention"):
        weights = net.weights

    for i, w in enumerate(helpers.npy(weights)):
        params_dict[f"fusion/weight_{i}"] = w

    if hasattr(net, "discriminators"):
        for i, discriminator in enumerate(net.discriminators):
            d0, dv = helpers.npy([discriminator.d0, discriminator.dv])
            params_dict[f"discriminator_{i}/d0/mean"] = d0.mean()
            params_dict[f"discriminator_{i}/d0/std"] = d0.std()
            params_dict[f"discriminator_{i}/dv/mean"] = dv.mean()
            params_dict[f"discriminator_{i}/dv/std"] = dv.std()

    return params_dict


def get_eval_data(dataset, sampler):
    """
    Create a dataloader to use for evaluation

    :param dataset: Inout dataset.
    :type dataset: th.utils.data.Dataset
    :param n_eval_samples: Number of samples to include in the evaluation dataset. Set to None to use all available
                           samples.
    :type n_eval_samples: int
    :param batch_size: Batch size used for training.
    :type batch_size: int
    :return: Evaluation dataset loader
    :rtype: th.utils.data.DataLoader
    """
    eval_loader = th.utils.data.DataLoader(
        dataset,
        sampler = sampler,
        num_workers = 0,
        pin_memory = False
    )
    return eval_loader


def batch_predict(net, eval_data, batch_size, graph_bool = False, return_semi_sup_idx = False):
    """
    Compute predictions for `eval_data` in batches. Batching does not influence predictions, but it influences the loss
    computations.

    :param net: Model
    :type net:
    :param eval_data: Evaluation dataloader
    :type eval_data: th.utils.data.DataLoader
    :param batch_size: Batch size
    :type batch_size: int
    :return: Label tensor, predictions tensor, list of dicts with loss values, array containing mean and std of cluster
             sizes.
    :rtype:
    """
    predictions = []
    labels = []
    losses = []
    cluster_sizes = []
    semi_sup_idx = []

    net.eval()
    with th.no_grad():
        for i, data in enumerate(eval_data):
            if not graph_bool:
                *batch, label = data
                pred = net([_[0] for _ in batch])
                label = label[0]
            else:
                try:
                    data, net.semi_supervised_labels = data
                except ValueError:
                    net.semi_supervised_labels = None
                if net.semi_supervised_labels is not None:
                    semi_sup_idx.extend(net.semi_supervised_labels.tolist())
                data_dict = dict(sorted(data.to_dict().items()))
                batch = [v for k, v in data_dict.items() if "view_" in k]
                label = data.y
                graph = th.sparse_coo_tensor(
                    th.cat((data.edge_index, data.edge_index[[1,0]]), dim = -1),
                    data.edge_attr.repeat(2),
                    (data.num_nodes, data.num_nodes)
                )
                pred = net([x.to(config.DEVICE) for x in batch], graph = graph.to(config.DEVICE)).to("cpu")
            labels.append(helpers.npy(label))
            predictions.append(helpers.npy(pred).argmax(axis=1))

            # Only calculate losses for full batches
            if label.size(0) >= batch_size: 
                batch_losses = net.calc_losses(labels=label, ignore_in_total=IGNORE_IN_TOTAL)
                losses.append(helpers.npy(batch_losses))
                cluster_sizes.append(helpers.npy(pred.sum(dim=0)))

    labels = np.concatenate(labels, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    if len(semi_sup_idx) == 0:
        semi_sup_idx = [-1 for _ in range(len(labels))]
    net.train()
    if not return_semi_sup_idx:
        return labels, predictions, losses, np.array(cluster_sizes).sum(axis=0)
    else:
        return labels, predictions, losses, np.array(cluster_sizes).sum(axis=0), np.array(semi_sup_idx)


def get_logs(cfg, net, eval_data, iter_losses=None, epoch=None, include_params=True, graph_bool=False):
    if iter_losses is not None:
        logs = helpers.add_prefix(helpers.dict_means(iter_losses), "iter_losses")
    else:
        logs = {}
    if (epoch is None) or ((epoch % cfg.eval_interval) == 0) or (epoch == 1):
        labels, pred, eval_losses, cluster_sizes, semi_sup_idx = batch_predict(net, eval_data, cfg.batch_size, graph_bool=graph_bool, return_semi_sup_idx=True)
        eval_losses = helpers.dict_means(eval_losses)
        logs.update(helpers.add_prefix(eval_losses, "eval_losses"))
        logs.update(helpers.add_prefix(calc_metrics(labels[semi_sup_idx == -1], pred[semi_sup_idx == -1]), "metrics"))
        logs.update(helpers.add_prefix({"mean": cluster_sizes.mean(), "sd": cluster_sizes.std()}, "cluster_size"))
    if include_params:
        logs.update(helpers.add_prefix(get_log_params(net), "params"))
    if epoch is not None:
        logs["epoch"] = epoch
    return logs


def eval_run(cfg, cfg_name, experiment_identifier, run, net, eval_data, callbacks=tuple(), load_best=True, graph_bool=False):
    """
    Evaluate a training run. 

    :param cfg: Experiment config
    :type cfg: config.defaults.Experiment
    :param cfg_name: Config name
    :type cfg_name: str
    :param experiment_identifier: 8-character unique identifier for the current experiment
    :type experiment_identifier: str
    :param run: Run to evaluate
    :type run: int
    :param net: Model
    :type net:
    :param eval_data: Evaluation dataloder
    :type eval_data: th.utils.data.DataLoader
    :param callbacks: List of callbacks to call after evaluation
    :type callbacks: List
    :param load_best: Load the "best.pt" model before evaluation?
    :type load_best: bool
    :return: Evaluation logs
    :rtype: dict
    """
    if load_best:
        model_path = helpers.get_save_dir(cfg_name, experiment_identifier, run) / "best.pt"
        if os.path.isfile(model_path):
            net.load_state_dict(th.load(model_path))
        else:
            print(f"Unable to load best model for evaluation. Model file not found: {model_path}")
    logs = get_logs(cfg, net, eval_data, include_params=True, graph_bool=graph_bool)
    for cb in callbacks:
        cb.at_eval(net=net, logs=logs)
    return logs


def eval_experiment(cfg_name, tag, plot=False):
    """
    Evaluate a full experiment

    :param cfg_name: Name of the config
    :type cfg_name: str
    :param tag: 8-character unique identifier for the current experiment
    :type tag: str
    :param plot: Display a scatterplot of the representations before and after fusion?
    :type plot: bool
    """
    max_n_runs = 500
    best_logs = None
    best_run = None
    best_net = None
    best_loss = np.inf
    accuracies = []
    best_acc = 0

    for run in range(max_n_runs):
        try:
            net, *_, cfg = from_file(cfg_name, tag, run, ckpt="best", return_data=True, return_config=True)
        except FileNotFoundError:
            break

        if (config.defaults.CNN in [type(__) for __ in cfg.model_config.backbone_configs]) and any(_.pretrained_model is not None for _ in cfg.model_config.backbone_configs if type(_) == config.defaults.CNN):
            downscale = True
        else:
            downscale = False

        if cfg.dataset_config.eval_sample_proportion is not None:
            (_, graph_bool), dataset = load_dataset(
                **cfg.dataset_config.dict(),
                load_graph = cfg.model_config.__class__ in [config.defaults.GraphCoMVC, config.defaults.GraphMVC],
                seed=tag,
                downscale_244=downscale
            )
        else:
            dataset, graph_bool = load_dataset(
                **cfg.dataset_config.dict(),
                load_graph = cfg.model_config.__class__ in [config.defaults.GraphCoMVC, config.defaults.GraphMVC],
                seed=tag,
                downscale_244=downscale
            )
        del _

        if not graph_bool:
            eval_data = get_eval_data(
                dataset,
                sampler(dataset, batch_size = int(cfg.batch_size), drop_last = False)
            )
        else:
            if cfg.batch_size == len(dataset.tensors[-1]):
                eval_data = get_graph_sampler(
                    dataset,
                    batch_size = int(cfg.batch_size),
                )
            elif cfg.batch_size < len(dataset.tensors[-1]):
                assert cfg.graphsaint_steps is not None
                eval_data = get_graph_sampler(
                    dataset,
                    loader = GraphSampler,
                    num_steps = len(dataset) // int(cfg.batch_size),
                    walk_length = cfg.graphsaint_steps,
                    batch_size = int(cfg.batch_size),
                    semi_supervised_N = None,
                )

        run_logs = eval_run(cfg, cfg_name, tag, run, net, eval_data, load_best=False, graph_bool=graph_bool)
        accuracies.append(run_logs["metrics/acc"])
        del eval_data
        del run_logs["metrics/cmat"]

        # if run_logs["metrics/acc"] > best_acc:
        if run_logs[f"eval_losses/{cfg.best_loss_term}"] < best_loss:
            best_acc = run_logs["metrics/acc"]
            best_loss = run_logs[f"eval_losses/{cfg.best_loss_term}"]
            best_logs = run_logs
            best_run = run
            best_net = net

    print(f"\nBest run was {best_run}.", end="\n\n")
    headers = ["Name", "Value"]
    values = list(best_logs.items())
    print(tabulate(values, headers=headers), "\n")
    pd.DataFrame(values).to_csv(f"../models/{cfg_name}-{tag}/{tag}.csv", header=False, index=False)
    
    if plot:
        print("Plotting")
        plot_representations(
            *from_file(
                cfg_name,
                tag,
                best_run,
                ckpt="best",
                return_data=True,
                return_diagnoses=True
            )[1:],
            best_net,
            batch_size=cfg.batch_size,
            n_views = cfg.model_config.fusion_config.n_views,
            save_folder = f"../models/{cfg_name}-{tag}/"
        )
        plt.savefig(f"../models/{cfg_name}-{tag}/fusion_plot.png", bbox_inches="tight")
    

def plot_representations(views, labels, graph, diagnoses, net, batch_size, n_views, project_method="tsne", save_folder=None):
    assert diagnoses is not None, "diagnoses.csv not provided"
    assert diagnoses.shape[0] == len(labels), "diagnoses.csv does not match dataset"
    output = []
    hiddens = [[] for _ in range(len(net.backbone_outputs))]
    fused = []
    with th.no_grad():
        for i in range(ceil(labels.shape[0] / batch_size)):
            idx_low = batch_size * i
            idx_high = min(batch_size * (i+1), labels.shape[0])
            views_ = [th.tensor(v[idx_low:idx_high], device=config.DEVICE) for v in views]
            if graph is not None:
                graph_ = graph[idx_low:idx_high, idx_low:idx_high].to(config.DEVICE, non_blocking=True)
                output.append(net(views_, graph=graph_).to("cpu"))
            else:
                output.append(net(views_).to("cpu"))

            hiddens_ = helpers.npy(net.backbone_outputs)
            for i in range(len(hiddens)):
                hiddens[i].append(hiddens_[i])

            fused.append(
                helpers.npy(net.fused)
            )

        output = th.cat(output, dim=0)
        _, _, (_, ci) = helpers.ordered_cmat(labels, output.argmax(1), return_ri_ci = True)
        output = output[:,ci]
        assert (output.argmax(1).numpy() == labels).mean() >= 1/len(np.unique(labels))

        pred = helpers.npy(output).argmax(axis=1)
        hidden = np.concatenate([
            np.concatenate(hidden_, axis=0) for hidden_ in hiddens
        ], axis=0)
        fused = np.concatenate(fused, axis=0)

    view_hue = sum([labels.shape[0] * [str(i + 1)] for i in range(n_views)], [])
    fused_hue = [str(l + 1) for l in pred]
    labels_style = n_views * list(labels)
    s = 3
    sample_size_scale = 10000
    sizes = [1 if "Sample" not in _ else 2 for _ in fused_hue]
    sizes_view = n_views * sizes

    if not save_folder is None:
        idx_types = ["static" if len(_.shape) <= 2 else "image" for _ in views]
        if "image" in idx_types:
            analysis.plot_xrays_groupings(
                save_folder+"sample_x-rays.png",
                fused_data=fused,
                raw_data=[view for i, view in enumerate(views) if idx_types[i] == "image"],
                pred=pred,
                true_labels=labels,
                N=6,
                fused_hue=fused_hue,
                view_hue=view_hue,
                labels = LABELS_STRING,
            )
        if "static" in idx_types:
            try:
                if diagnoses is not None:
                    sign_col = analysis.analyse_grouping(
                        diagnoses,
                        [view for i, view in enumerate(views) if idx_types[i] == "static"],
                        pred,
                        save_folder=save_folder
                    )
                    analysis.create_word_cloud(
                        diagnoses,
                        sign_col,
                        pred,
                        save_folder=save_folder,
                        labels = LABELS_STRING,
                    )
            except FileNotFoundError:
                pass
    del views

    view_cmap = "tab10"
    class_cmap = "hls"
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    plot_projection(X=hidden, method=project_method, hue=view_hue, ax=ax[0], title="Before fusion",
                    legend_title="View", hue_order=sorted(list(set(view_hue))), cmap=view_cmap,
                    alpha=.7, style=[f"GT label {int(_)}" for _ in labels_style])
    plot_projection(X=fused, method=project_method, hue=fused_hue, ax=ax[1], title="After fusion",
                    legend_title="Prediction", hue_order=sorted(list(set(fused_hue))), cmap=class_cmap,
                    alpha=.7, style=[f"GT label {int(_)}" for _ in labels])


def plot_projection(X, method, hue, ax, title=None, cmap="tab10", legend_title=None, legend_loc=1, **kwargs):
    X = project(X, method)
    shuffle_idx = np.random.choice(range(X.shape[0]), size=X.shape[0], replace=False)

    pl = sns.scatterplot(
        x=X[shuffle_idx, 0], y=X[shuffle_idx, 1], hue=np.array(hue)[shuffle_idx],
        ax=ax, legend="full", palette=cmap, **kwargs
    )
    leg = pl.get_legend()
    leg._loc = legend_loc
    if title is not None:
        ax.set_title(title)
    if legend_title is not None:
        leg.set_title(legend_title)


def project(X, method):
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2).fit_transform(X)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2).fit_transform(X)
    elif method is None:
        return X
    else:
        raise RuntimeError()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="cfg_name", required=True)
    parser.add_argument("-t", "--tag", dest="tag", required=True)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    eval_experiment(args.cfg_name, args.tag, args.plot)
