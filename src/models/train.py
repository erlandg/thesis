from sklearn import semi_supervised
import wandb
import torch as th
from tqdm import tqdm

import config
from config.defaults import CNN
import helpers
from data.load import load_dataset, sampler, get_graph_sampler, GraphSampler
from models import callback
from models.build_model import build_model
from models import evaluate


def wandb_init(name, cfg):
    wandb.login()
    if config.defaults.CNN in [type(__) for __ in cfg.model_config.backbone_configs]:
        try:
            pretrained_cnn = ",".join(
                sorted([_.pretrained_model for _ in cfg.model_config.backbone_configs if type(_) == config.defaults.CNN])
            )
        except TypeError:
            pretrained_cnn = None
    else:
        pretrained_cnn = None

    try:
        graph_layers = cfg.model_config.graph_attention_configs.layers
        attention_features = cfg.model_config.graph_attention_configs.attention_features
        dropout = cfg.model_config.graph_attention_configs.dropout
        mask_power = cfg.model_config.graph_attention_configs.mask_power
    except AttributeError:
        graph_layers = attention_features = dropout = mask_power = None

    parameters = {
        "dataset": cfg.dataset_config.name,
        "n_views": cfg.model_config.fusion_config.n_views,
        "learning_rate": cfg.model_config.optimizer_config.learning_rate,
        "evaluation_set": True if cfg.dataset_config.eval_sample_proportion is not None else False,
        "batch_size": cfg.batch_size,
        "supervision": any("supervised" in _ for _ in cfg.model_config.loss_config.funcs.split("|")),
        "funcs": ", ".join(cfg.model_config.loss_config.funcs.split("|")),
        "n_semi_supervised": cfg.model_config.loss_config.n_semi_supervised,
        "semi_supervised_weight": cfg.model_config.loss_config.semi_supervised_weight,
        "pretrained_cnn": pretrained_cnn,
        "graph_layers": graph_layers,
        "attention_heads": attention_features,
        "graph_dropout": dropout,
        "graph_mask_power": mask_power,
        "shared_graph_weights": cfg.model_config.shared_weights,
        "layer_type": cfg.model_config.cm_config.layer_type,
        "n_hidden": cfg.model_config.cm_config.n_hidden,
        "sigma": cfg.model_config.loss_config.rel_sigma,
        "tau": cfg.model_config.loss_config.tau,
        "delta": cfg.model_config.loss_config.delta,
        "epsilon_features": cfg.model_config.loss_config.epsilon_features,
        "epsilon_structure": cfg.model_config.loss_config.epsilon_structure,
        **{f"view_{i}": str((_,))[1:-2] for i, _ in enumerate(cfg.model_config.backbone_configs)}
    }
    return {
        "config": name,
        **parameters
    }


def warmup(net, loader, cfg, graph_bool=False):
    losses = cfg.warmup_funcs.split("|")
    ignore_in_total = tuple(_ for _ in net.loss.TERM_CLASSES.keys() if _ not in losses)
    for e in tqdm(range(cfg.warmup_epochs), desc="Warm-up"):
        for data in loader:
            if graph_bool:
                # Homogeneous graph data loader
                data_dict = dict(sorted(data.to_dict().items()))
                batch = [v for k, v in data_dict.items() if "view_" in k]
                labels = data.y 
                # Important note! self.graph only considers upper triangle, therefore "manual" fill
                graph = th.sparse_coo_tensor(
                    th.cat((data.edge_index, data.edge_index[[1,0]]), dim = -1),
                    data.edge_attr.repeat(2),
                    (data.num_nodes, data.num_nodes)
                )
            else:
                *batch, labels = data
                graph = None
                labels = labels[0]
                batch = [_[0] for _ in batch]

            try:
                _ = net.warmup_step(
                    [_.to(config.DEVICE) for _ in batch],
                    labels.to(config.DEVICE),
                    graph.to(config.DEVICE),
                    epoch=0,
                    it=0,
                    n_batches=len(loader),
                    ignore_in_total=ignore_in_total
                )
            except Exception as e:
                print(f"Model warm-up stopped due to exception: {e}")
                return


def train(cfg, net, loader, eval_data = None, graph_bool=False, callbacks=tuple(), track_logs=False):
    """
    Train the model for one run.

    :param cfg: Experiment config
    :type cfg: config.defaults.Experiment
    :param net: Model
    :type net:
    :param loader: DataLoder for training data
    :type loader:  th.utils.data.DataLoader
    :param eval_data: DataLoder for evaluation data
    :type eval_data:  th.utils.data.DataLoader
    :param callbacks: Training callbacks.
    :type callbacks: List
    :return: None
    :rtype: None
    """
    n_batches = len(loader)
    e = 1
    if (cfg.model_config.warmup_epochs is not None) and (cfg.model_config.warmup_epochs > 0):
        warmup(net, loader, cfg.model_config, graph_bool)

    while e <= cfg.n_epochs:
        iter_losses = []
        for i, data in enumerate(loader):
            if graph_bool:
                try:
                    data, ss_labels = data
                except ValueError:
                    ss_labels = None
                # Homogeneous graph data loader
                data_dict = dict(sorted(data.to_dict().items()))
                batch = [v for k, v in data_dict.items() if "view_" in k]
                labels = data.y 
                # Important note! self.graph only considers upper triangle, therefore "manual" fill
                graph = th.sparse_coo_tensor(
                    th.cat((data.edge_index, data.edge_index[[1,0]]), dim = -1),
                    data.edge_attr.repeat(2),
                    (data.num_nodes, data.num_nodes)
                )
            else:
                # Standard data loader
                *batch, labels = data
                graph = None
                batch = [_[0] for _ in batch]
                labels = labels[0]
            try:
                batch_losses = net.train_step(
                    [_.to(config.DEVICE) for _ in batch],
                    labels.to(config.DEVICE),
                    graph.to(config.DEVICE) if graph is not None else None,
                    semi_supervised_labels=ss_labels,
                    epoch=(e-1),
                    it=i,
                    n_batches=n_batches
                )
            except Exception as e:
                print(f"Training stopped due to exception: {e}")
                return

            iter_losses.append(helpers.npy(batch_losses))
        th.cuda.empty_cache()

        logs = evaluate.get_logs(
            cfg,
            net,
            eval_data = eval_data if eval_data is not None else loader,
            iter_losses = iter_losses,
            epoch = e,
            include_params = True,
            graph_bool=graph_bool
        )

        try:
            for cb in callbacks:
                cb.epoch_end(e, logs=logs, net=net)
        except callback.StopTraining as err:
            print(err)
            break

        if track_logs: wandb.log(logs)
        e += 1


def main(name="mvc"):
    """
    Run an experiment.
    """
    experiment_name, cfg, track_logs = config.get_experiment_config()
    if track_logs:
        wandb_config = wandb_init(experiment_name, cfg)
    experiment_identifier = wandb.util.generate_id()
    load_graph = cfg.model_config.__class__ in [config.defaults.GraphCoMVC, config.defaults.GraphMVC]


    if (config.defaults.CNN in [type(__) for __ in cfg.model_config.backbone_configs]) and any(_.pretrained_model is not None for _ in cfg.model_config.backbone_configs if type(_) == config.defaults.CNN):
        # apply downscaling to 244 x 244 if pretrained model
        downscale = True
    else:
        downscale = False

    if cfg.dataset_config.eval_sample_proportion is not None:
        (dataset, graph_bool), eval_dataset = load_dataset(
            **cfg.dataset_config.dict(),
            load_graph = load_graph,
            seed=experiment_identifier,
            downscale_244=downscale
        )
        if not graph_bool:
            eval_data = evaluate.get_eval_data(
                eval_dataset,
                sampler = sampler(eval_dataset, batch_size = int(cfg.batch_size), drop_last = False),
            )
        else:
            if cfg.batch_size == len(dataset.tensors[-1]):
                loader = get_graph_sampler(
                    eval_data,
                    batch_size = int(cfg.batch_size),
                    semi_supervised_N = cfg.model_config.loss_config.n_semi_supervised,
                )
            elif cfg.batch_size < len(dataset.tensors[-1]):
                assert cfg.graphsaint_steps is not None
                loader = get_graph_sampler(
                    eval_data,
                    loader = GraphSampler,
                    num_steps = len(dataset) // int(cfg.batch_size),
                    walk_length = cfg.graphsaint_steps,
                    batch_size = int(cfg.batch_size),
                    semi_supervised_N = cfg.model_config.loss_config.n_semi_supervised,
                )
    else:
        dataset, graph_bool = load_dataset(
            **cfg.dataset_config.dict(),
            load_graph = load_graph,
            seed=experiment_identifier,
            downscale_244=downscale
        )
        eval_data = None

    if not graph_bool:
        loader = th.utils.data.DataLoader(
            dataset,
            sampler = sampler(dataset, batch_size = int(cfg.batch_size), drop_last = False),
            num_workers = 0,
            pin_memory = False
        )
    else:
        # Initialises a heterogeneous graph loader from the torch_geometric library
        if cfg.batch_size == len(dataset.tensors[-1]):
            loader = get_graph_sampler(
                dataset,
                batch_size = int(cfg.batch_size),
                semi_supervised_N = cfg.model_config.loss_config.n_semi_supervised,
            )
        elif cfg.batch_size < len(dataset.tensors[-1]):
            assert cfg.graphsaint_steps is not None
            loader = get_graph_sampler(
                dataset,
                loader = GraphSampler,
                walk_length = cfg.graphsaint_steps,
                num_steps = len(dataset) // int(cfg.batch_size),
                batch_size = int(cfg.batch_size),
                semi_supervised_N = cfg.model_config.loss_config.n_semi_supervised,
            )
        else:
            assert ValueError

    run_logs = []
    run = 0
    while run < cfg.n_runs:
        if track_logs:
            wandb_run = wandb.init(
                project=name,
                entity="erlandg",
                group=experiment_identifier,
                config=wandb_config,
                reinit=True,
                settings=wandb.Settings(start_method="fork"),
            )
        net = build_model(cfg.model_config)
        print(net)
        callbacks = (
            callback.Printer(print_confusion_matrix=(cfg.model_config.cm_config.n_clusters <= 100)),
            callback.ModelSaver(cfg=cfg, experiment_name=experiment_name, identifier=experiment_identifier,
                                run=run, epoch_interval=1, best_loss_term=cfg.best_loss_term,
                                checkpoint_interval=cfg.checkpoint_interval),
            callback.EarlyStopping(patience=cfg.patience, best_loss_term=cfg.best_loss_term, epoch_interval=1)
        )
        try:
            # Updates semi-supervised indices
            loader.update_index
        except AttributeError:
            pass
        tr_ = train(
            cfg,
            net,
            loader,
            eval_data = eval_data,
            graph_bool = graph_bool,
            callbacks = callbacks,
            track_logs = track_logs
        )
        th.cuda.empty_cache()

        run_logs.append(evaluate.eval_run(
            cfg = cfg,
            cfg_name = experiment_name,
            experiment_identifier = experiment_identifier,
            run = run,
            net = net,
            eval_data = eval_data if eval_data is not None else loader,
            callbacks = callbacks,
            load_best = True,
            graph_bool = graph_bool
        ))
        if track_logs: wandb_run.finish()
        if tr_ != 'not valid':
            run += 1


if __name__ == '__main__':
    main("mvc-final-redo-woho")
