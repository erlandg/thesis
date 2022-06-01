import argparse
from ast import parse
import pickle
import yaml

from .constants import *
from .config import Config
from . import defaults, experiments
from .eamc import experiments as eamc_experiments
from .eamc import defaults as eamc_defaults


def parse_config_name_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config_name", required=True)
    parser.add_argument("-l", "--logs", dest="track_logs", action="store_true")
    args = parser.parse_known_args()[0]
    return args.config_name, args.track_logs


def set_cfg_value(cfg, key_list, value):
    sub_cfg = cfg
    for key in key_list[:-1]:
        sub_cfg = getattr(sub_cfg, key)
    setattr(sub_cfg, key_list[-1], value)


def get_yaml(filepath):
    with open(filepath, "r") as f:
        yamlfile = yaml.safe_load(f)
    return yamlfile


def update_cfg(cfg):
    sep = "__"
    parser = argparse.ArgumentParser()
    cfg_dict = hparams_dict(cfg, sep=sep)

    parser.add_argument("-c", "--config", dest="config_name")
    parser.add_argument("-l", "--logs", dest="track_logs", action="store_true")

    for key, value in cfg_dict.items():
        args_ = ["--" + key]
        split_key = key.split("__")
        if (len(split_key) > 1) and (sum([split_key[-1] == _.split("__")[-1] for _ in cfg_dict.keys()]) == 1):
            # If there is a single occurrence, allow for only use of name. For use in W&B sweeps.
            # e.g. --shared_weights rather than --model_config__shared_weights
            args_.append("--" + split_key[-1])

        value_type = type(value) if isinstance(value, (int, float, bool)) else None
        parser.add_argument(*args_, dest=key, default=value, type=value_type)

    parser.add_argument(
        '--graph_layers',
        dest="model_config__graph_attention_configs__layers"
    )
    args = parser.parse_args()
    if args.model_config__graph_attention_configs__layers is not None:
        args.model_config__graph_attention_configs__layers = [int(_) for _ in args.model_config__graph_attention_configs__layers.split(" ")]
    if args.model_config__loss_config__funcs != cfg_dict["model_config__loss_config__funcs"]:
        if ", " in args.model_config__loss_config__funcs:
            func_sep = ", "
        elif " " in args.model_config__loss_config__funcs:
            func_sep = " "
        args.model_config__loss_config__funcs = "|".join(args.model_config__loss_config__funcs.split(func_sep))

    for key in cfg_dict.keys():
        key_list = key.split(sep)
        value = getattr(args, key)
        set_cfg_value(cfg, key_list, value)


def get_config_by_name(name):
    try:
        if name.startswith("eamc"):
            cfg = getattr(eamc_experiments, name)
        else:
            cfg = getattr(experiments, name)
    except Exception as err:
        raise RuntimeError(f"Config not found: {name}") from err
    cfg.model_config.loss_config.n_clusters = cfg.model_config.cm_config.n_clusters
    return cfg


def get_config_from_file(name=None, tag=None, file_path=None, run=0):
    if file_path is None:
        file_path = MODELS_DIR / f"{name}-{tag}" / f"run-{run}" / "config.pkl"
    with open(file_path, "rb") as f:
        cfg = pickle.load(f)
    return cfg


def get_experiment_config():
    name, track_logs = parse_config_name_arg()
    cfg = get_config_by_name(name)
    update_cfg(cfg)
    return name, cfg, track_logs


def _insert_hparams(cfg_dict, hp_dict, key_prefix, skip_keys, sep="/"):
    hparam_types = (str, int, float, bool)
    for key, value in cfg_dict.items():
        if key in skip_keys:
            continue
        _key = f"{key_prefix}{sep}{key}" if key_prefix else key
        if isinstance(value, hparam_types) or value is None:
            hp_dict[_key] = value
        elif isinstance(value, dict):
            _insert_hparams(value, hp_dict, _key, skip_keys, sep=sep)


def hparams_dict(cfg, sep="/"):
    skip_keys = []
    hp_dict = {}
    _insert_hparams(cfg.dict(), hp_dict, "", skip_keys, sep=sep)
    return hp_dict
