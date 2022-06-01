from sklearn.decomposition import SparsePCA
import torch as th
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors

import config
import helpers
from lib.loss import Loss
from lib.optimizer import Optimizer
from lib.backbones import Backbones, Graph_Backbones, MLP
from lib.fusion import get_fusion_module
from lib.graph_operations import GraphConv, LearnedGraph, normalise_graph
from models.clustering_module import DDC
from models.model_base import ModelBase


class GraphCoMVC(ModelBase):
    def __init__(self, cfg, **kwargs):
        """
        Implementation of the GraphCoMVC model.

        :param cfg: Model config. See `config.defaults.GraphCoMVC` for documentation on the config object.
        """
        super().__init__()

        self.cfg = cfg
        self.output = self.hidden = self.fused = self.backbone_outputs = self.projections = None
        self.reconstruction = any(["reconstruction" in _ for _ in cfg.loss_config.funcs.split("|")])
        self.reconstruction_features = any(["reconstruction_feature" in _ for _ in cfg.loss_config.funcs.split("|")])

        # Define Backbones and Fusion modules
        self.backbones = Backbones(cfg.backbone_configs, reconstruction = self.reconstruction_features)
        bb_sizes = self.backbones.output_sizes
        assert all([bb_sizes[0] == s for s in bb_sizes]), f"GraphCoMVC requires all backbones to have the same " \
                                                          f"output size. Got: {bb_sizes}"

        if cfg.graph_attention_configs.input_size is None:
            # Set graph attention input shape to backbone output.
            setattr(cfg.graph_attention_configs, "input_size", bb_sizes[0])
        if cfg.shared_weights:
            self.graph_backbones = Graph_Backbones([cfg.graph_attention_configs])
        else:
            self.graph_backbones = Graph_Backbones([cfg.graph_attention_configs for _ in range(len(cfg.backbone_configs))])
        self.fusion = get_fusion_module(cfg.fusion_config, self.graph_backbones.output_sizes)

        if cfg.projector_config is None:
            self.projector = nn.Identity()
            # self.projector = GraphConv(
            #     laplacian=True,
            #     s = cfg.graph_attention_configs.laplacian_s,
            #     m = cfg.graph_attention_configs.laplacian_m
            # )
        else:
            self.projector = MLP(cfg.projector_config, input_size=bb_sizes[0])

        # Define clustering module
        self.ddc = DDC(input_dim=self.fusion.output_size, cfg=cfg.cm_config)
        # Init learned graph
        self.nn = NearestNeighbors(n_neighbors=11, algorithm='auto')

        # Define loss-module
        self.loss = Loss(cfg=cfg.loss_config)
        # Initialize weights.
        self.apply(helpers.he_init_weights)
        # Instantiate optimizer
        self.optimizer = Optimizer(cfg.optimizer_config, self.parameters())
        self.warmup_optimizer = Optimizer(cfg.warmup_optimizer, self.parameters())

    def forward(self, views, graph=None, **kwargs):
        N = views[0].shape[0]
        self.input = views
        # self.graph = graph

        if (self.cfg.graph_attention_configs.graph_format == "dense") and graph.is_sparse:
            self.graph = graph.to_dense()
        else:
            self.graph = graph
        
        self.pre_graph_backbones = self.backbones(views)
        self.backbone_outputs = self.graph_backbones(
            self.pre_graph_backbones,
            graph = self.graph
        )
        if self.reconstruction:
            self.embedding_reconstructed, self.G_reconstructed = self.graph_backbones.decode(
                self.backbone_outputs,
                sparse = self.graph.is_sparse
            )
            if self.reconstruction_features:
                self.X_reconstructed = self.backbones.decode(self.embedding_reconstructed)
        self.fused = self.fusion(self.backbone_outputs)

        self.projections = self.projector(th.cat(self.backbone_outputs, dim=0))
        # self.projections = th.cat([self.projector(bb, graph = self.graph) for bb in self.pre_graph_backbones], dim=0)

        # x = self.fused
        # for layer in self.hidden_layers: x = layer(x)

        self.output, self.hidden = self.ddc(self.fused, graph = self.graph)
        return self.output



def plot_embeddings(fusion, backbone_outputs, labels):
    from sklearn.manifold import SpectralEmbedding
    import matplotlib.pyplot as plt
    lap = SpectralEmbedding(n_components=2)
    a = lap.fit_transform(fusion(backbone_outputs).detach().cpu()).T
    for label in th.unique(labels.cpu()):
        plt.scatter(*a[:,labels.cpu() == label], label=f"Label {int(label)}")
    plt.legend()
    plt.savefig(config.PROJECT_ROOT / "test.png", bbox_inches="tight")
