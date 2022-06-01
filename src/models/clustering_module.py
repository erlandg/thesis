import torch.nn as nn
from lib.graph_operations import GraphConv, ContrastiveGraphConv, LearnedGraph


class DDC(nn.Module):
    def __init__(self, input_dim, cfg):
        """
        DDC clustering module

        :param input_dim: Shape of inputs.
        :param cfg: DDC config. See `config.defaults.DDC`
        """
        super().__init__()

        if cfg.layer_type == "graph_conv":
            self.learned_graph = [GraphConv(input_dim[0], cfg.n_hidden, bias=True, mask=True, mask_weighting=True, mask_power=2)]
        elif cfg.layer_type == "laplacian":
            self.learned_graph = [
                nn.Linear(input_dim[0], cfg.n_hidden),
                GraphConv(laplacian=True, s=.5, m=2)
            ]
        elif cfg.layer_type == "linear":
            self.learned_graph = [nn.Linear(input_dim[0], cfg.n_hidden)]
            
        hidden_layers = self.learned_graph
        if cfg.use_bn:
            hidden_layers.append(nn.BatchNorm1d(num_features=cfg.n_hidden))
        hidden_layers.append(nn.ReLU())
        self.hidden = nn.ModuleList(hidden_layers)
        self.output = nn.Sequential(nn.Linear(cfg.n_hidden, cfg.n_clusters), nn.Softmax(dim=1))

    def forward(self, x, **kwargs):
        for layer in self.hidden:
            if type(layer) not in (GraphConv, ContrastiveGraphConv):
                x = layer(x)
            else:
                x = layer(x, **kwargs)
        output = self.output(x)
        return output, x
