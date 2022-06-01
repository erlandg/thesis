import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import helpers
from lib.graph_operations import GraphConv


class Backbone(nn.Module):
    LAYER_MAPPINGS = [
        nn.Conv2d,
        nn.MaxPool2d,
        nn.AdaptiveAvgPool2d,
        nn.Linear,
        nn.Flatten,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.ReLU,
        nn.LeakyReLU,
    ]

    def __init__(self):
        """
        Backbone base class
        """
        super().__init__()
        self.layers = nn.ModuleList()

    def forward(self, x, **kwargs):
        for layer in self.layers:
            if type(layer) is not GraphConv:
                x = layer(x)
            else:
                x = layer(x, **kwargs)
        return x

    def get_decoder(self, input_size, **_):
        decoder = []
        act = False # Keep track of activation functions
        for i, (layer, input, output) in enumerate(zip(self.layers[::-1], input_size[::-1][:-1], input_size[::-1][1:])):
            type_ = type(layer)
            if type_ is nn.Conv2d:
                dim_out = helpers.convtranspose2d_output_shape(
                    input[1:],
                    kernel_size = layer.kernel_size,
                    stride = layer.stride,
                    pad = layer.padding,
                    dilation = layer.dilation,
                )
                diff = [i - j for i, j in zip(output[1:], dim_out)]
                decoder.append(nn.ConvTranspose2d(
                    in_channels = input[0],
                    out_channels = output[0],
                    kernel_size = layer.kernel_size,
                    stride = layer.stride,
                    padding = layer.padding,
                    dilation = layer.dilation,
                ))
                if any(diff):
                    decoder.append(nn.Upsample(size = output[1:], mode="bilinear", align_corners=False))
                if i != (len(self.layers)-1):
                    decoder.append(nn.BatchNorm2d(num_features=output[0]))
                    decoder.append(nn.ReLU())
                else: decoder.append(nn.Sigmoid())
            elif type_ is nn.MaxPool2d:
                decoder.append(nn.Upsample(size = output[1:], mode="bilinear", align_corners=False))
            elif type_ is nn.Linear:
                assert (len(input) == 1) and (len(output) == 1), "Input and output should be dimension 1"
                decoder.append(nn.Linear(in_features = input[0], out_features = output[0]))
                if i != (len(self.layers)-1):
                    decoder.append(nn.BatchNorm1d(num_features=output[0]))
                    decoder.append(nn.ReLU())
            elif type_ is GraphConv:
                decoder.append(GraphConv(in_features = input[0], out_features = output[0]))
                if act:
                    decoder.append(nn.ReLU())
                    act = False
            elif type_ is nn.Flatten:
                if len(output) > 1:
                    decoder.append(nn.Unflatten(dim = -1, unflattened_size = output))
            elif type_ is nn.BatchNorm1d:
                continue
            elif type_ is nn.BatchNorm2d:
                continue
            elif type_ is nn.ReLU:
                act = True
                continue
            elif type_ is nn.LeakyReLU:
                act = True
                continue
        return nn.ModuleList(decoder)


class CNN(Backbone):
    def __init__(self, cfg, flatten_output=True, **kwargs):
        """
        CNN backbone

        :param cfg: CNN config
        :type cfg: config.defaults.CNN
        :param flatten_output: Flatten the backbone output?
        :type flatten_output: bool
        :param _:
        :type _:
        """
        super().__init__()

        self.output_size = [list(cfg.input_size)]

        for layer_type, *layer_params in cfg.layers:
            if layer_type == "conv":
                params = {
                    'in_channels' : self.output_size[-1][0],
                    'out_channels' : layer_params[2],
                    'kernel_size' : layer_params[:2]
                }
                if len(layer_params) > 4:
                    for key_param in layer_params[4:]:
                        assert len(key_param) == 2, "Additional Conv parameters must be in tuples of size 2, e.g. ('stride', 1), ('pad', 1), ..."
                        params[key_param[0].lower()] = key_param[1]

                self.layers.append(nn.Conv2d(**params))
                # Update output size
                self.output_size.append([layer_params[2]])
                self.output_size[-1].extend(helpers.conv2d_output_shape(
                    self.output_size[-2][1:],
                    kernel_size = params["kernel_size"],
                    stride = params['stride'] if 'stride' in params else 1,
                    pad = params['padding'] if 'padding' in params else 0,
                    dilation = params['dilation'] if 'dilation' in params else 1
                ))
                # Add activation
                if layer_params[3] == "relu":
                    self.layers.append(nn.ReLU())
                    self.output_size.append(self.output_size[-1])

            elif layer_type == "graph":
                if len(self.output_size[-1]) > 1:
                    self.layers.append(nn.Flatten())
                    self.output_size.append([np.prod(self.output_size[-1])])
                self.layers.append(GraphConv(in_features=self.output_size[-1][0], out_features=layer_params[0]))
                self.output_size.append([layer_params[0]])

            elif layer_type == "pool":
                self.layers.append(nn.MaxPool2d(kernel_size=layer_params))
                # Update output size
                self.output_size.append([self.output_size[-1][0]])
                self.output_size[-1].extend(helpers.conv2d_output_shape(self.output_size[-2][1:], kernel_size=layer_params,
                                                                   stride=layer_params))

            elif layer_type == "relu":
                self.layers.append(nn.ReLU())
                self.output_size.append(self.output_size[-1])

            elif layer_type == "lrelu":
                self.layers.append(nn.LeakyReLU(layer_params[0]))
                self.output_size.append(self.output_size[-1])

            elif layer_type == "bn":
                if len(self.output_size[-1]) > 1:
                    self.layers.append(nn.BatchNorm2d(num_features=self.output_size[-1][0]))
                else:
                    self.layers.append(nn.BatchNorm1d(num_features=self.output_size[-1][0]))
                self.output_size.append(self.output_size[-1])

            elif layer_type == "fc":
                if len(self.output_size[-1]) > 1:
                    self.layers.append(nn.Flatten())
                    self.output_size.append([np.prod(self.output_size[-1])])
                self.layers.append(nn.Linear(self.output_size[-1][0], layer_params[0], bias=True))
                self.output_size.append([layer_params[0]])

            else:
                raise RuntimeError(f"Unknown layer type: {layer_type}")

        if flatten_output and (len(self.output_size[-1]) > 1):
            self.layers.append(nn.Flatten())
            self.output_size.append([np.prod(self.output_size[-1])])


class MLP(Backbone):
    def __init__(self, cfg, input_size=None, flatten_output=None, **_):
        """
        MLP backbone

        :param cfg: MLP config
        :type cfg: config.defaults.MLP
        :param input_size: Optional input size which overrides the one set in `cfg`.
        :type input_size: Optional[Union[List, Tuple]]
        :param _:
        :type _:
        """
        super().__init__()
        self.output_size = self.create_linear_layers(cfg, self.layers, input_size=input_size)

    @staticmethod
    def get_activation_module(a):
        if a == "relu":
            return nn.ReLU()
        elif a == "sigmoid":
            return nn.Sigmoid()
        elif a == "tanh":
            return nn.Tanh()
        elif a == "softmax":
            return nn.Softmax(dim=1)
        elif a.startswith("leaky_relu"):
            neg_slope = float(a.split(":")[1])
            return nn.LeakyReLU(neg_slope)
        else:
            raise RuntimeError(f"Invalid MLP activation: {a}.")

    @classmethod
    def create_linear_layers(cls, cfg, layer_container, input_size=None):
        # `input_size` takes priority over `cfg.input_size`
        if input_size is not None:
            output_size = [list(input_size)]
        else:
            output_size = [list(cfg.input_size)]

        if len(output_size[0]) > 1:
            layer_container.append(nn.Flatten())
            output_size.append([np.prod(output_size[0])])

        n_layers = len(cfg.layers)
        activations = helpers.ensure_iterable(cfg.activation, expected_length=n_layers)
        use_bias = helpers.ensure_iterable(cfg.use_bias, expected_length=n_layers)
        use_bn = helpers.ensure_iterable(cfg.use_bn, expected_length=n_layers)

        for i, (n_units, act, _use_bias, _use_bn) in enumerate(zip(cfg.layers, activations, use_bias, use_bn)):
            # If we get n_units = -1, then the number of units should be the same as the previous number of units, or
            # the input dim.
            if n_units == -1:
                n_units = output_size[-1][0]
            elif n_units == "graph":
                n_units = output_size[-1][0]
                layer_container.append(GraphConv(in_features=output_size[-1][0], out_features=n_units))
            else:
                layer_container.append(nn.Linear(in_features=output_size[-1][0], out_features=n_units, bias=_use_bias))
            output_size.append([n_units])

            # if i != len(cfg.layers) - 1:
            if _use_bn:
                # Add BN before activation
                layer_container.append(nn.BatchNorm1d(num_features=n_units))
                output_size.append([n_units])
            if act is not None:
                # Add activation
                layer_container.append(cls.get_activation_module(act))
                output_size.append([n_units])
            # else:
            #     if act is not None:
            #         # Add activation
            #         layer_container.append(cls.get_activation_module('sigmoid'))
            #         output_size.append([n_units])

        return output_size


class GraphAttention(Backbone):
    def __init__(self, cfg, input_size=None, flatten_output=None, **_):
        """
        GraphAttention backbone

        :param cfg: GraphAttention config
        :type cfg: config.defaults.GraphAttention
        :param input_size: Optional input size which overrides the one set in `cfg`.
        :type input_size: Optional[Union[List, Tuple]]
        :param _:
        :type _:
        """
        super().__init__()
        self.output_size = self.create_linear_layers(cfg, self.layers, input_size=input_size)
        self.skip_connection = cfg.skip_connection

    @staticmethod
    def get_activation_module(a):
        if a == "relu":
            return nn.ReLU()
        elif a == "sigmoid":
            return nn.Sigmoid()
        elif a == "tanh":
            return nn.Tanh()
        elif a == "softmax":
            return nn.Softmax(dim=1)
        elif a.startswith("leaky_relu"):
            neg_slope = float(a.split(":")[1])
            return nn.LeakyReLU(neg_slope)
        else:
            raise RuntimeError(f"Invalid Graph Attention activation: {a}.")

    @classmethod
    def create_linear_layers(cls, cfg, layer_container, input_size=None):
        # `input_size` takes priority over `cfg.input_size`
        if input_size is not None:
            output_size = [list(input_size)]
        else:
            output_size = [list(cfg.input_size)]

        if len(output_size[0]) > 1:
            layer_container.append(nn.Flatten())
            output_size.append([np.prod(output_size[0])])

        n_layers = len(cfg.layers)
        activations = helpers.ensure_iterable(cfg.activation, expected_length=n_layers)
        use_bias = helpers.ensure_iterable(cfg.use_bias, expected_length=n_layers)
        use_bn = helpers.ensure_iterable(cfg.use_bn, expected_length=n_layers)

        for i, (n_units, act, _use_bias, _use_bn) in enumerate(zip(cfg.layers, activations, use_bias, use_bn)):
            # If we get n_units = -1, then the number of units should be the same as the previous number of units, or
            # the input dim.
            if n_units == -1:
                n_units = output_size[-1][0]

            layer_container.append(GraphConv(
                in_features=output_size[-1][0],
                out_features=n_units,
                bias = _use_bias,
                attention_features = cfg.attention_features,
                mask = cfg.mask,
                mask_weighting = cfg.mask_weighting,
                mask_power = cfg.mask_power,
                dropout = cfg.dropout
            ))
            output_size.append([n_units])

            if (i != len(cfg.layers) - 1):
                if _use_bn:
                    # Add BN before activation
                    layer_container.append(nn.BatchNorm1d(num_features=n_units))
                    output_size.append([n_units])
                if act is not None:
                    # Add activation
                    layer_container.append(cls.get_activation_module(act))
                    output_size.append([n_units])

        return output_size

    def forward(self, x, **kwargs):
        C = []
        for layer in self.layers:
            if type(layer) == GraphConv:
                if not self.skip_connection:
                    x, c_ = layer(x, **kwargs)
                else:
                    x2, c_ = layer(x, **kwargs)
                    if x2.shape == x.shape:
                        x = x + x2
                    else:
                        x = x2
                C.append(c_)
            else:
                x = layer(x)
        return x, C

    def backward(self, x, C, reverse_order_C = True, **kwargs):
        if reverse_order_C: C = C[::-1]
        act = None
        bn = None
        c_i = 0
        for i, layer in enumerate(self.layers[::-1]):
            if type(layer) == GraphConv:
                x = layer(x, graph = None, C = C[c_i], backward = True, **kwargs)
                # if res.shape == x.shape:
                #     x = x + res
                # else:
                #     x = res
                if i != len(self.layers) - 1:
                    if act is not None:
                        x = act(x)
                        act = None
                c_i += 1
            elif type(layer) == nn.ReLU:
                act = F.relu
            elif type(layer) == nn.LeakyReLU:
                act = F.leaky_relu
            else:
                x = layer(x)
        return x


class Pretrained_CNN(Backbone):

    def __init__(self, cfg, input_size=None, flatten_output=None, **_):
        super().__init__()
        assert cfg.input_size == (3, 244,244)
        self.net = self.load_pretrained_cnn(cfg.pretrained_model, cfg.pretrained_features_out, pretrained = True)
        self.layers = nn.ModuleList(self.net.modules())
        self.output_size = [[cfg.pretrained_features_out]]

    def forward(self, x, **kwargs):
        return self.net(x)

    @staticmethod
    def load_pretrained_cnn(model_str, features_out, pretrained=False):
            if model_str == "resnet18":
                net = th.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=pretrained)
                net.fc = nn.Linear(net.fc.in_features, features_out)
                nn.init.kaiming_normal_(net.fc.weight)
                net.fc.bias.data.fill_(0.01)
            elif model_str == "vgg16":
                net = th.hub.load("pytorch/vision:v0.10.0", "vgg16_bn", pretrained=pretrained)
                net.classifier[6] = nn.Linear(net.classifier[6].in_features, features_out)
                nn.init.kaiming_normal_(net.classifier[6].weight)
                net.classifier[6].bias.data.fill_(0.01)
            elif model_str == "alexnet":
                net = th.hub.load("pytorch/vision:v0.10.0", "alexnet", pretrained=pretrained)
                net.classifier[6] = nn.Linear(net.classifier[6].in_features, features_out)
                nn.init.kaiming_normal_(net.classifier[6].weight)
                net.classifier[6].bias.data.fill_(0.01)
            return net


class Backbones(nn.Module):
    BACKBONE_CONSTRUCTORS = {
        "CNN": CNN,
        "MLP": MLP,
    }

    def __init__(self, backbone_configs, flatten_output=True, reconstruction=False):
        """
        Class representing multiple backbones. Call with list of inputs, where inputs[0] goes into the first backbone,
        and so on.

        :param backbone_configs: List of backbone configs. Each element corresponds to a backbone.
        :type backbone_configs: List[Union[config.defaults.MLP, config.defaults.CNN], ...]
        :param flatten_output: Flatten the backbone outputs?
        :type flatten_output: bool
        """
        super().__init__()

        self.reconstruction = reconstruction
        self.backbones = nn.ModuleList()
        for cfg in backbone_configs:
            if (cfg.class_name == "CNN") and (cfg.pretrained_model is not None):
                bb_ = Pretrained_CNN(cfg, flatten_output=flatten_output)
            else:
                bb_ = self.create_backbone(cfg, flatten_output=flatten_output)
            self.backbones.append(bb_)
        if self.reconstruction:
            self.decoder_backbones = nn.ModuleList()
            for backbone in self.backbones:
                self.decoder_backbones.append(backbone.get_decoder(input_size = backbone.output_size))

    @property
    def output_sizes(self):
        return [bb.output_size[-1] for bb in self.backbones]

    @classmethod
    def create_backbone(cls, cfg, flatten_output=True):
        if cfg.class_name not in cls.BACKBONE_CONSTRUCTORS:
            raise RuntimeError(f"Invalid backbone: '{cfg.class_name}'")
        return cls.BACKBONE_CONSTRUCTORS[cfg.class_name](
            cfg,
            flatten_output=flatten_output
        )

    def forward(self, views, **kwargs):
        assert len(views) == len(self.backbones), f"n_views ({len(views)}) != n_backbones ({len(self.backbones)})."
        outputs = [bb(v, **kwargs) for bb, v in zip(self.backbones, views)]
        return outputs

    def decode(self, compressed_views):
        assert len(compressed_views) == len(self.backbones), f"n_views ({len(compressed_views)}) != n_backbones ({len(self.backbones)})."
        outputs = []
        for bb, v in zip(self.decoder_backbones, compressed_views):
            for layer in bb:
                v = layer(v)
            outputs.append(v)
        return outputs


class Graph_Backbones(nn.Module):

    def __init__(self, backbone_configs, flatten_output=True):
        """
        Class representing multiple backbones. Call with list of inputs, where inputs[0] goes into the first backbone,
        and so on.

        :param backbone_configs: List of backbone configs. Each element corresponds to a backbone.
        :type backbone_configs: List[Union[config.defaults.MLP, config.defaults.CNN], ...]
        :param flatten_output: Flatten the backbone outputs?
        :type flatten_output: bool
        """
        super().__init__()
        self.backbones = nn.ModuleList()
        # self.decoder_backbones = nn.ModuleList()
        if len(backbone_configs) != 1:
            self.shared_weights = False
        else:
            self.shared_weights = True
        for cfg in backbone_configs:
            self.backbones.append(GraphAttention(
                cfg,
                flatten_output=flatten_output
            ))
        # for backbone in self.backbones:
        #     self.decoder_backbones.append(backbone.get_decoder(input_size = backbone.output_size))

    @property
    def output_sizes(self):
        return [bb.output_size[-1] for bb in self.backbones]

    @staticmethod
    def graph_reconstruction(H, sparse = False):
        return th.matmul(H, H.T)

    def forward(self, views, graph):
        X = []
        self.C = [[] for _ in range(len(views))]
        if not self.shared_weights:
            for i, (bb, v) in enumerate(zip(self.backbones, views)):
                x_, c_ = bb(v, graph = graph, return_attention=True)
                X.append(x_)
                self.C[i] = c_
        else:
            for i, v in enumerate(views):
                x_, c_ = self.backbones[0](v, graph = graph, return_attention=True)
                X.append(x_)
                self.C[i] = c_
        return X

    def decode(self, compressed_views, sparse = False):
        outputs = []
        graphs = []
        if not self.shared_weights:
            for i, (bb, v) in enumerate(zip(self.backbones, compressed_views)):
                graphs.append(self.graph_reconstruction(v, sparse = sparse))
                v = bb.backward(v, C = self.C[i], reverse_order_C = True)
                outputs.append(v)
        else:
            for i, v in enumerate(compressed_views):
                graphs.append(self.graph_reconstruction(v, sparse = sparse))
                v = self.backbones[0].backward(v, C = self.C[i], reverse_order_C = True)
                outputs.append(v)
        return outputs, graphs
