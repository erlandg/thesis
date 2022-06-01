from typing import Tuple, List, Union, Optional
from typing_extensions import Literal

from config import Config


class Dataset(Config):
    # Name of the dataset. Must correspond to a filename in data/processed/
    name: str
    # Number of samples to load. Set to None to load all samples
    n_samples: int = None
    # Proportion of samples to use for evaluation. Set to None to use all samples in the dataset.
    eval_sample_proportion: float = None
    # Subset of views to load. Set to None to load all views
    select_views: Tuple[int, ...] = None
    # Subset of labels (classes) to load. Set to None to load all classes
    select_labels: Tuple[int, ...] = None
    # Number of samples to load for each class. Set to None to load all samples
    label_counts: Tuple[int, ...] = None
    # Standard deviation of noise added to the views `noise_views`.
    noise_sd: float = None
    # Minimum value of graph.
    graph_tol: float = None
    # Subset of views to add noise to
    noise_views: Tuple[int, ...] = None
    normalise_images: bool = True


class Loss(Config):
    # Number of clusters
    n_clusters: int = None
    # Terms to use in the loss, separated by '|'. E.g. "ddc_1|ddc_2|ddc_3|" for the DDC clustering loss
    funcs: str
    # Use labelled data for semi-supervised loss
    n_semi_supervised: int = None
    semi_supervised_weight: float = 1.
    # Optional weights for the loss terms. Set to None to have all weights equal to 1.
    weights: Tuple[Union[float, int], ...] = None
    # Multiplication factor for the sigma hyperparameter
    rel_sigma = 0.15
    # Tau hyperparameter
    tau = 0.1
    # Delta hyperparameter
    delta = 0.1
    # Epsilon hyperparameter for reconstruction loss
    epsilon_features = 0.1
    epsilon_structure = 0.1
    # Fraction of batch size to use as the number of negative samples in the contrastive loss. Set to -1 to use all
    # pairs (except the positive) as negative pairs.
    negative_samples_ratio: float = 0.25
    # Similarity function for the contrastive loss. "cos" (default) and "gauss" are supported.
    contrastive_similarity: Literal["cos", "gauss"] = "cos"
    # Enable the adaptive contrastive weighting?
    adaptive_contrastive_weight = True


class Optimizer(Config):
    # Base learning rate
    learning_rate: float = 0.001
    # Max gradient norm for gradient clipping.
    clip_norm: float = 5.0
    # Step size for the learning rate scheduler. None disables the scheduler.
    scheduler_step_size: int = None
    # Multiplication factor for the learning rate scheduler
    scheduler_gamma: float = 0.1


class DDC(Config):
    # Number of clusters
    n_clusters: int = None
    # Number of units in the first fully connected layer
    n_hidden = 100
    layer_type: Literal["graph_conv", "linear"] = "linear"
    # Use batch norm after the first fully connected layer?
    use_bn = True


class CNN(Config):
    pretrained_model: str = None
    pretrained_features_out: int = 512
    # Shape of the input image. Format: CHW
    input_size: Tuple[int, ...] = None
    # Network layers
    layers = (
        ("conv", 5, 5, 32, "relu"),
        ("conv", 5, 5, 32, None),
        ("bn",),
        ("relu",),
        ("pool", 2, 2),
        ("conv", 3, 3, 32, "relu"),
        ("conv", 3, 3, 32, None),
        ("bn",),
        ("relu",),
        ("pool", 2, 2),
    )


class MLP(Config):
    # Shape of the input
    input_size: Tuple[int, ...] = None
    # Units in the network layers
    layers: Tuple[Union[int, str], ...] = (512, 512, 256)
    # Activation function. Can be a single string specifying the activation function for all layers, or a list/tuple of
    # string specifying the activation function for each layer.
    activation: Union[str, None, List[Union[None, str]], Tuple[Union[None, str], ...]] = "relu"
    # Include bias parameters? A single bool for all layers, or a list/tuple of booleans for individual layers.
    use_bias: Union[bool, Tuple[bool, ...]] = True
    # Include batch norm after layers? A single bool for all layers, or a list/tuple of booleans for individual layers.
    use_bn: Union[bool, Tuple[bool, ...]] = False


class GraphAttention(Config):
    graph_format: Literal["sparse", "dense"] = "dense"
    # Shape of the input. Defaults to number of features in embedded space.
    input_size: Tuple[int, ...] = None
    # Units in the network layers
    layers: Tuple[Union[int, str], ...] = (256,)
    # Activation function. Can be a single string specifying the activation function for all layers, or a list/tuple of
    # string specifying the activation function for each layer.
    activation: Union[str, None, List[Union[None, str]], Tuple[Union[None, str], ...]] = None
    # Include bias parameters? A single bool for all layers, or a list/tuple of booleans for individual layers.
    use_bias: Union[bool, Tuple[bool, ...]] = False
    # Include batch norm after layers? A single bool for all layers, or a list/tuple of booleans for individual layers.
    use_bn: Union[bool, Tuple[bool, ...]] = False
    # For laplacian graph conv
    laplacian_s: float = .5
    # For laplacian graph conv
    laplacian_m: int = 2
    skip_connection: bool = False
    attention_features: int = 1
    mask: bool = False
    mask_weighting: bool = False
    mask_power: int = 1
    dropout: float = None


class Fusion(Config):
    # Fusion method. "mean" constant weights = 1/V. "weighted_mean": Weighted average with learned weights.
    method: Literal["mean", "weighted_mean"]
    # Number of views in the dataset
    n_views: int


class DDCModel(Config):
    # Encoder network config
    backbone_config: Union[MLP, CNN]
    # Whether to share graph weights.
    shared_weights: bool = None
    # Clustering module config
    cm_config: Union[DDC]
    # Loss function config
    loss_config: Loss
    # Optimizer config
    optimizer_config = Optimizer()
    warmup_epochs: int = None
    warmup_funcs: str = "ddc_1"
    warmup_optimizer: Optimizer = None


class SiMVC(Config):
    # Tuple of encoder configs. One for each modality.
    backbone_configs: Tuple[Union[MLP, CNN], ...]
    # Whether to share graph weights.
    shared_weights: bool = None
    # Fusion module config.
    fusion_config: Fusion
    # Clustering module config.
    cm_config: Union[DDC]
    # Loss function config
    loss_config: Loss
    # Optimizer config
    optimizer_config = Optimizer()
    warmup_epochs: int = None
    warmup_funcs: str = "ddc_1"
    warmup_optimizer: Optimizer = None


class CoMVC(Config):
    # Tuple of encoder configs. One for each modality.
    backbone_configs: Tuple[Union[MLP, CNN], ...]
    # Whether to share graph weights.
    shared_weights: bool = None
    # Projection head config. Set to None to remove the projection head.
    projector_config: Optional[MLP]
    # Fusion module config.
    fusion_config: Fusion
    # Clustering module config.
    cm_config: Union[DDC]
    # Loss function config
    loss_config: Loss
    # Optimizer config
    optimizer_config = Optimizer()
    warmup_epochs: int = None
    warmup_funcs: str = "contrast"
    warmup_optimizer: Optimizer = None


class GraphMVC(Config):
    # Tuple of encoder configs. One for each modality.
    backbone_configs: Tuple[Union[MLP, CNN], ...]
    # Tuple of graph attention layers. One for each modality.
    graph_attention_configs: GraphAttention
    # Whether to share graph weights.
    shared_weights: bool = True
    # Projection head config. Set to None to remove the projection head.
    projector_config: Optional[MLP]
    # Fusion module config.
    fusion_config: Fusion
    # Clustering module config.
    cm_config: Union[DDC]
    # Loss function config
    loss_config: Loss
    # Optimizer config
    optimizer_config = Optimizer()
    warmup_epochs: int = None
    warmup_funcs: str = "ddc_1"
    warmup_optimizer: Optimizer = None


class GraphCoMVC(Config):
    # Tuple of encoder configs. One for each modality.
    backbone_configs: Tuple[Union[MLP, CNN], ...]
    # Tuple of graph attention layers. One for each modality.
    graph_attention_configs: GraphAttention
    # Whether to share graph weights.
    shared_weights: bool = True
    # Projection head config. Set to None to remove the projection head.
    projector_config: Optional[MLP]
    # Fusion module config.
    fusion_config: Fusion
    # Clustering module config.
    cm_config: Union[DDC]
    # Loss function config
    loss_config: Loss
    # Optimizer config
    optimizer_config = Optimizer()
    warmup_epochs: int = None
    warmup_funcs: str = "contrast"
    warmup_optimizer: Optimizer = None


class Experiment(Config):
    # Dataset config
    dataset_config: Dataset
    # Model config
    model_config: Union[GraphCoMVC, GraphMVC, CoMVC, SiMVC, DDC]
    # Number of training runs
    n_runs = 20
    # Number of training epochs
    n_epochs = 100
    # Batch size
    batch_size = 100
    # Number of epochs between model evaluation.
    eval_interval: int = 4
    # Number of epochs between model checkpoints.
    checkpoint_interval = 20
    # Patience for early stopping.
    patience = 50000
    # Term in loss function to use for model selection. Set to "tot" to use the sum of all terms.
    best_loss_term = "ddc_1"
    graphsaint_steps: int = None
