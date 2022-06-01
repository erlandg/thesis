from sklearn import semi_supervised
from config.defaults import (
    Experiment,
    CNN,
    MLP,
    GraphAttention,
    DDC,
    Fusion,
    Loss,
    Dataset,
    Optimizer,
    GraphMVC,
    GraphCoMVC,
)


mimic_graph = Experiment(
    dataset_config=Dataset(name="graph_dataset"),
    model_config=GraphMVC(
        backbone_configs=(
            CNN(
                input_size = (1, 256, 256),
                layers = [
                    ("conv", 11, 11, 16, None, ('stride', 4)),
                    ("bn",),
                    ("relu",),
                    ("pool", 2, 2),
                    ("conv", 5, 5, 32, "relu", ('padding', 2), ('stride', 2)),
                    ("conv", 5, 5, 32, None, ('padding', 2), ('stride', 2)),
                    ("bn",),
                    ("relu",),
                    ("pool", 2, 2),
                    ("conv", 3, 3, 64, "relu", ('padding', 1)),
                    ("conv", 3, 3, 64, "relu", ('padding', 1)),
                    ("conv", 3, 3, 64, None, ('padding', 1)),
                    ("bn",),
                    ("relu",),
                    ("fc", 256),
                ],
            ),
            MLP(
                input_size = (486,),
                layers = [512, 256],
                use_bn = True
            ),
            MLP(
                input_size = (240,),
                layers = [512, 256],
                use_bn = True
            ),
        ),
        graph_attention_configs = GraphAttention(
            layers = [128],
        ),
        projector_config=None,
        fusion_config=Fusion(method="weighted_mean", n_views=3),
        cm_config=DDC(n_clusters=2),
        loss_config=Loss(
            funcs="ddc_1|ddc_2|ddc_3|reconstruction",
            # Additional loss parameters go here
            epsilon_features=1,
            epsilon_structure=1,
        ),
        optimizer_config=Optimizer(
            learning_rate=1e-4,
            # Additional optimizer parameters go here
        ),
        shared_weights = True,
    ),
    n_epochs=200,
    n_runs=7,
    batch_size=100
)

mimic_graph_contrast = Experiment(
    dataset_config = Dataset(
        # knn_edema_dataset 3520 x (1,256,256), (66,), (210,) - 2 classes
        # mimic_sample 1747 x (1,256,256), (66,), (210,) - 2 classes
        # knn_edema_dataset_no_images
        # knn_edema_dataset_image_vital
        # knn_edema_dataset_image_lab

        # cora 2708 x (1433,), (2708,) - 7 classes
        # balanced_cora 1260 x (1433,), (1260,) - 7 classes

        # citeseer 3327 x (3703,), (3327,) - 6 classes
        # balanced_citeseer 1584 x (3703,), (1584,) - 6 classes
        
        # pubmed 19717 x (500,), (19717,) - 3 classes
        # balanced_pubmed 12309 x (500,), (12309,) - 3 classes

        # amazon_photo 7650 x (745,), (745,) - 8 classes
        # amazon_computers 13752 x (767,), (767,) - 10 classes
        
        name = "knn_edema_dataset",
        normalise_images = True
    ),
    model_config = GraphCoMVC(
        backbone_configs = (
            CNN(
                input_size = (3, 256, 256),
                # pretrained_model = "alexnet",
                # pretrained_features_out = 512,
                layers = [
                    ('conv', 11, 11, 8, None, ('stride', 4)),
                    ('bn',),
                    ('relu',),
                    ('pool', 2, 2),
                    ('conv', 5, 5, 16, 'relu', ('padding', 2), ('stride', 2)),
                    ('conv', 5, 5, 16, None, ('padding', 2), ('stride', 2)),
                    ('bn',),
                    ('relu',),
                    ('pool', 2, 2),
                    ('conv', 3, 3, 32, 'relu', ('padding', 1)),
                    ('conv', 3, 3, 32, 'relu', ('padding', 1)),
                    ('conv', 3, 3, 32, None, ('padding', 1)),
                    ('bn',),
                    ('relu',),
                    ('fc', 512)
                ],
            ),
            MLP(
                input_size = (66,),
                layers = [1024, 512],
                activation = None,
                use_bn = False,
            ),
            MLP(
                input_size = (210,),
                layers = [1024, 512],
                activation = None,
                use_bn = False,
            ),
        ),
        graph_attention_configs = GraphAttention(
            graph_format = "dense",
            layers = [512, 512],
            activation = "relu",
            use_bn = True,
            skip_connection = False,
            use_bias = True,
            mask = True,
            mask_weighting = True,
            mask_power = 2,
            attention_features = 1,
            dropout = .5,
        ),
        projector_config = None,
        fusion_config = Fusion(method = "weighted_mean", n_views = 3),
        cm_config = DDC(
            n_clusters = 2,
            layer_type = "graph_conv"
        ),
        loss_config = Loss(
            # ddc_1, ddc_2, ddc_2_flipped, ddc_3, contrast, contrast_graph,
            # reconstruction_structure, reconstruction_feature, semi_supervised
            funcs = "ddc_1|ddc_2|ddc_3|contrast",
            n_semi_supervised = 250,
            semi_supervised_weight = 1.,
            delta = 10.,
            # negative_samples_ratio = .50,
            epsilon_features = .1,
            epsilon_structure = .2,
        ),
        optimizer_config = Optimizer(
            learning_rate = 1e-3
            # Additional optimizer parameters go here
        ),
        shared_weights = True,
        # warmup_epochs = 20,
        warmup_funcs = "contrast",
        warmup_optimizer = Optimizer(
            learning_rate = 1e-3
        ),
    ),
    best_loss_term = "tot",
    n_epochs = 40,
    n_runs = 75,
    batch_size = 512,
    graphsaint_steps = 2,
    eval_interval = 5,
)
