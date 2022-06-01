from config.defaults import (
    Experiment,
    SiMVC,
    CNN,
    MLP,
    DDC,
    Fusion,
    Loss,
    Dataset,
    CoMVC,
    Optimizer,
)


mimic = Experiment(
    # edema_dataset 3520 x (1,256,256), (66,), (210,) - 2 classes

    # cora 2708 x (1433,), (2708,) - 7 classes
    # balanced_cora 1260 x (1433,), (1260,) - 7 classes

    # citeseer 3327 x (3703,), (3327,) - 6 classes
    # balanced_citeseer 1584 x (3703,), (1584,) - 6 classes
    
    # pubmed 19717 x (500,), (19717,) - 3 classes
    # balanced_pubmed 12309 x (500,), (12309,) - 3 classes

    # amazon_photo 7650 x (745,), (745,) - 8 classes
    # amazon_computers 13752 x (767,), (767,) - 10 classes

    dataset_config=Dataset(
        name="pubmed",
        normalise_images = False
    ),
    model_config=SiMVC(
        backbone_configs=(
            # CNN(
            #     input_size = (1, 256, 256),
            #     # pretrained_model = "alexnet",
            #     # pretrained_features_out = 512,
            #     layers = [
            #         ("conv", 11, 11, 8, None, ('stride', 4)),
            #         ("bn",),
            #         ("relu",),
            #         ("pool", 2, 2),
            #         ("conv", 5, 5, 16, "relu", ('padding', 2), ('stride', 2)),
            #         ("conv", 5, 5, 16, None, ('padding', 2), ('stride', 2)),
            #         ("bn",),
            #         ("relu",),
            #         ("pool", 2, 2),
            #         ("conv", 3, 3, 32, "relu", ('padding', 1)),
            #         ("conv", 3, 3, 32, "relu", ('padding', 1)),
            #         ("conv", 3, 3, 32, None, ('padding', 1)),
            #         ("bn",),
            #         ("relu",),
            #         ("fc", 512),
            #     ],
            # ),
            MLP(
                input_size = (500,),
                layers = [256, 128],
                use_bn = True
            ),
            MLP(
                input_size = (19717,),
                layers = [256, 128],
                use_bn = True
            ),
        ),
        fusion_config=Fusion(method="weighted_mean", n_views=2),
        cm_config=DDC(n_clusters=3),
        loss_config=Loss(
            funcs="ddc_1|ddc_2|ddc_3",
            # Additional loss parameters go here
        ),
        optimizer_config=Optimizer(
            learning_rate=1e-3,
            # Additional optimizer parameters go here
        ),
    ),
    best_loss_term="tot",
    n_epochs=20,
    n_runs=150,
    batch_size=512,
    eval_interval = 5,
)


mimic_contrast = Experiment(
    dataset_config=Dataset(
        # edema_dataset 3520 x (1,256,256), (66,), (210,) - 2 classes

        # cora 2708 x (1433,), (2708,) - 7 classes
        # balanced_cora 1260 x (1433,), (1260,) - 7 classes

        # citeseer 3327 x (3703,), (3327,) - 6 classes
        # balanced_citeseer 1584 x (3703,), (1584,) - 6 classes
        
        # pubmed 19717 x (500,), (19717,) - 3 classes
        # balanced_pubmed 12309 x (500,), (12309,) - 3 classes

        # amazon_photo 7650 x (745,), (745,) - 8 classes
        # amazon_computers 13752 x (767,), (767,) - 10 classes

        name="cora",
        normalise_images = False
    ),
    model_config=CoMVC(
        backbone_configs=(
            # CNN(
            #     input_size = (1, 256, 256),
            #     # pretrained_model = "alexnet",
            #     # pretrained_features_out = 512,
            #     layers = [
            #         ("conv", 11, 11, 8, None, ('stride', 4)),
            #         ("bn",),
            #         ("relu",),
            #         ("pool", 2, 2),
            #         ("conv", 5, 5, 16, "relu", ('padding', 2), ('stride', 2)),
            #         ("conv", 5, 5, 16, None, ('padding', 2), ('stride', 2)),
            #         ("bn",),
            #         ("relu",),
            #         ("pool", 2, 2),
            #         ("conv", 3, 3, 32, "relu", ('padding', 1)),
            #         ("conv", 3, 3, 32, "relu", ('padding', 1)),
            #         ("conv", 3, 3, 32, None, ('padding', 1)),
            #         ("bn",),
            #         ("relu",),
            #         ("fc", 512),
            #     ],
            # ),
            MLP(
                input_size = (1433,),
                layers = [512, 512],
                activation = None,
                use_bn = False,
            ),
            MLP(
                input_size = (2708,),
                layers = [512, 512],
                activation = None,
                use_bn = False,
            ),
        ),
        projector_config=None,
        fusion_config=Fusion(method="weighted_mean", n_views=2),
        cm_config=DDC(n_clusters=7),
        loss_config=Loss(
            funcs="ddc_1|ddc_2|ddc_3|contrast",
            n_semi_supervised=100,
            delta=6.,
        ),
        optimizer_config=Optimizer(
            learning_rate=1e-3,
            # Additional optimizer parameters go here
        ),
    ),
    best_loss_term="tot",
    n_epochs=20,
    n_runs=500,
    batch_size=1024,
    eval_interval = 5,
)
