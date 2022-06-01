import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

import numpy as np
from cnn import CXR_Dataset, CXR_Net


WANDB = False
DATASET = "knn_edema_dataset"
MODEL_STR = None
N_RUNS = 10
LR = 2e-5
EPOCHS = 15
BS = 100
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
PRETRAIN = True


if __name__ == "__main__":

    labels = np.load(f"data/processed/{DATASET}.npz")["labels"]
    train_idx = np.random.choice(
        range(len(labels)),
        size=int(.8*len(labels)),
        replace=False,
    )
    test_idx = np.random.choice(
        [_ for _ in list(range(len(labels))) if _ not in train_idx],
        size=len(labels)-len(train_idx),
        replace=False,
    )

    tr_DS = CXR_Dataset(
        np.load(f"data/processed/{DATASET}.npz")["view_0"][train_idx],
        labels[train_idx],
        train = True,
        downscale = True if MODEL_STR is not None else False,
        norm_mean = MEAN,
        norm_std = STD
    )
    te_DS = CXR_Dataset(
        np.load(f"data/processed/{DATASET}.npz")["view_0"][test_idx],
        labels[test_idx],
        train=False,
        downscale = True if MODEL_STR is not None else False,
        norm_mean = MEAN,
        norm_std = STD
    )

    training_generator = DataLoader(
        tr_DS,
        batch_size=BS,
        shuffle=True
    )
    test_generator = DataLoader(
        te_DS,
        batch_size=BS,
        shuffle=False
    )

    num_classes = len(np.unique(labels))
    if num_classes <= 2:
        num_classes = 1
    if WANDB: wandb.login()

    best_acc = 0

    for run in range(N_RUNS):
        epoch = 0
        model = CXR_Net(num_classes = num_classes, channels_in=3, model=MODEL_STR, pretrained=PRETRAIN)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        if torch.cuda.is_available():
            model.to('cuda')
        loss_objective = nn.BCELoss()

        tr_loss = []
        te_loss = []
        tr_acc = []
        te_acc = []

        if WANDB:
            wandb.init(
                project="CNN-test",
                entity="erlandg",
                group=MODEL_STR+("_pretrained" if PRETRAIN else ""),
                config={
                    "batch_size": BS,
                    "learning_rate": LR,
                    "pretrained": PRETRAIN
                },
                reinit=True
            )
        while epoch < EPOCHS:
            model.train()
            tot_loss = []
            for x_i, y_i in training_generator:
                optimizer.zero_grad()

                if y_i.shape[-1] > 1:
                    pred = F.softmax(model(x_i), 1)
                else:
                    pred = F.sigmoid(model(x_i))
                loss = loss_objective(pred, y_i)
                loss.backward()
                tot_loss.append(loss)
                tr_loss.append(loss.item())
                optimizer.step()
            
            epoch += 1

            model.eval()
            eval_acc, eval_loss = model.evaluate(test_generator, loss_objective)
            te_acc.append(eval_acc)
            te_loss.append(eval_loss)

            if WANDB:
                wandb.log({
                    "train/loss": torch.Tensor(tot_loss).mean(),
                    "test/loss": eval_loss,
                    "test/acc": eval_acc
                })

            print(f"{25*'*'} epoch {epoch} {25*'*'}")
            print(f"Loss: {torch.Tensor(tot_loss).mean()}")
            print(f"Testing loss: {eval_loss}")
            print(f"Testing accuracy: {eval_acc}")
        if WANDB: wandb.finish()
        best_acc = max(best_acc, max(te_acc))
    print(f"Max accuracy : {best_acc}")