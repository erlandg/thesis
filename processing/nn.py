from calendar import EPOCH
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


class myDataset(Dataset):
    
    def __init__(self, x, y, train, normalise = True):
        self.x = torch.Tensor(self.normalise(x))
        self.y = torch.Tensor(y).reshape(-1,1)
        # self.y = F.one_hot(
        #     torch.Tensor(y).type(torch.long),
        #     len(np.unique(y))
        # ).type(torch.float)
        self.train = train
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def normalise(x):
        assert not np.isnan(x).any()
        return (x - x.mean(0)[None,:])/x.std(0)[None,:]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx].to(self.device), self.y[idx].to(self.device)


class Net(nn.Module):

    def __init__(self, input_features, layers) -> None:
        super(Net, self).__init__()
        layers = layers
        mods = []
        for out_ in layers[:-1]:
            mods.extend(self.linear_layer(input_features, out_))
            input_features = out_
        mods.extend([
            nn.Linear(input_features, layers[-1]),
            nn.Sigmoid()
        ])
        self.net = nn.Sequential(*mods)
        print(self.net)

    @staticmethod
    def linear_layer(in_features, out_features):
        return [
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU()
        ] 

    def forward(self, x):
        return self.net(x)

    def evaluate(self, te_loader, loss_objective):
        no_cor = 0
        no_tot = 0
        loss = []
        
        with torch.no_grad():
            for x, y in te_loader:
                y = y.to("cpu")
                if y.shape[-1] > 1:
                    y_gt = torch.argmax(y, 1)
                else:
                    y_gt = y
                out = self.forward(x).to('cpu')
                if out.shape[-1] == 1:
                    model_pred = out.round()
                else:
                    model_pred = torch.argmax(out, 1)
                no_cor += (model_pred == y_gt).sum()
                no_tot += len(y)
                batch_loss = loss_objective(out, y)
                loss.append(batch_loss)
        return no_cor/no_tot, torch.Tensor(loss).mean()


LAYERS = [1024, 512]
OUTPUT_DIM = 1
DATASET = "edema_dataset"
N_RUNS = 5
EPOCHS = 20
BS = 100
LR = 1e-3



if __name__ == "__main__":

    labels = np.load(f"data/processed/{DATASET}.npz")["labels"]
    num_classes = len(np.unique(labels))
    vital_signs = np.load(f"data/processed/{DATASET}.npz")["view_1"]
    lab_data = np.load(f"data/processed/{DATASET}.npz")["view_2"]
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

    vs_loader = DataLoader(
        myDataset(
            vital_signs[train_idx],
            labels[train_idx],
            train = True,
            normalise=True
        ),
        batch_size=BS,
        shuffle=True
    )
    vs_te_loader = DataLoader(
        myDataset(
            vital_signs[test_idx],
            labels[test_idx],
            train = False,
            normalise=True
        ),
        batch_size=BS,
        shuffle=True
    )
    lab_loader = DataLoader(
        myDataset(
            lab_data[train_idx],
            labels[train_idx],
            train = False,
            normalise=True
        ),
        batch_size=BS,
        shuffle=True
    )
    lab_te_loader = DataLoader(
        myDataset(
            lab_data[test_idx],
            labels[test_idx],
            train = False,
            normalise=True
        ),
        batch_size=BS,
        shuffle=True
    )

accs = {}

for name, data, tr_loader, te_loader in zip(["Vital signs", "Lab data"], [vital_signs, lab_data], [vs_loader, lab_loader], [vs_te_loader, lab_te_loader]):

    best_acc = 0

    for run in range(N_RUNS):

        tr_loss = []
        te_loss = []
        tr_acc = []
        te_acc = []

        net = Net(
            input_features=data.shape[1],
            layers = LAYERS + [OUTPUT_DIM]
        )
        if torch.cuda.is_available(): net = net.to("cuda")
        optimizer = torch.optim.Adam(net.parameters(), lr=LR)
        loss_objective = nn.BCELoss()

        for e in range(EPOCHS):
            net.train()
            tot_loss = []
            for x_i, y_i in tr_loader:
                optimizer.zero_grad()

                pred = net(x_i)
                loss = loss_objective(pred, y_i)
                loss.backward()
                tot_loss.append(loss)
                tr_loss.append(loss.item())
                optimizer.step()

            net.eval()
            eval_acc, eval_loss = net.evaluate(te_loader, loss_objective)
            te_acc.append(eval_acc)
            te_loss.append(eval_loss)
            print(f"{25*'*'} epoch {e} {25*'*'}")
            print(f"Loss: {torch.Tensor(tot_loss).mean()}")
            print(f"Testing loss: {eval_loss}")
            print(f"Testing accuracy: {eval_acc}")

        best_acc = max(best_acc, max(te_acc))
    accs[name] = best_acc
            

print(f"\n\n{50*'*'}")
print(f"With layers {LAYERS} :")
for name_, acc_ in accs.items():
    print(f"{name_} max val. accuracy : {acc_}")
