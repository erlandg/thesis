import enum
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms

import numpy as np

from skimage.exposure import equalize_hist, adjust_sigmoid
from skimage.filters import sobel



class CXR_Dataset(Dataset):
    
    def __init__(self, x, y, train, dims=3, downscale = False, norm_mean=None, norm_std=None):
        self.x = torch.cat(
            tuple(torch.Tensor(x) for _ in range(3)),
            dim=1
        ).type(torch.float)

        if (norm_mean is not None) and (norm_std is not None):
            assert (dims == len(norm_mean)) and (dims == len(norm_std)), "Mean and std requires an equal number of dimensions as the images."
            if downscale: preproc = [transforms.CenterCrop(224)]
            else: preproc = []
            preprocess = transforms.Compose(preproc + [
                transforms.Normalize(mean=norm_mean, std=norm_std)
            ])
            self.x = preprocess(self.x)

        self.y = torch.Tensor(y)
        if len(np.unique(y)) <= 2:
            self.y = torch.Tensor(y).reshape(-1,1)
        else:
            self.y = F.one_hot(
                self.y.type(torch.long),
                len(torch.unique(self.y))
            ).type(torch.float)
        self.train = train
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.train:
            return self.x[idx].to(self.device), self.y[idx].to(self.device)
        else:
            return self.x[idx].to(self.device), self.y[idx].to(self.device)


class CXR_Net(nn.Module):
    
    def __init__(self, num_classes, channels_in, model=None, pretrained=False):
        super(CXR_Net, self).__init__()
        self.num_classes = num_classes
        self.channels_in = channels_in

        if model is None:
            self.net = nn.Sequential(
                *self.cnn_layer(self.channels_in, 8, (11,11), bn=True, stride=4),
                nn.MaxPool2d(kernel_size=2),
                *self.cnn_layer(8, 16, (5,5), padding=2, stride=2),
                *self.cnn_layer(16, 16, (5,5), bn=True, padding=2, stride=2),
                nn.MaxPool2d(kernel_size=2),
                *self.cnn_layer(16, 32, (3,3), padding=1),
                *self.cnn_layer(32, 32, (3,3), padding=1),
                *self.cnn_layer(32, 32, (3,3), bn=True, padding=1),
                nn.Flatten(),
                nn.Linear(512, 512),
                nn.Linear(512, self.num_classes),
            )
        elif model == "resnet18":
            assert channels_in == 3, f"ResNet requires 3 channels in, got {channels_in}."
            self.net = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=pretrained)
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)
        elif model == "vgg16":
            assert channels_in == 3, f"VGG-16 requires 3 channels in, got {channels_in}."
            self.net = torch.hub.load("pytorch/vision:v0.10.0", "vgg16_bn", pretrained=pretrained)
            self.net.classifier[6] = nn.Linear(self.net.classifier[6].in_features, num_classes)
        elif model == "alexnet":
            assert channels_in == 3, f"AlexNet requires 3 channels in, got {channels_in}."
            self.net = torch.hub.load("pytorch/vision:v0.10.0", "alexnet", pretrained=pretrained)
            self.net.classifier[6] = nn.Linear(self.net.classifier[6].in_features, num_classes)

        if pretrained is not True:
            self.net.apply(self.__init_weights)
        print(self.net)

    @staticmethod
    def cnn_layer(channels_in, channels_out, kernel_size, bn=False, **kwargs):
        seq = [nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, **kwargs)]
        if bn:
            seq.append(nn.BatchNorm2d(channels_out))
        seq.append(nn.ReLU())
        return seq

    def __init_weights(self, w):
        if type(w) in [nn.Conv2d, nn.Linear]:
            nn.init.kaiming_uniform_(w.weight)
            if w.bias is not None:
                w.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net(x)

    def evaluate(self, test_generator, loss_objective):
        no_cor = 0
        no_tot = 0
        loss = []
        
        with torch.no_grad():
            for x, y in test_generator:
                y = y.to("cpu")
                if y.shape[1] > 1:
                    y_gt = torch.argmax(y, 1)
                    out = F.softmax(self.forward(x), 1).to('cpu')
                    model_pred = torch.argmax(out, 1)
                else:
                    y_gt = y
                    out = torch.sigmoid(self.forward(x)).to("cpu")
                    model_pred = out.round()

                no_cor += (model_pred == y_gt).sum()
                no_tot += len(y_gt)
                batch_loss = loss_objective(out, y)
                loss.append(batch_loss)
        return no_cor/no_tot, torch.Tensor(loss).mean()
