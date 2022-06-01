import numpy as np
import matplotlib.pyplot as plt
from numpy import random


def plot_xrays_groupings(
    save_path,
    fused_data,
    raw_data,
    pred,
    true_labels,
    N,
    fused_hue,
    view_hue,
    labels = None,
):
    k = len(np.unique(pred))

    n_views = int(len(view_hue) / len(fused_hue))
    
    idxs = np.concatenate([np.random.choice(
        np.where(pred == k_)[0],
        size=int(N)
    ) for k_ in range(k)])

    if k == 2:
        dims = (np.floor(np.sqrt(k*N)).astype(int), np.ceil(np.sqrt(k*N)).astype(int))
    else:
        dims = (k, np.ceil(len(idxs)/k).astype(int))
    plt.figure(figsize=(10,8.8), dpi=300)
    plt.suptitle("Sample X-rays with predictions", fontweight="semibold")
    if len(raw_data) == 1:
        for i, idx in enumerate(idxs):
            plt.subplot(dims[0],dims[1],i+1)
            if labels is None:
                plt.title(f"{fused_hue[idx]}\nTrue: {true_labels[idx]}")
            else:
                plt.title(f"{labels[int(fused_hue[idx])-1]}\nTrue: {labels[true_labels[idx]]}")
            plt.imshow(raw_data[0][idx][0], cmap="binary_r")
            plt.axis('off')
            for v in range(n_views):
                view_hue[idx+(v*len(fused_hue))] = "Sample, v{}".format(v)
            fused_hue[idx] = "Sample"
        plt.subplots_adjust(hspace=.15, wspace=0.)
        plt.savefig(save_path, bbox_inches="tight")
        plt.clf()
    else:
        for img in raw_data:
            for i, idx in enumerate(idxs):
                plt.subplot(len(raw_data) * dims[0],dims[1],i+1)
                if labels is None:
                    plt.title(f"{fused_hue[idx]}\nTrue: {true_labels[idx]}")
                else:
                    plt.title(f"{labels[int(fused_hue[idx])-1]}\nTrue: {labels[true_labels[idx]]}")
                plt.imshow(img[idx][0], cmap="binary_r")
                plt.axis('off')
                for v in range(n_views):
                    view_hue[idx+(v*len(fused_hue))] = "Sample, v{}".format(v)
                fused_hue[idx] = "Sample"
            plt.subplots_adjust(wspace=0)
            plt.savefig(save_path, bbox_inches="tight")
            plt.clf()
