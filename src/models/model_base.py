import torch.nn as nn

import config


class ModelBase(nn.Module):
    def __init__(self):
        """
        Model base class
        """
        super().__init__()

        self.fusion = None
        self.optimizer = None
        self.warmup_optimizer = None
        self.loss = None

    def calc_losses(self, labels=None, ignore_in_total=tuple()):
        if labels is not None: self.labels = labels
        return self.loss(self, ignore_in_total=ignore_in_total)

    def warmup_step(self, batch, labels, graph, epoch, it, n_batches, ignore_in_total=tuple()):
        self.labels = labels
        self.warmup_optimizer.zero_grad()
        _ = self([_ for _ in batch], graph = graph)
        losses = self.calc_losses(ignore_in_total=ignore_in_total)
        losses["tot"].backward()
        self.warmup_optimizer.step(epoch + it / n_batches)
        return losses

    def train_step(self, batch, labels, graph, epoch, it, n_batches, ignore_in_total=tuple(), semi_supervised_labels = None, **_):
        self.labels = labels
        self.semi_supervised_labels = semi_supervised_labels
        self.optimizer.zero_grad()
        _ = self([_ for _ in batch], graph = graph)
        
        # from torchviz import make_dot
        # make_dot(
        #     self([_ for _ in batch], graph = graph),
        #     params=dict(self.named_parameters()),
        #     show_attrs=True,
        #     show_saved=True
        # ).render(config.PROJECT_ROOT / "torchviz", format="png")

        losses = self.calc_losses(ignore_in_total=ignore_in_total)
        losses["tot"].backward()
        self.optimizer.step(epoch + it / n_batches)
        return losses
