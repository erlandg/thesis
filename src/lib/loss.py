from cv2 import norm
from numpy import dtype
import torch as th
import torch.nn as nn

import config
from lib import kernel

EPSILON = 1E-9
DEBUG_MODE = False


def triu(X):
    # Sum of strictly upper triangular part
    return th.sum(th.triu(X, diagonal=1))


def _atleast_epsilon(X, eps=EPSILON):
    """
    Ensure that all elements are >= `eps`.

    :param X: Input elements
    :type X: th.Tensor
    :param eps: epsilon
    :type eps: float
    :return: New version of X where elements smaller than `eps` have been replaced with `eps`.
    :rtype: th.Tensor
    """
    return th.where(X < eps, X.new_tensor(eps), X)


def d_cs(A, K, n_clusters):
    """
    Cauchy-Schwarz divergence.

    :param A: Cluster assignment matrix
    :type A:  th.Tensor
    :param K: Kernel matrix
    :type K: th.Tensor
    :param n_clusters: Number of clusters
    :type n_clusters: int
    :return: CS-divergence
    :rtype: th.Tensor
    """
    nom = th.t(A) @ K @ A
    dnom_squared = th.unsqueeze(th.diagonal(nom), -1) @ th.unsqueeze(th.diagonal(nom), 0)

    nom = _atleast_epsilon(nom)
    dnom_squared = _atleast_epsilon(dnom_squared, eps=EPSILON**2)

    d = 2 / (n_clusters * (n_clusters - 1)) * triu(nom / th.sqrt(dnom_squared))
    return d


# ======================================================================================================================
# Loss terms
# ======================================================================================================================

class LossTerm:
    # Names of tensors required for the loss computation
    required_tensors = []

    def __init__(self, *args, **kwargs):
        """
        Base class for a term in the loss function.

        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        """
        pass

    def __call__(self, net, cfg, extra):
        raise NotImplementedError()


class DDC1(LossTerm):
    """
    L_1 loss from DDC
    """
    required_tensors = ["hidden_kernel"]

    def __call__(self, net, cfg, extra):
        return d_cs(net.output, extra["hidden_kernel"], cfg.n_clusters)


class DDC2(LossTerm):
    """
    L_2 loss from DDC
    """
    def __call__(self, net, cfg, extra):
        n = net.output.size(0)
        return 2 / (n * (n - 1)) * triu(net.output @ th.t(net.output))


class DDC2Flipped(LossTerm):
    """
    Flipped version of the L_2 loss from DDC. Used by EAMC
    """

    def __call__(self, net, cfg, extra):
        return 2 / (cfg.n_clusters * (cfg.n_clusters - 1)) * triu(th.t(net.output) @ net.output)


class DDC3(LossTerm):
    """
    L_3 loss from DDC
    """
    required_tensors = ["hidden_kernel"]

    def __init__(self, cfg):
        super().__init__()
        self.eye = th.eye(cfg.n_clusters, device=config.DEVICE)

    def __call__(self, net, cfg, extra):
        m = th.exp(-kernel.cdist(net.output, self.eye))
        return d_cs(m, extra["hidden_kernel"], cfg.n_clusters)


class Contrastive(LossTerm):
    large_num = 1e9

    def __init__(self, cfg):
        """
        Contrastive loss function

        :param cfg: Loss function config
        :type cfg: config.defaults.Loss
        """
        super().__init__()
        # Select which implementation to use
        if cfg.negative_samples_ratio == -1:
            self._loss_func = self._loss_without_negative_sampling
        else:
            self.eye = th.eye(cfg.n_clusters, device=config.DEVICE)
            self._loss_func = self._loss_with_negative_sampling

        # Set similarity function
        if cfg.contrastive_similarity == "cos":
            self.similarity_func = self._cosine_similarity
        elif cfg.contrastive_similarity == "gauss":
            self.similarity_func = kernel.vector_kernel
        else:
            raise RuntimeError(f"Invalid contrastive similarity: {cfg.contrastive_similarity}")

    @staticmethod
    def _norm(mat):
        return th.nn.functional.normalize(mat, p=2, dim=1)

    @staticmethod
    def get_weight(net):
        w = th.min(net.fusion.w)
        return w

    @classmethod
    def _normalized_projections(cls, net):
        n = net.projections.size(0) // 2
        h1, h2 = net.projections[:n], net.projections[n:]
        h2 = cls._norm(h2)
        h1 = cls._norm(h1)
        return n, h1, h2

    @classmethod
    def _cosine_similarity(cls, projections):
        h = cls._norm(projections)
        return h @ h.t()

    def _draw_negative_samples(self, net, cfg, v, pos_indices):
        """
        Construct set of negative samples.

        :param net: Model
        :type net: Union[models.simple_mvc.SiMVC, models.contrastive_mvc.CoMVC]
        :param cfg: Loss config
        :type cfg: config.defaults.Loss
        :param v: Number of views
        :type v: int
        :param pos_indices: Row indices of the positive samples in the concatenated similarity matrix
        :type pos_indices: th.Tensor
        :return: Indices of negative samples
        :rtype: th.Tensor
        """
        cat = net.output.detach().argmax(dim=1)
        cat = th.cat(v * [cat], dim=0)

        weights = (1 - self.eye[cat])[:, cat[[pos_indices]]].T
        n_negative_samples = int(cfg.negative_samples_ratio * cat.size(0))
        if not weights.any():
            return "empty"
        negative_sample_indices = th.multinomial(weights, n_negative_samples, replacement=True)
        if DEBUG_MODE:
            self._check_negative_samples_valid(cat, pos_indices, negative_sample_indices)
        return negative_sample_indices

    @staticmethod
    def _check_negative_samples_valid(cat, pos_indices, neg_indices):
        pos_cats = cat[pos_indices].view(-1, 1)
        neg_cats = cat[neg_indices]
        assert (pos_cats != neg_cats).detach().cpu().numpy().all()

    @staticmethod
    def _get_positive_samples(logits, v, n):
        """
        Get positive samples

        :param logits: Input similarities
        :type logits: th.Tensor
        :param v: Number of views
        :type v: int
        :param n: Number of samples per view (batch size)
        :type n: int
        :return: Similarities of positive pairs, and their indices
        :rtype: Tuple[th.Tensor, th.Tensor]
        """
        diagonals = []
        inds = []
        for i in range(1, v):
            diagonal_offset = i * n
            diag_length = (v - i) * n
            _upper = th.diagonal(logits, offset=diagonal_offset)
            _lower = th.diagonal(logits, offset=-1 * diagonal_offset)
            _upper_inds = th.arange(0, diag_length)
            _lower_inds = th.arange(i * n, v * n)
            if DEBUG_MODE:
                assert _upper.size() == _lower.size() == _upper_inds.size() == _lower_inds.size() == (diag_length,)
            diagonals += [_upper, _lower]
            inds += [_upper_inds, _lower_inds]

        pos = th.cat(diagonals, dim=0)
        pos_inds = th.cat(inds, dim=0)
        return pos, pos_inds

    def _loss_with_negative_sampling(self, net, cfg, extra):
        """
        Contrastive loss implementation with negative sampling.

        :param net: Model
        :type net: Union[models.simple_mvc.SiMVC, models.contrastive_mvc.CoMVC]
        :param cfg: Loss config
        :type cfg: config.defaults.Loss
        :param extra:
        :type extra:
        :return: Loss value
        :rtype: th.Tensor
        """
        n = net.output.size(0)
        v = len(net.backbone_outputs)
        logits = self.similarity_func(net.projections) / cfg.tau

        pos, pos_inds = self._get_positive_samples(logits, v, n)
        neg_inds = self._draw_negative_samples(net, cfg, v, pos_inds)
        if neg_inds == "empty":
            # Skipping batch as there are no negative samples
            return th.tensor(0., requires_grad=True).to(logits.device)
        neg = logits[pos_inds.view(-1, 1), neg_inds]

        inputs = th.cat((pos.view(-1, 1), neg), dim=1)
        labels = th.zeros(v * (v - 1) * n, device=logits.device, dtype=th.long)
        loss = th.nn.functional.cross_entropy(inputs, labels)

        if cfg.adaptive_contrastive_weight:
            loss *= self.get_weight(net)

        return cfg.delta * loss

    def _loss_without_negative_sampling(self, net, cfg, extra):
        """
        Contrastive loss implementation without negative sampling.
        Adapted from: https://github.com/google-research/simclr/blob/master/objective.py

        :param net: Model
        :type net: Union[models.simple_mvc.SiMVC, models.contrastive_mvc.CoMVC]
        :param cfg: Loss config
        :type cfg: config.defaults.Loss
        :param extra:
        :type extra:
        :return:
        :rtype:
        """
        assert len(net.backbone_outputs) == 2, "Contrastive loss without negative sampling only supports 2 views."
        n, h1, h2 = self._normalized_projections(net)

        labels = th.arange(0, n, device=h1.device, dtype=th.long)
        masks = th.eye(n, device=h1.device)

        logits_aa = ((h1 @ h1.t()) / cfg.tau) - masks * self.large_num
        logits_bb = ((h2 @ h2.t()) / cfg.tau) - masks * self.large_num

        logits_ab = (h1 @ h2.t()) / cfg.tau
        logits_ba = (h2 @ h1.t()) / cfg.tau

        loss_a = th.nn.functional.cross_entropy(th.cat((logits_ab, logits_aa), dim=1), labels)
        loss_b = th.nn.functional.cross_entropy(th.cat((logits_ba, logits_bb), dim=1), labels)

        loss = (loss_a + loss_b)

        if cfg.adaptive_contrastive_weight:
            loss = loss * self.get_weight(net)

        return cfg.delta * loss

    def __call__(self, net, cfg, extra):
        return self._loss_func(net, cfg, extra)



class FeatureReconstruction(LossTerm):
    
    def _loss_func(self, net, cfg, extra):
        loss = 0
        # for fusion_weight, x_hat, x, z_hat, z in zip(net.fusion.w, net.X_reconstructed, net.input, net.embedding_reconstructed, net.pre_graph_backbones):
        for fusion_weight, x_hat, x in zip(net.fusion.w, net.X_reconstructed, net.input):
            loss = loss + (fusion_weight * th.nn.functional.mse_loss(x_hat, x, reduction="mean"))
            # loss = loss + (fusion_weight * th.nn.functional.mse_loss(z_hat, z, reduction="mean"))
        return cfg.epsilon_features * loss

    def __call__(self, net, cfg, extra):
        return self._loss_func(net, cfg, extra)


class StructureReconstruction(LossTerm):

    def _loss_func(self, net, cfg, extra):
        loss = 0
        for fusion_weight, G_hat in zip(net.fusion.w, net.G_reconstructed):
            # loss = loss + (fusion_weight * G_hat[net.graph == 0].mean())
            loss = loss + (fusion_weight * th.nn.functional.mse_loss(G_hat, net.graph, reduction="mean"))/G_hat.shape[1]
        return cfg.epsilon_structure * loss

    def __call__(self, net, cfg, extra):
        return self._loss_func(net, cfg, extra)


class ContrastiveGraph(LossTerm):

    def _get_contrastive_loss(self, net, idxs):
        loss = 0
        for i, (row, knn_idx) in enumerate(zip(net.ddc.learned_graph.graph, idxs)):
            sum_ = th.sum(th.exp(row)) - th.exp(row[i])
            for idx in knn_idx[1:]:
                loss = loss - th.log(th.exp(row[idx]) / sum_)
        return loss

    def _loss_func(self, net, cfg, extra):
        loss = 0
        for weight, H_v in zip(net.fusion.w, net.H):
            net.nn.fit(H_v.detach().cpu())
            _, idxs = net.nn.kneighbors(H_v.detach().cpu())
            loss_ = th.norm(H_v.T - th.matmul(H_v.T, net.ddc.learned_graph.graph))**2 + 1 * self._get_contrastive_loss(net, idxs)
            loss = loss + weight * loss_ / H_v.shape[0]**2
        return loss

    def __call__(self, net, cfg, extra):
        return self._loss_func(net, cfg, extra)


class SemiSupervised(LossTerm):

    def __call__(self, net, cfg, extra):
        idx_ = net.semi_supervised_labels != -1
        if len(idx_) > 0:
            labels = th.nn.functional.one_hot(net.semi_supervised_labels[idx_].long(), num_classes=net.output.shape[1]).float()
            return th.nn.functional.binary_cross_entropy(net.output[idx_], labels, reduction="mean")
        else: return 0.

# ======================================================================================================================
# Extra functions
# ======================================================================================================================

def hidden_kernel(net, cfg):
    return kernel.vector_kernel(net.hidden, cfg.rel_sigma)


# ======================================================================================================================
# Loss class
# ======================================================================================================================

class Loss(nn.Module):
    # Possible terms to include in the loss
    TERM_CLASSES = {
        "ddc_1": DDC1,
        "ddc_2": DDC2,
        "ddc_2_flipped": DDC2Flipped,
        "ddc_3": DDC3,
        "contrast": Contrastive,
        "contrast_graph": ContrastiveGraph,
        "reconstruction_structure": StructureReconstruction,
        "reconstruction_feature": FeatureReconstruction,
        "semi_supervised": SemiSupervised,
    }
    # Functions to compute the required tensors for the terms.
    EXTRA_FUNCS = {
        "hidden_kernel": hidden_kernel,
    }

    def __init__(self, cfg):
        """
        Implementation of a general loss function

        :param cfg: Loss function config
        :type cfg: config.defaults.Loss
        """
        super().__init__()
        self.cfg = cfg

        self.names = cfg.funcs.split("|")
        self.weights = cfg.weights if cfg.weights is not None else len(self.names) * [1]

        self.terms = []
        for term_name in self.names:
            try:
                self.terms.append(self.TERM_CLASSES[term_name](cfg))
            except KeyError:
                print(f"'{term_name}' not recognised as a loss term")

        self.required_extras_names = list(set(sum([t.required_tensors for t in self.terms], [])))

    def forward(self, net, ignore_in_total=tuple()):
        extra = {name: self.EXTRA_FUNCS[name](net, self.cfg) for name in self.required_extras_names}
        loss_values = {}
        for name, term, weight in zip(self.names, self.terms, self.weights):
            value = term(net, self.cfg, extra)
            # If we got a dict, add each term from the dict with "name/" as the scope.
            if isinstance(value, dict):
                for key, _value in value.items():
                    loss_values[f"{name}/{key}"] = weight * _value
            # Otherwise, just add the value to the dict directly
            else:
                loss_values[name] = weight * value

        loss_values["tot"] = sum([loss_values[k] for k in loss_values.keys() if k not in ignore_in_total])
        return loss_values

