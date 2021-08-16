import torch
import pickle
import numpy as np

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses import DEFAULT_EPS
from allrank.models.model_utils import get_torch_device


def sinkhorn_scaling(mat, mask=None, tol=1e-6, max_iter=50):
    """
    Sinkhorn scaling procedure.
    :param mat: a tensor of square matrices of shape N x M x M, where N is batch size
    :param mask: a tensor of masks of shape N x M
    :param tol: Sinkhorn scaling tolerance
    :param max_iter: maximum number of iterations of the Sinkhorn scaling
    :return: a tensor of (approximately) doubly stochastic matrices
    """
    if mask is not None:
        mat = mat.masked_fill(mask[:, None, :] | mask[:, :, None], 0.0)
        mat = mat.masked_fill(mask[:, None, :] & mask[:, :, None], 1.0)

    # import gcsfs
    # pickle.dump(mat.data.cpu().numpy(), gcsfs.GCSFileSystem().open("gs://kraken-task-data/MZHE-4010/test_ot_perm_matrix.pickle", "wb"))

    for _ in range(max_iter):
        mat = mat / mat.sum(dim=1, keepdim=True).clamp(min=DEFAULT_EPS)
        mat = mat / mat.sum(dim=2, keepdim=True).clamp(min=DEFAULT_EPS)

        if torch.max(torch.abs(mat.sum(dim=2) - 1.)) < tol and torch.max(torch.abs(mat.sum(dim=1) - 1.)) < tol:
            break

    if mask is not None:
        mat = mat.masked_fill(mask[:, None, :] | mask[:, :, None], 0.0)

    return mat


def ground_fun(x, y):
    return (x-y) ** 2


def expit(x):
    return 1. / (1. + torch.exp(-x))


def transform_fun(x, mask):
    x = x.masked_fill(mask, 0)
    m = x.size()[1]
    mask_sums = torch.sum(mask, dim=-1)
    sizes = m - mask_sums
    sums = torch.sum(x, dim=-1)
    means = sums / sizes

    sq_diffs = (x - means[:, None]) ** 2
    sq_diffs_sums = torch.sum(sq_diffs, dim=-1)
    sq_diffs_sums_corr = sq_diffs_sums - mask_sums * (means ** 2)
    std_devs = torch.sqrt(torch.clamp(sq_diffs_sums_corr / sizes, min=1e-6))

    return (x - means[:, None]) / std_devs[:, None]


def softmin(x, eps=1e-2, dim=-1, keepdim=True):
    return -eps * torch.logsumexp(-x / eps, dim=dim, keepdim=keepdim)


def ot_neural_sort(s, tau, mask, tol=1e-3, max_iter=20):
    dev = get_torch_device()

    s_transformed = transform_fun(-s, mask)
    s_transformed = s_transformed.masked_fill(mask, torch.max(s_transformed) + 6.)
    n, m = s.size()
    alphas = torch.zeros((n, m, 1), dtype=torch.float32, device=dev)
    betas = torch.zeros_like(alphas).transpose(1, 2)
    y_target = torch.arange(m, device=dev, dtype=torch.float32) / (m-1)

    # mod - only one value as we've got uniform priors on the source/target dists
    a = 1. / m
    b = 1. / m
    loga = np.log(a)
    logb = np.log(b)

    C = ground_fun(s_transformed[:, :, None], y_target[None, None, :])
    center = C - alphas - betas
    diff = 100.
    k = 0

    while diff > tol and k < max_iter:
        center = C - alphas - betas
        alphas += (tau * loga + softmin(center, tau, dim=2, keepdim=True))
        center = C - alphas - betas
        betas += (tau * logb + softmin(center, tau, dim=1, keepdim=True))
        if k % 5 == 0:
            center = C - alphas - betas
            diff = torch.max(torch.abs(torch.exp(-center) - b))
        k += 1

    P = torch.clamp(torch.exp(-center / tau), max=1e4)
    # ranks = torch.arange(m, dtype=torch.float32, device=dev)
    # S = (m * torch.matmul(P.transpose(1, 2), (s[:, :, None]))).squeeze()
    # R = (m * torch.matmul(P, ranks[None, :, None])).squeeze()

    return P.transpose(1, 2)


def deterministic_neural_sort(s, tau, mask):
    """
    Deterministic neural sort.
    Code taken from "Stochastic Optimization of Sorting Networks via Continuous Relaxations", ICLR 2019.
    Minor modifications applied to the original code (masking).
    :param s: values to sort, shape [batch_size, slate_length]
    :param tau: temperature for the final softmax function
    :param mask: mask indicating padded elements
    :return: approximate permutation matrices of shape [batch_size, slate_length, slate_length]
    """
    dev = get_torch_device()

    n = s.size()[1]

    s = transform_fun(s[:, :, 0], mask).unsqueeze(-1)

    one = torch.ones((n, 1), dtype=torch.float32, device=dev)
    s = s.masked_fill(mask[:, :, None], -1e8)
    A_s = torch.abs(s - s.permute(0, 2, 1))
    A_s = A_s.masked_fill(mask[:, :, None] | mask[:, None, :], 0.0)

    B = torch.matmul(A_s, torch.matmul(one, torch.transpose(one, 0, 1)))

    temp = [n - m + 1 - 2 * (torch.arange(n - m, device=dev) + 1) for m in mask.squeeze(-1).sum(dim=1)]
    temp = [t.type(torch.float32) for t in temp]
    temp = [torch.cat((t, torch.zeros(n - len(t), device=dev))) for t in temp]
    scaling = torch.stack(temp).type(torch.float32).to(dev)  # type: ignore

    s = s.masked_fill(mask[:, :, None], 0.0)
    C = torch.matmul(s, scaling.unsqueeze(-2))

    P_max = (C - B).permute(0, 2, 1)
    P_max = P_max.masked_fill(mask[:, :, None] | mask[:, None, :], -np.inf)
    P_max = P_max.masked_fill(mask[:, :, None] & mask[:, None, :], 1.0)
    sm = torch.nn.Softmax(-1)
    P_hat = sm(P_max / tau)
    return P_hat


def sample_gumbel(samples_shape, device, eps=1e-10) -> torch.Tensor:
    """
    Sampling from Gumbel distribution.
    Code taken from "Stochastic Optimization of Sorting Networks via Continuous Relaxations", ICLR 2019.
    Minor modifications applied to the original code (masking).
    :param samples_shape: shape of the output samples tensor
    :param device: device of the output samples tensor
    :param eps: epsilon for the logarithm function
    :return: Gumbel samples tensor of shape samples_shape
    """
    U = torch.rand(samples_shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def stochastic_neural_sort(s, n_samples, tau, mask, beta=1.0, log_scores=True, eps=1e-10):
    """
    Stochastic neural sort. Please note that memory complexity grows by factor n_samples.
    Code taken from "Stochastic Optimization of Sorting Networks via Continuous Relaxations", ICLR 2019.
    Minor modifications applied to the original code (masking).
    :param s: values to sort, shape [batch_size, slate_length]
    :param n_samples: number of samples (approximations) for each permutation matrix
    :param tau: temperature for the final softmax function
    :param mask: mask indicating padded elements
    :param beta: scale parameter for the Gumbel distribution
    :param log_scores: whether to apply the logarithm function to scores prior to Gumbel perturbation
    :param eps: epsilon for the logarithm function
    :return: approximate permutation matrices of shape [n_samples, batch_size, slate_length, slate_length]
    """
    dev = get_torch_device()

    batch_size = s.size()[0]
    n = s.size()[1]
    s_positive = s + torch.abs(s.min())
    samples = beta * sample_gumbel([n_samples, batch_size, n, 1], device=dev)
    if log_scores:
        s_positive = torch.log(s_positive + eps)

    s_perturb = (s_positive + samples).view(n_samples * batch_size, n, 1)
    mask_repeated = mask.repeat_interleave(n_samples, dim=0)

    P_hat = deterministic_neural_sort(s_perturb, tau, mask_repeated)
    P_hat = P_hat.view(n_samples, batch_size, n, n)
    return P_hat
