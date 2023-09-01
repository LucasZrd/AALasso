import numpy as np
from scipy.spatial.distance import pdist, squareform
from typing import Tuple, List


def get_exponential_similarity(
    condensed_distance_matrix: np.ndarray, bandwidth: float, threshold: float
) -> np.ndarray:
    """Returns the similarity matrix using a gaussian kernel

    Args:
        condensed_distance_matrix (np.ndarray): Distances matrix
        bandwidth (float): Variance of the gaussian kernel
        threshold (float): Threshold to cut to 0 small values

    Returns:
        np.ndarray: Similarity matrix
    """
    exp_similarity = np.exp(-(condensed_distance_matrix**2) / bandwidth / bandwidth)
    res_arr = np.where(exp_similarity > threshold, exp_similarity, 0.0)
    return res_arr


def generate_stat_model(
    dim: int,
    order: int,
    thresh: float,
    mispecified: float,
    out_of_euclidean: float,
    sigma_noises: List[float],
    directed: bool = True,
    thresh_Z: bool = False,
    sigma: float = 1
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray, List[float]]:
    """Generates randomly parameters for the statistical model.

    Args:
        dim (int): Dimension of the model
        order (int): Order of the VAR model
        thresh (float): Threshold to promote sparsity (if 1 empty graph, if 0 dense graph)
        mispecified (float): Ratio of edges randomly removed from A* (if 1 empty graph, if 0 nothing changes)
        out_of_euclidean (float): Number of edges to randomly add
        sigma_noises (float): List of Std of the noise to add to
        directed (bool, optional): To forbide double causaility i -> j and j -> i . Defaults to True.
        thresh_Z (bool, optional): Whether or not cut small values in the prior. Defaults to False.
        sigma (float, optional): Factor to chose the kernel variance

    Returns:
        Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray, List[float]]: _description_
    """
    eps = 10e-3
    out = out_of_euclidean / (dim * (dim - 1))
    pos = np.random.rand(dim, 2)
    dist_mat_condensed = pdist(pos, metric="euclidean")
    sigma_median = sigma*np.median(dist_mat_condensed)
    if thresh_Z:
        prior_adjacency = squareform(
            get_exponential_similarity(dist_mat_condensed, sigma_median, thresh)
        )
    else:
        prior_adjacency = squareform(
            get_exponential_similarity(dist_mat_condensed, sigma_median, 0)
        )
    Z = squareform(get_exponential_similarity(dist_mat_condensed, sigma_median, thresh))

    priors = []
    SNR = []

    for sigma_noise in sigma_noises:
        noise = np.random.normal(0, sigma_noise, size = dim*dim).reshape(dim,dim)
        SNR.append(np.mean((noise+noise.T)**2/4))
        prior_adjacency = prior_adjacency + (noise + noise.T)/2
        prior_adjacency = np.where(prior_adjacency > 0, prior_adjacency, 0)
        priors.append(prior_adjacency)

    out_adjacency = np.zeros((dim, dim))

    for i in range(dim):
        for j in range(i + 1, dim):
            if Z[i, j]:
                if np.random.rand() < mispecified:
                    Z[i, j] = 0
                    Z[j, i] = 0
            else:
                if np.random.rand() < out:
                    if np.random.rand() < 0.5:
                        Z[i, j] = np.random.rand()
                        out_adjacency[i, j] = 1
                    else:
                        Z[j, i] = np.random.rand()
                        out_adjacency[j, i] = 1
    params = np.zeros((order, dim, dim))
    for o in range(order):
        for i in range(dim):
            for j in range(dim):
                params[o, i, j] = np.random.laplace(0, Z[i, j], 1)[0]
                if directed:
                    if np.random.rand() < 0.5:
                        params[o, i, j] = 0
                    else:
                        params[o, j, i] = 0

    if order == 1:
        while True:
            vp, _ = np.linalg.eig(params)
            if np.max(np.abs(vp)) < 0.95:
                break
            params *= 0.9
    else:
        bottom = np.hstack(
            [np.eye(dim * (order - 1)), np.zeros((dim * (order - 1), dim))]
        )
        while True:
            A = np.vstack([np.concatenate(list(params), axis=1), bottom])
            vp, _ = np.linalg.eig(A)
            if np.max(np.abs(vp)) < 0.95:
                break
            params *= 0.9
    return pos, params, priors, Z, SNR


def generate_var_signals(
    T: int, params: np.ndarray, order: int, dim: int, sigma: float
) -> np.ndarray:
    """Generate multivariate time series of length T following a VAR(order) model defined by params

    Args:
        T (int): Length of the time series
        params (np.ndarray): Parameters of the VAR model
        order (int): Order of the model
        dim (int): Dimension of the multivariate time series
        sigma (float): Standard deviation of the gaussian noise

    Returns:
        np.ndarray: Multivariate time series following the VAR model
    """
    X = np.random.normal(0, 1, dim * order).reshape(dim, order)
    for _ in range(T - order):
        Xt = np.zeros(dim)
        for i, A in enumerate(params):
            Xt += A.dot(X[:, -i - 1])
        Xt += np.random.normal(0, sigma, dim)
        X = np.hstack([X, Xt.reshape(-1, 1)])
    return X.T
