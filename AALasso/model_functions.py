import numpy as np
from typing import Union, Tuple
import logging
import sys
from metrics import rNMSE
import asgl


class VAR:
    """
    VAR model class taking as parameters the dimension, the order and the parameters and allowing to forecast
    """

    def __init__(self, order: int, dim: int, params: np.ndarray) -> None:
        self.P = order
        self.params = params
        self.dim = dim

    def predict_next(self, X: np.ndarray) -> np.ndarray:
        """Forecast next point given a multivariate time series

        Args:
            X (_type_): Multivariate time series

        Returns:
            np.ndarray : next point
        """
        pred = np.zeros(self.dim)
        for t in range(self.P):
            pred += X[-1 - t] @ self.params[t].T
        return pred

    def forecast(self, X: np.ndarray, lag: int) -> np.ndarray:
        """Forecast lag next points given a multivariate time series X

        Args:
            X (np.ndarray): multivariate time series
            lag (int): number of points to forecast

        Returns:
            np.ndarray: next lag points
        """
        preds = []
        x = np.copy(X)
        for k in range(lag):
            pred = self.predict_next(x)
            preds.append(pred)
            x = np.vstack([X, pred])
        return np.array(preds)


def outsamples_predictions(model: VAR, data: np.ndarray, steps: int) -> list:
    """Reconstruct time series using the VAR model

    Args:
        model (VAR): VAR model
        data (np.ndarray): original multivariate time series
        steps (int): nb of points to forecast at each step

    Returns:
        list: reconstruction
    """
    preds = []
    for k in range(model.P, len(data), steps):
        preds += list(model.forecast(data[:k], steps))
    return preds


def transform_data(X):
    Y = X[-1]
    new_X = X[-2]
    for k in range(1, len(X) - 1):
        new_X = np.concatenate([new_X, X[-k - 2]], axis=1)
    return new_X, Y


def complete_coeffs(params: np.ndarray, i: int, dim: int, order: int) -> list:
    """Complete the parameters matrix with a zero for auto-edges

    Args:
        params (np.ndarray): parameters of the VAR model
        i (int): _index
        dim (int): dimension
        order (int): order of the model

    Returns:
        list: completed parameters
    """
    p = []
    params_list = list(params)
    for t in range(order):
        p += (
            params_list[t * (dim - 1) : t * (dim - 1) + i]
            + [0]
            + params_list[t * (dim - 1) + i : (t + 1) * (dim - 1)]
        )
    return p



def train_lasso(
    x: np.ndarray,
    y: np.ndarray,
    dim: int,
    lbdas: list,
    order: int,
    self_arrow: bool = False,
) -> list:
    """Compute the lasso estimator

    Args:
        x (np.ndarray): X samples
        y (np.ndarray): Y samples
        dim (int): dimension (number of time series)
        lbdas (list): possible values of lambda
        order (int): order of the model
        self_arrow (bool, optional): allowing or not auto-edges. If False, allow auto-edges. Defaults to False.

    Returns:
        list: list of VAR parameters for each value in lbdas
    """
    if not self_arrow:

        coeffs = []

        for i in range(dim):
            group_lasso_model = asgl.ASGL(
                model="lm", penalization="lasso", lambda1=lbdas, intercept=False
            )
            group_lasso_model.fit(x=x, y=y[:, i])
            coef = group_lasso_model.coef_
            coeffs.append(coef)
        coeffs = np.array(coeffs)
        fit_params = []
        for k in range(len(lbdas)):
            fit_params.append(coeffs[:, k, :])
        return fit_params

    else:
        coeffs = []
        for i in range(dim):
            x_dim = np.zeros((len(x), order * (dim - 1)))
            for t in range(order):
                x_dim[:, t * (dim - 1) : t * (dim - 1) + i] = x[
                    :, t * dim : t * dim + i
                ]
                x_dim[:, t * (dim - 1) + i : (t + 1) * (dim - 1)] = x[
                    :, t * dim + i + 1 : (t + 1) * dim
                ]
            group_lasso_model = asgl.ASGL(
                model="lm", penalization="lasso", lambda1=lbdas, intercept=False
            )
            group_lasso_model.fit(x=x_dim, y=y[:, i])
            coef = group_lasso_model.coef_
            for k in range(len(lbdas)):
                coefs = list(coef[k])
                for t in range(order):
                    coef[k] = complete_coeffs(coefs, i, dim, order)
            coeffs.append(coef)
        coeffs = np.array(coeffs)
        fit_params = []
        for k in range(len(lbdas)):
            fit_params.append(coeffs[:, k, :])
        return fit_params


def train_alasso(
    x: np.ndarray,
    y: np.ndarray,
    dim: int,
    lbdas: list,
    order: int,
    prior_adjacency: np.ndarray,
    self_arrow: bool = False,
) -> list:
    """Train Adaptive lasso given matrix of weights

    Args:
        x (np.ndarray): X samples
        y (np.ndarray): Y samples
        dim (int): dimension (number of time series)
        lbdas (list): possible values of lambda
        order (int): order of the model
        prior_adjacency (np.ndarray): Prior matrix defining weights of the adaptive lasso
        self_arrow (bool, optional):  allowing or not auto-edges. If False, allow auto-edges. Defaults to False.

    Returns:
        list: list of VAR parameters for each value in lbdas
    """
    if not self_arrow:

        coeffs = []

        for i in range(dim):
            group_lasso_model = asgl.ASGL(
                model="lm",
                penalization="alasso",
                lambda1=lbdas,
                lasso_weights=np.repeat(1 / prior_adjacency[:, i]),
                intercept=False,
            )
            group_lasso_model.fit(x=x, y=y[:, i])
            coef = group_lasso_model.coef_
            coeffs.append(coef)
        coeffs = np.array(coeffs)
        fit_params = []
        for k in range(len(lbdas)):
            fit_params.append(coeffs[:, k, :])
        return fit_params

    else:
        coeffs = []
        for i in range(dim):
            x_dim = np.zeros((len(x), order * (dim - 1)))
            for t in range(order):
                x_dim[:, t * (dim - 1) : t * (dim - 1) + i] = x[
                    :, t * dim : t * dim + i
                ]
                x_dim[:, t * (dim - 1) + i : (t + 1) * (dim - 1)] = x[
                    :, t * dim + i + 1 : (t + 1) * dim
                ]
            group_lasso_model = asgl.ASGL(
                model="lm",
                penalization="alasso",
                lambda1=lbdas,
                lasso_weights=np.repeat(
                    1
                    / np.hstack([prior_adjacency[i, :i], prior_adjacency[i, i + 1 :]]),
                    order,
                ),
                intercept=False,
            )
            group_lasso_model.fit(x=x_dim, y=y[:, i])
            coef = group_lasso_model.coef_
            for k in range(len(lbdas)):
                coefs = list(coef[k])
                for t in range(order):
                    coef[k] = complete_coeffs(coefs, i, dim, order)
            coeffs.append(coef)
        coeffs = np.array(coeffs)
        fit_params = []
        for k in range(len(lbdas)):
            fit_params.append(coeffs[:, k, :])
        return fit_params


def cross_val_lasso(
    X: np.ndarray,
    dim: int,
    order: int,
    size: int,
    lbda_values: list,
    return_idx: bool = False,
    print_better_idx: bool = False,
    return_errors: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, float, int]]:
    """Fit Lasso estimator performing cross validation to chose lambda

    Args:
        X (np.ndarray): multivariate time series
        dim (int): dimension
        order (int): order of the VAR model
        size (int): train size
        lbda_values (list): possible values of lambda
        return_idx (bool, optional): None . Defaults to False.
        print_better_idx (bool, optional): Print the better index of lambda. Defaults to False.
        return_errors (bool, optional): Return reconstruction error and index of the lambda selected. Defaults to False.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, float, int]]: Parameters of the VAR model
    """

    n = len(X) - order

    Xt_list = []
    for t in range(order + 1):
        Xt_list.append(X[t : n + t])

    Xt_list = np.array(Xt_list)

    nx, ny = transform_data(Xt_list[:, : size // 2])

    fit_params = train_lasso(nx, ny, dim, lbda_values, order, self_arrow=True)

    var_models = []

    for param in fit_params:
        n_params = np.zeros((order, dim, dim))
        for t in range(order):
            n_params[t] = param[:, t * dim : (t + 1) * dim]
            var_models.append(VAR(order, dim, n_params))

    errors = []

    for k in range(len(lbda_values)):
        preds = np.array(
            outsamples_predictions(var_models[k], X[size // 2 - order :], 1)
        )
        errors.append(rNMSE(preds, X[size // 2 :]))
    i = np.argmin(errors)
    preds = np.array(outsamples_predictions(var_models[i], X[size // 2 - order :], 1))
    if print_better_idx:
        print(i)
    if return_errors:
        return var_models[i].params, rNMSE(preds, X[size // 2 :]), i
    return var_models[i].params


def cross_val_alasso(
    X: np.ndarray,
    dim: int,
    order: int,
    size: int,
    prior: np.ndarray,
    lbda_values: list,
    return_idx: bool = False,
    print_better_idx: bool = False,
    return_errors: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, float, int]]:
    """Fit Lasso estimator performing cross validation to chose lambda

    Args:
        X (np.ndarray): multivariate time series
        dim (int): dimension
        order (int): order of the VAR model
        size (int): train size
        prior (np.ndarray): Symetric adjacency matrix of the prior graph
        lbda_values (list): possible values of lambda
        return_idx (bool, optional): None . Defaults to False.
        print_better_idx (bool, optional): Print the better index of lambda. Defaults to False.
        return_errors (bool, optional): Return reconstruction error and index of the lambda selected. Defaults to False.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, float, int]]: Parameters of the VAR model
    """

    n = len(X) - order

    Xt_list = []
    for t in range(order + 1):
        Xt_list.append(X[t : n + t])

    Xt_list = np.array(Xt_list)

    nx, ny = transform_data(Xt_list[:, : size // 2])

    fit_params = train_alasso(
        nx, ny, dim, lbda_values, order, prior_adjacency=prior, self_arrow=True
    )

    var_models = []

    for param in fit_params:
        n_params = np.zeros((order, dim, dim))
        for t in range(order):
            n_params[t] = param[:, t * dim : (t + 1) * dim]
            var_models.append(VAR(order, dim, n_params))

    errors = []

    for k in range(len(lbda_values)):
        preds = np.array(
            outsamples_predictions(var_models[k], X[size // 2 - order :], 1)
        )
        errors.append(rNMSE(preds, X[size // 2 :]))
    i = np.argmin(errors)
    preds = np.array(outsamples_predictions(var_models[i], X[size // 2 - order :], 1))
    if print_better_idx:
        print(i)
    if return_errors:
        return var_models[i].params, rNMSE(preds, X[size // 2 :]), i
    return var_models[i].params


def Z_sym_update(Z, A, beta, lbda, gamma, delta, alpha, eps, n_iter):
    """Update Z performing gradient descent

    Args:
        Z (_type_): Current prior
        A (_type_): Prior
        beta (_type_): Current VAR parameters
        lbda (_type_): Lambda
        gamma (_type_): Gamma
        delta (_type_): Delta
        alpha (_type_): Step coefficient
        eps (_type_): Epsilon
        n_iter (_type_): Number of step in gradient descent

    Returns:
        _type_: Return the updated version of Z
    """
    Z = Z
    for _ in range(n_iter):
        grad_Z = (
            -lbda
            * (
                delta * np.abs(beta) / Z ** (delta + 1)
                + delta * np.abs(beta.T) / Z ** (delta + 1)
            )
            + 2 * lbda / Z
            + 2 * gamma * (Z - A)
        )
        nZ = Z - eps * grad_Z
        Z = np.where(nZ > 0, nZ, 0.01)
        for i in range(1, len(Z)):
            for j in range(i):
                Z[i, j] = Z[j, i]
    return Z


def Z_sym_update_exact(Z, A, beta, lbda, gamma, delta, alpha, eps, n_iter):
    Z = Z
    for i in range(1, len(Z)):
        for j in range(i):
            coeffs = [
                2 * gamma,
                -2 * gamma * A[i, j],
                lbda,
                -lbda * (np.abs(beta[i, j]) + np.abs(beta[j, i])),
            ]
            roots = np.roots(coeffs)
            best_root = None
            root_eval = np.inf
            for root in roots:
                if root > 1e-3 and np.real(root) == root:
                    r_ev = (
                        lbda
                        * (
                            (np.abs(beta[i, j]) + np.abs(beta[j, i])) / root
                            + np.log(2 * root)
                        )
                        + gamma * (root - A[i, j]) ** 2
                        < root_eval
                    )
                    if r_ev < root_eval:
                        root_eval = r_ev
                        best_root = root
            if best_root is None:
                if 0 + 0j in roots:
                    Z[i, j] = 1e-2
                    Z[j, i] = 1e-2
            else:
                Z[i, j] = best_root
                Z[j, i] = best_root
    return Z


def alternating_minimization(
    N_step: int,
    prior: np.ndarray,
    lbda: float,
    gamma: float,
    dim: int,
    order: int,
    size: int,
    X: np.ndarray,
    n_iter: int,
    history: bool = True,
):
    """Alternating minimization to compute the MAP

    Args:
        N_step (int): Number of steps of the alternating minimization algorithm
        prior (np.ndarray): Adjacency matrix of the prior graph
        lbda (float): Lambda (hyper parameter)
        gamma (float): Gamma (hyper parameter)
        dim (int): int
        order (int): int
        size (int): int
        X (np.ndarray): Multivariate time series
        n_iter (int): Number of iterations in the gradient descent
        history (bool, optional): Save or not history of C and Z. Defaults to True.

    Returns:
        _type_: _description_
    """

    Z = np.copy(prior) + 0.001
    if history:
        errors_prior = []
        betas_errors_prior = []
        Zs_prior = [np.copy(Z)]
    for k in range(N_step):
        beta, error, i = cross_val_alasso(
            X[:size], dim, order, size, Z, [lbda], return_errors=True
        )
        Z = Z_sym_update(Z, prior, beta[0], lbda, gamma, 1, lbda, 0.02, n_iter)
        if history:
            betas_errors_prior.append(np.copy(beta[0]))
            Zs_prior.append(np.copy(Z))
            errors_prior.append(error)
    if history:
        return beta[0], Z, betas_errors_prior, Zs_prior, errors_prior
    return beta[0], Z
