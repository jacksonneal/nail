import time
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


def torch_sle_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Calculate the Squared Log Error loss."""
    return 1 / 2 * (torch.log1p(y_pred) - torch.log1p(y_true)) ** 2


def torch_autodiff_grad_hess(
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    y_true: np.ndarray,
    y_pred: np.ndarray,
):
    """Perform automatic differentiation to get the
    Gradient and the Hessian of `loss_function`."""
    y_true = torch.tensor(y_true, dtype=torch.float, requires_grad=False)
    y_pred = torch.tensor(y_pred, dtype=torch.float, requires_grad=True)
    loss_function_sum = lambda y_pred: loss_function(y_true, y_pred).sum()

    loss_function_sum(y_pred).backward()
    grad = y_pred.grad

    hess_matrix = torch.autograd.functional.hessian(
        loss_function_sum, y_pred, vectorize=True
    )
    hess = torch.diagonal(hess_matrix)

    return grad, hess


def jax_sle_loss(y_true: np.ndarray, y_pred: np.ndarray):
    """Calculate the Squared Log Error loss."""
    return 1 / 2 * (jnp.log1p(y_pred) - jnp.log1p(y_true)) ** 2


def hvp(f, inputs, vectors):
    """Hessian-vector product."""
    return jax.jvp(jax.grad(f), inputs, vectors)[1]


def jax_autodiff_grad_hess(
    loss_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y_true: np.ndarray,
    y_pred: np.ndarray,
):
    """Perform automatic differentiation to get the
    Gradient and the Hessian of `loss_function`."""
    loss_function_sum = lambda y_pred: loss_function(y_true, y_pred).sum()

    grad_fn = jax.grad(loss_function_sum)
    grad = grad_fn(y_pred)

    hess = hvp(loss_function_sum, (y_pred,), (jnp.ones_like(y_pred),))

    return grad, hess


X, y = datasets.fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.2, random_state=0
)
print(X.shape)

start = time.monotonic()
# torch_objective = partial(torch_autodiff_grad_hess, torch_sle_loss)
jax_objective = jax.jit(partial(jax_autodiff_grad_hess, jax_sle_loss))

reg = XGBRegressor(objective=jax_objective, n_estimators=100)
reg.fit(X_train, y_train)
end = time.monotonic()

print(f"duration {end - start:.2f}s")
