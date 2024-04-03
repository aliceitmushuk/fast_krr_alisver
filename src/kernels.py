import torch
from pykeops.torch import LazyTensor

def _get_rbf_kernel(x1_lazy, x2_lazy, sigma):
    D = ((x1_lazy - x2_lazy) ** 2).sum(dim=2)
    K = (-D / (2 * sigma ** 2)).exp()

    return K

def _get_l1_laplace_kernel(x1_lazy, x2_lazy, sigma):
    # Compute the L1 distance (Manhattan distance) between each pair of points
    D = (x1_lazy - x2_lazy).abs().sum(dim=2)
    # Compute the L1 Laplacian kernel
    K = (-D / sigma).exp()

    return K

# TODO: Modify to re-use computations
def _get_matern_kernel(x1_lazy, x2_lazy, sigma, nu):
    D = ((x1_lazy - x2_lazy) ** 2).sum(dim=2).sqrt()

    if nu == 0.5:
        K = (-D / sigma).exp()
    elif nu == 1.5:
        K = (1 + torch.sqrt(torch.tensor(3.0)) * D / sigma) * \
            (-(torch.sqrt(torch.tensor(3.0)) * D / sigma)).exp()
    else:  # nu == 2.5
        K = (1 + torch.sqrt(torch.tensor(5.0)) * D / sigma + 5 * D ** 2 /
             (3 * sigma ** 2)) * (-(torch.sqrt(torch.tensor(5.0)) * D / sigma)).exp()

    return K

def _check_kernel_params(kernel_params):
    if kernel_params["type"] not in ["rbf", "l1_laplace", "matern"]:
        raise ValueError("Invalid kernel type")

    if "sigma" not in kernel_params:
        raise ValueError("Missing sigma for kernel")

    if kernel_params["type"] == "matern":
        if "nu" not in kernel_params:
            raise ValueError("Missing nu for Matern kernel")
        if kernel_params["nu"] not in [0.5, 1.5, 2.5]:
            raise ValueError("nu must be 0.5, 1.5, or 2.5")

def _get_kernel(x1_lazy, x2_lazy, kernel_params):
    _check_kernel_params(kernel_params)

    if kernel_params["type"] == "rbf":
        return _get_rbf_kernel(x1_lazy, x2_lazy, kernel_params["sigma"])
    elif kernel_params["type"] == "l1_laplace":
        return _get_l1_laplace_kernel(x1_lazy, x2_lazy, kernel_params["sigma"])
    else:
        return _get_matern_kernel(x1_lazy, x2_lazy, kernel_params["sigma"], kernel_params["nu"])
    
def _get_kernels_start(x, x_tst, kernel_params):
    x_i = LazyTensor(x[:, None, :])
    x_j = LazyTensor(x[None, :, :])
    x_tst_i = LazyTensor(x_tst[:, None, :])

    K = _get_kernel(x_i, x_j, kernel_params)

    K_tst = _get_kernel(x_tst_i, x_j, kernel_params)

    return x_j, K, K_tst