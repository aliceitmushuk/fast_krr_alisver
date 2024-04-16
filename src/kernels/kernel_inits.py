from pykeops.torch import LazyTensor

from .l1_laplace import L1Laplace
from .matern import Matern
from .rbf import Rbf

def _get_kernel_type(kernel_params):
    ker_type = None

    if kernel_params["type"] == "rbf":
        ker_type = Rbf
    elif kernel_params["type"] == "l1_laplace":
        ker_type = L1Laplace
    else:
        ker_type = Matern

    return ker_type

def _get_kernel(x1_lazy, x2_lazy, kernel_params):
    ker_type = _get_kernel_type(kernel_params)

    kernel_params_copy = kernel_params.copy()
    kernel_params_copy.pop("type")
    return ker_type(x1_lazy, x2_lazy, kernel_params_copy).K


def _get_kernels_start(x, x_tst, kernel_params):
    x_i = LazyTensor(x[:, None, :])
    x_j = LazyTensor(x[None, :, :])
    x_tst_i = LazyTensor(x_tst[:, None, :])

    K = _get_kernel(x_i, x_j, kernel_params)

    K_tst = _get_kernel(x_tst_i, x_j, kernel_params)

    return x_j, K, K_tst
