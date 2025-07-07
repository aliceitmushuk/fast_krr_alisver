from pykeops.torch import LazyTensor

from fast_krr.kernels.l1_laplace import L1Laplace
from fast_krr.kernels.matern import Matern
from fast_krr.kernels.rbf import Rbf
from fast_krr.kernels.rbf_invquad import Rbf_invquad
KERNEL_CLASSES = {
    "rbf": Rbf,
    "l1_laplace": L1Laplace,
    "matern": Matern,
    "rbf_invquad": Rbf_invquad
}


def _get_kernel_type(kernel_params):
    return KERNEL_CLASSES[kernel_params["type"]]


def _get_kernel(x1_lazy, x2_lazy, kernel_params):
    ker_type = _get_kernel_type(kernel_params)

    kernel_params_copy = kernel_params.copy()
    kernel_params_copy.pop("type")
    return ker_type._get_kernel(x1_lazy, x2_lazy, kernel_params_copy)


def _get_kernels_start(x, x_tst, kernel_params, Ktr_needed=True):
    x_i = LazyTensor(x[:, None, :])
    x_j = LazyTensor(x[None, :, :])

    K = None
    if Ktr_needed:
        K = _get_kernel(x_i, x_j, kernel_params)

    # NOTE(pratik): we set x_tst to None when computing the projection in EigenPro4
    K_tst = None
    if x_tst is not None:
        x_tst_i = LazyTensor(x_tst[:, None, :])
        K_tst = _get_kernel(x_tst_i, x_j, kernel_params)

    return x_j, K, K_tst


def _get_row(x_i, x, kernel_params):
    ker_type = _get_kernel_type(kernel_params)
    return ker_type._get_row(x_i, x, kernel_params)


def _get_trace(n, kernel_params):
    ker_type = _get_kernel_type(kernel_params)
    return ker_type._get_trace(n)


def _get_diag(n, kernel_params):
    ker_type = _get_kernel_type(kernel_params)
    return ker_type._get_diag(n)
