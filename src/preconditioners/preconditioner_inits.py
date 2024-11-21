from .falkon import Falkon
from .newton import Newton
from .nystrom import Nystrom
from .partial_cholesky import PartialCholesky

PRECOND_CLASSES = {
    "falkon": Falkon,
    "newton": Newton,
    "nystrom": Nystrom,
    "partial_cholesky": PartialCholesky,
}


def _get_precond(precond_params, update_params, device):
    if precond_params is None:
        return None

    precond_params_sub = {
        k: v for k, v in precond_params.items() if k != "type" and k != "blk_size"
    }
    precond = PRECOND_CLASSES[precond_params["type"]](device, **precond_params_sub)
    precond.update(**update_params)
    return precond


def _set_nystrom_damping(precond, precond_params, lambd):
    if isinstance(precond, Nystrom):
        if isinstance(precond_params["rho"], float):
            precond.rho = precond_params["rho"]
        elif precond_params["rho"] == "regularization":
            precond.rho = lambd
        elif precond_params["rho"] == "damped":
            precond.rho = lambd + precond.S[-1]
