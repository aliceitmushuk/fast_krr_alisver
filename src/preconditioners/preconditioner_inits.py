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
