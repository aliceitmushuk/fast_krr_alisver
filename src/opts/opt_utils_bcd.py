import torch

from .opt_utils import _get_L, _apply_precond
from ..preconditioners.nystrom import Nystrom
from ..preconditioners.newton import Newton


def _get_blocks(n, B):
    # Permute the indices
    idx = torch.randperm(n)

    # Partition the indices into B blocks of roughly equal size
    # Do this by first computing the block size then making a list of block sizes
    block_size = n // B
    remainder = n % B
    sizes = [block_size] * B

    for i in range(remainder):
        sizes[i] += 1

    blocks = torch.split(idx, sizes)

    return blocks


def _get_block_precond(model, block, precond_params):
    block_lin_op, block_lin_op_reg, block_trace = model._get_block_lin_ops(block)

    precond = None

    if precond_params is not None:
        if precond_params["type"] == "nystrom":
            precond_params_sub = {
                k: v for k, v in precond_params.items() if k != "type"
            }
            precond = Nystrom(model.device, **precond_params_sub)
            precond.update(block_lin_op, block_trace, block.shape[0])

            # Automatically set rho to lambda + S[-1] if not provided
            precond.rho = (
                model.lambd + precond.S[-1]
                if "rho" not in precond_params_sub
                else precond_params_sub["rho"]
            )
        elif precond_params["type"] == "newton":
            precond = Newton(model.device)
            precond.update(block_lin_op_reg, block.shape[0])

    return precond, block_lin_op_reg


def _get_block_precond_L(model, block, precond_params):
    precond, block_lin_op_reg = _get_block_precond(model, block, precond_params)

    if precond is not None:
        if isinstance(precond, Nystrom):
            L = _get_L(
                block_lin_op_reg, precond.inv_sqrt_lin_op, block.shape[0], model.device
            )
        elif isinstance(precond, Newton):
            L = 1.0
    else:  # No preconditioner
        L = _get_L(block_lin_op_reg, lambda x: x, block.shape[0], model.device)

    return precond, L


def _get_block_properties(model, blocks, precond_params, no_store_precond):
    block_preconds, block_etas, block_Ls = [], [], []

    for block in blocks:
        precond, L = _get_block_precond_L(model, block, precond_params)

        if not no_store_precond:
            block_preconds.append(precond)
        block_Ls.append(L)
        block_etas.append(1 / L)

    return block_preconds, block_etas, block_Ls


def _get_block_update(model, w, block, precond):
    # Compute the block gradient
    gb = model._get_block_grad(w, block)

    # Apply the preconditioner
    dir = _apply_precond(gb, precond)

    return dir
