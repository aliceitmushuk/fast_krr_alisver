import torch

from .opt_utils import _get_L
from ..preconditioners.nystrom import Nystrom


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


def _get_block_precond_L(
    block_lin_op, block_lin_op_reg, lambd, block, precond_params, device
):
    precond = None

    if precond_params is not None:
        if precond_params["type"] == "nystrom":
            precond_params_sub = {
                k: v for k, v in precond_params.items() if k != "type"
            }
            precond = Nystrom(device, **precond_params_sub)
            precond.update(block_lin_op, block.shape[0])

            # Automatically set rho to lambda + S[-1] if not provided
            precond.rho = (
                lambd + precond.S[-1]
                if "rho" not in precond_params_sub
                else precond_params_sub["rho"]
            )
            L = _get_L(
                block_lin_op_reg, precond.inv_sqrt_lin_op, block.shape[0], device
            )
    else:  # No preconditioner
        L = _get_L(block_lin_op_reg, lambda x: x, block.shape[0], device)

    return precond, L


def _get_block_properties(model, blocks, precond_params):
    block_preconds, block_etas, block_Ls = [], [], []

    for _, block in enumerate(blocks):
        Kb_lin_op, Kb_lin_op_reg = model._get_block_lin_ops(block)

        precond, L = _get_block_precond_L(
            Kb_lin_op, Kb_lin_op_reg, model.lambd, block, precond_params, model.device
        )

        block_preconds.append(precond)
        block_Ls.append(L)
        block_etas.append(1 / L)

    return block_preconds, block_etas, block_Ls


def _get_block_update(model, w, block_idx, blocks, block_preconds, block_etas):
    # Get the block and its corresponding preconditioner
    block = blocks[block_idx]
    precond = block_preconds[block_idx]
    eta = block_etas[block_idx]

    # Compute the block gradient
    gb = model._get_block_grad(w, block)

    # Apply the preconditioner
    if precond is not None:
        dir = precond.inv_lin_op(gb)
    else:
        dir = gb

    return block, eta, dir
