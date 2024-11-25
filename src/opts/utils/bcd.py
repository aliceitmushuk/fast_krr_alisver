import torch

from .general import _get_L, _apply_precond
from ...preconditioners import preconditioner_inits as pi


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


def _get_block_precond(model, precond_params, block):
    block_lin_op, block_lin_op_reg, block_trace = model._get_block_lin_ops(block)

    update_params = None
    if precond_params is not None:
        type = precond_params["type"]
        if type == "newton":
            update_params = {"K_lin_op": block_lin_op, "n": block.shape[0]}
        elif type == "nystrom":
            update_params = {
                "K_lin_op": block_lin_op,
                "K_trace": block_trace,
                "n": block.shape[0],
            }
        elif type == "partial_cholesky":
            K_fn = model._get_kernel_fn()
            K_diag = model._get_diag(sz=block.shape[0])
            blk_size = precond_params.get("blk_size", None)
            update_params = {
                "K_fn": K_fn,
                "K_diag": K_diag,
                "x": model.x[block],
                "blk_size": blk_size,
            }
    precond = pi._get_precond(precond_params, update_params, model.lambd, model.device)

    return precond, block_lin_op_reg


def _get_block_precond_L(model, precond_params, block):
    precond, block_lin_op_reg = _get_block_precond(model, precond_params, block)
    L = _get_L(block_lin_op_reg, precond, block.shape[0], model.device)

    return precond, L


def _get_block_properties(model, precond_params, blocks, no_store_precond):
    block_preconds, block_etas, block_Ls = [], [], []

    for block in blocks:
        precond, L = _get_block_precond_L(model, precond_params, block)

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
