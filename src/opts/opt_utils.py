import torch
from pykeops.torch import LazyTensor

from ..preconditioners.nystrom import Nystrom
from ..kernels import _get_kernel, _get_kernels_start

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

# TODO: Give a better name to this function
def _get_needed_quantities(x, x_tst, kernel_params, b, B):
    x_j, K, K_tst = _get_kernels_start(x, x_tst, kernel_params)

    b_norm = torch.norm(b)
    blocks = _get_blocks(x.shape[0], B)

    return x_j, K, K_tst, b_norm, blocks

def _get_L(K_lin_op, lambd, precond_inv_sqrt_lin_op, n, device):
    v = torch.randn(n, device=device)
    v = v / torch.linalg.norm(v)

    max_eig = None

    for _ in range(10):  # TODO: Make this a parameter or check tolerance instead
        v_old = v.clone()

        v = precond_inv_sqrt_lin_op(v)
        v = K_lin_op(v) + lambd * v
        v = precond_inv_sqrt_lin_op(v)

        max_eig = torch.dot(v_old, v)

        v = v / torch.linalg.norm(v)

    return max_eig

def _get_block_precond_L(Kb, lambd, block, precond_params, device):
    precond = None

    if precond_params['type'] == 'nystrom':
        precond = Nystrom(device, **precond_params)
        precond.update(lambda v: Kb @ v, block.shape[0])
        precond.rho = lambd + precond.S[-1]
        L = _get_L(lambda v: Kb @ v, lambd, precond.inv_sqrt_lin_op, block.shape[0], device)
    else: # No preconditioner
        L = _get_L(lambda v: Kb @ v, lambd, lambda x: x, block.shape[0], device)

    return precond, L

def _get_block_properties(x, kernel_params, lambd, blocks, precond_params, device):
    block_preconds, block_etas, block_Ls = [], [], []

    for _, block in enumerate(blocks):
        xb_i = LazyTensor(x[block][:, None, :])
        xb_j = LazyTensor(x[block][None, :, :])
        Kb = _get_kernel(xb_i, xb_j, kernel_params)

        precond, L = _get_block_precond_L(Kb, lambd, block, precond_params, device)

        block_preconds.append(precond)
        block_Ls.append(L)
        block_etas.append(1 / L)

    return block_preconds, block_etas, block_Ls

def _get_block_grad(x, x_j, kernel_params, a, b, lambd, block):
    xb_i = LazyTensor(x[block][:, None, :])
    Kbn = _get_kernel(xb_i, x_j, kernel_params)

    return Kbn @ a + lambd * a[block] - b[block]

def _get_block_update(block_idx, blocks, block_preconds, block_etas, 
                     x, x_j, kernel_params, a, b, lambd):
    
    # Get the block and its corresponding preconditioner
    block = blocks[block_idx]
    precond = block_preconds[block_idx]
    eta = block_etas[block_idx]

    # Compute the block gradient
    gb = _get_block_grad(x, x_j, kernel_params, a, b, lambd, block)

    # Apply the preconditioner
    if precond is not None:
        dir = precond.inv_lin_op(gb)
    else:
        dir = gb

    return block, eta, dir
