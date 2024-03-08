import time
from pykeops.torch import LazyTensor
import torch
import wandb

from .kernels import get_kernel, get_kernels_start

# TODO: Give a better name to this function
def get_needed_quantities(x, x_tst, kernel_params, b, B):
    x_j, K, K_tst = get_kernels_start(x, x_tst, kernel_params)

    b_norm = torch.norm(b)
    blocks = get_blocks(x.shape[0], B)

    return x_j, K, K_tst, b_norm, blocks

def rand_nys_appx(K, n, r, device):
    # Calculate sketch
    Phi = torch.randn((n, r), device=device) / (n ** 0.5)
    Phi = torch.linalg.qr(Phi, mode='reduced')[0]

    Y = K @ Phi

    # Calculate shift
    shift = torch.finfo(Y.dtype).eps
    Y_shifted = Y + shift * Phi

    # Calculate Phi^T * K * Phi (w/ shift) for Cholesky
    choleskytarget = torch.mm(Phi.t(), Y_shifted)

    # Perform Cholesky decomposition
    C = torch.linalg.cholesky(choleskytarget)

    B = torch.linalg.solve_triangular(C.t(), Y_shifted, upper=True, left=False)
    U, S, _ = torch.linalg.svd(B, full_matrices=False)
    S = torch.max(torch.square(S) - shift, torch.tensor(0.0))

    return U, S

def get_blocks(n, B):
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

def get_L(K, lambd, U, S, rho):
    n = U.shape[0]
    v = torch.randn(n, device=U.device)
    v = v / torch.linalg.norm(v)

    max_eig = None

    for _ in range(10):  # TODO: Make this a parameter or check tolerance instead
        v_old = v.clone()

        UTv = U.t() @ v
        v = U @ (UTv / ((S + rho) ** (0.5))) + 1/(rho ** 0.5) * (v - U @ UTv)

        v = K @ v + lambd * v

        UTv = U.t() @ v
        v = U @ (UTv / ((S + rho) ** (0.5))) + 1/(rho ** 0.5) * (v - U @ UTv)

        max_eig = torch.dot(v_old, v)

        v = v / torch.linalg.norm(v)

    return max_eig

def get_block_nys_precond_L(Kb, lambd, block, r, device):
    U, S = rand_nys_appx(Kb, block.shape[0], r, device)
    rho = lambd + S[-1]
    L = get_L(Kb, lambd, U, S, rho)

    return U, S, rho, L

def get_block_properties(x, blocks, kernel_params, lambd, r, device):
    block_preconds, block_etas, block_Ls = [], [], []

    # Compute randomized Nystrom approximation corresponding to each block
    for _, block in enumerate(blocks):
        xb_i = LazyTensor(x[block][:, None, :])
        xb_j = LazyTensor(x[block][None, :, :])
        Kb = get_kernel(xb_i, xb_j, kernel_params)

        U, S, rho, L = get_block_nys_precond_L(Kb, lambd, block, r, device)

        block_preconds.append((U, S, rho))
        block_Ls.append(L)
        block_etas.append(1 / L)

    return block_preconds, block_etas, block_Ls

def get_block_grad(x, x_j, kernel_params, a, b, lambd, block):
    xb_i = LazyTensor(x[block][:, None, :])
    Kbn = get_kernel(xb_i, x_j, kernel_params)

    return Kbn @ a + lambd * a[block] - b[block]

def apply_nys_precond(U, S, rho, g):
    UTg = U.t() @ g
    dir = U @ (UTg / (S + rho)) + 1/rho * (g - U @ UTg)

    return dir

def get_block_update(block_idx, blocks, block_preconds, block_etas, 
                     x, x_j, kernel_params, a, b, lambd):
    # Get the block and its corresponding preconditioner
    block = blocks[block_idx]
    U, S, rho = block_preconds[block_idx]
    eta = block_etas[block_idx]

    # Compute block gradient
    gb = get_block_grad(x, x_j, kernel_params, a, b, lambd, block)

    # Apply preconditioner
    dir = apply_nys_precond(U, S, rho, gb)

    return block, eta, dir

def compute_metrics_dict(K, K_tst, a, b, b_tst, lambd, b_norm, task):
    residual = K @ a + lambd * a - b
    rel_residual = torch.norm(residual) / b_norm
    loss = 1/2 * torch.dot(a, residual - b)

    test_metric_name = 'test_acc' if task == 'classification' else 'test_mse'
    if task == 'classification':
        test_metric = torch.sum(torch.sign(K_tst @ a) == b_tst) / b_tst.shape[0]
    else:
        test_metric = 1/2 * torch.norm(K_tst @ a - b_tst) ** 2 / b_tst.shape[0]

    return {'rel_residual': rel_residual, test_metric_name: test_metric,
             'train_loss': loss}
    # return {test_metric_name: test_metric}

def compute_and_log_metrics(K, K_tst, y, b, b_tst, lambd, b_norm, iter_time,
                             task, i, log_freq):
    iter_time_dict = {'iter_time': iter_time}
    if (i + 1) % log_freq == 0:
        wandb.log(iter_time_dict | 
                    compute_metrics_dict(K, K_tst, y, b, b_tst, lambd, b_norm, task))
    else:
        wandb.log(iter_time_dict)

def bcd(x, b, x_tst, b_tst, kernel_params, lambd, task, a0, B, r, max_iter, log_freq, device):
    x_j, K, K_tst, b_norm, blocks = get_needed_quantities(x, x_tst, kernel_params, b, B)

    start_time = time.time()

    block_preconds, block_etas, _ = get_block_properties(x, blocks, 
                                                kernel_params, lambd, r, device)

    a = a0.clone()
    iter_time = time.time() - start_time

    compute_and_log_metrics(K, K_tst, a, b, b_tst, lambd, b_norm, iter_time,
                            task, -1, log_freq)

    for i in range(max_iter):
        start_time = time.time()

        # Randomly select a block
        block_idx = torch.randint(B, (1,))

        # Get the block, step size, and update direction
        block, eta, dir = get_block_update(block_idx, blocks, block_preconds,
                                           block_etas, x, x_j, kernel_params,
                                             a, b, lambd)

        # Update block
        a[block] -= eta * dir

        iter_time = time.time() - start_time

        compute_and_log_metrics(K, K_tst, a, b, b_tst, lambd, b_norm, iter_time, 
                                task, i, log_freq)

    return a

def abcd(x, b, x_tst, b_tst, kernel_params, lambd, task, a0, B, r, max_iter, log_freq, device):
    x_j, K, K_tst, b_norm, blocks = get_needed_quantities(x, x_tst, kernel_params, b, B)

    alpha = 1/2 # Controls acceleration

    start_time = time.time()

    block_preconds, block_etas, block_Ls = get_block_properties(x, blocks,
                                                    kernel_params, lambd, r, device)
    
    S_alpha = sum([L ** alpha for L in block_Ls])

    block_probs = torch.tensor([L ** alpha / S_alpha for L in block_Ls])
    sampling_dist = torch.distributions.categorical.Categorical(block_probs)
    tau = 2 / (1 + (4 * (S_alpha ** 2) / lambd + 1) ** 0.5)
    gamma = 1 / (tau * S_alpha ** 2)

    a = a0.clone()
    y = a0.clone()
    z = a0.clone()

    iter_time = time.time() - start_time

    compute_and_log_metrics(K, K_tst, y, b, b_tst, lambd, b_norm, iter_time,
                            task, -1, log_freq)

    for i in range(max_iter):
        start_time = time.time()

        # Randomly select a block
        block_idx = sampling_dist.sample()

        # Get the block, step size, and update direction
        block, eta, dir = get_block_update(block_idx, blocks, block_preconds,
                                           block_etas, x, x_j, kernel_params,
                                           a, b, lambd)

        # Update y
        y = a.clone()
        y[block] -= eta * dir

        # Update z
        z = (1 / (1 + gamma * lambd)) * (z + gamma * lambd * a)
        z[block] -= (1 / (1 + gamma * lambd)) * \
            gamma / block_probs[block_idx] * dir

        # Update x
        a = tau * z + (1 - tau) * y

        iter_time = time.time() - start_time

        compute_and_log_metrics(K, K_tst, y, b, b_tst, lambd, b_norm, iter_time,
                                task, i, log_freq)

    return y