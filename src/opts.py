import time
from pykeops.torch import LazyTensor
import torch
import wandb

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

# For RBF kernel only -- need to generalize
def bcd(x, b, sigma, lambd, x_tst, b_tst, a0, B, r, max_iter, device):
    x_j = LazyTensor(x[None, :, :])
    x_tst_i = LazyTensor(x_tst[:, None, :])

    n = x.shape[0]
    blocks = get_blocks(n, B)
    block_preconds = []
    block_rhos = []
    block_etas = []

    b_norm = torch.norm(b)

    start_time = time.time()

    # Compute randomized Nystrom approximation corresponding to each block
    for _, block in enumerate(blocks):
        xb_i = LazyTensor(x[block][:, None, :])
        xb_j = LazyTensor(x[block][None, :, :])
        Db = ((xb_i - xb_j) ** 2).sum(dim=2)
        Kb = (-Db / (2 * sigma ** 2)).exp()

        U, S = rand_nys_appx(Kb, block.shape[0], r, device)
        rho = lambd + S[-1]
        block_preconds.append((U, S))
        block_rhos.append(rho)
        block_etas.append(2 / get_L(Kb, lambd, U, S, rho))

    a = a0.clone()
    elapsed_time = time.time() - start_time

    Ka = torch.zeros_like(a)
    K_tsta = torch.zeros_like(b_tst)

    residual = Ka + lambd * a - b
    test_acc = torch.sum(torch.sign(K_tsta) == b_tst) / b_tst.shape[0]
    wandb.log({'elapsed_time': elapsed_time,
                'residual': torch.norm(residual) / b_norm,
                'test_acc': test_acc})

    for _ in range(max_iter):
        iter_time = 0
        start_time = time.time()

        # Randomly select a block
        block_idx = torch.randint(B, (1,))

        # Get the block and its corresponding preconditioner
        block = blocks[block_idx]
        U, S = block_preconds[block_idx]
        rho = block_rhos[block_idx]
        eta = block_etas[block_idx]

        # Compute block gradient
        xb_i = LazyTensor(x[block][:, None, :])
        Dbn = ((xb_i - x_j) ** 2).sum(dim=2)
        Kbn = (-Dbn / (2 * sigma ** 2)).exp()

        gb = Kbn @ a + lambd * a[block] - b[block]

        iter_time += time.time() - start_time

        # Get subset of kernel matrix for test acc
        xb_j = LazyTensor(x[block][None, :, :])
        D_tstnb = ((x_tst_i - xb_j) ** 2).sum(dim=2)
        K_tstnb = (-D_tstnb / (2 * sigma ** 2)).exp()

        start_time = time.time()

        # Apply preconditioner
        UTgb = U.t() @ gb
        dir = U @ (UTgb / (S + rho)) + 1/rho * (gb - U @ UTgb)

        # Update block
        a[block] -= eta * dir

        iter_time += time.time() - start_time

        Ka -= Kbn.t() @ (eta * dir)
        K_tsta -= K_tstnb @ (eta * dir)

        # Update residual
        residual = Ka + lambd * a - b
        test_acc = torch.sum(torch.sign(K_tsta) == b_tst) / b_tst.shape[0]
        wandb.log({'elapsed_time': iter_time,
               'residual': torch.norm(residual) / b_norm,
               'test_acc': test_acc})


    return a

# For RBF kernel only -- need to generalize
def abcd(x, b, sigma, lambd, x_tst, b_tst, a0, B, r, max_iter, device):
    x_j = LazyTensor(x[None, :, :])
    x_tst_i = LazyTensor(x_tst[:, None, :])

    n = x.shape[0]
    blocks = get_blocks(n, B)
    block_preconds = []
    block_rhos = []
    block_Ls = []
    block_etas = []

    b_norm = torch.norm(b)

    alpha = 1/2
    S_alpha = 0

    start_time = time.time()

    # Compute randomized Nystrom approximation corresponding to each block
    for _, block in enumerate(blocks):
        xb_i = LazyTensor(x[block][:, None, :])
        xb_j = LazyTensor(x[block][None, :, :])
        Db = ((xb_i - xb_j) ** 2).sum(dim=2)
        Kb = (-Db / (2 * sigma ** 2)).exp()

        U, S = rand_nys_appx(Kb, block.shape[0], r, device)
        rho = lambd + S[-1]
        block_preconds.append((U, S))
        block_rhos.append(rho)

        L = get_L(Kb, lambd, U, S, rho)
        S_alpha += L ** alpha
        block_Ls.append(L)
        block_etas.append(2 / L)

    block_probs = torch.tensor([L ** alpha / S_alpha for L in block_Ls])
    sampling_dist = torch.distributions.categorical.Categorical(block_probs)
    tau = 2 / (1 + (4 * (S_alpha ** 2) / lambd + 1) ** 0.5)
    gamma = 1 / (tau * S_alpha ** 2)

    a = a0.clone()
    y = a0.clone()
    z = a0.clone()

    elapsed_time = time.time() - start_time

    # NOTE: This only works if a0 = 0
    Ka = torch.zeros_like(a)
    Ky = torch.zeros_like(a)
    Kz = torch.zeros_like(a)

    K_tsta = torch.zeros_like(b_tst)
    K_tsty = torch.zeros_like(b_tst)
    K_tstz = torch.zeros_like(b_tst)

    residual = Ky + lambd * y - b
    test_acc = torch.sum(torch.sign(K_tsty) == b_tst) / b_tst.shape[0]
    wandb.log({'elapsed_time': elapsed_time,
            'residual': torch.norm(residual) / b_norm,
            'test_acc': test_acc})

    for _ in range(max_iter):
        iter_time = 0
        start_time = time.time()

        # Randomly select a block
        block_idx = sampling_dist.sample()

        # Get the block and its corresponding preconditioner
        block = blocks[block_idx]
        U, S = block_preconds[block_idx]
        rho = block_rhos[block_idx]
        eta = block_etas[block_idx]

        # Compute block gradient
        xb_i = LazyTensor(x[block][:, None, :])
        Dbn = ((xb_i - x_j) ** 2).sum(dim=2)
        Kbn = (-Dbn / (2 * sigma ** 2)).exp()

        gb = Kbn @ a + lambd * a[block] - b[block]

        iter_time += time.time() - start_time

        # Get subset of kernel matrix for test acc
        xb_j = LazyTensor(x[block][None, :, :])
        D_tstnb = ((x_tst_i - xb_j) ** 2).sum(dim=2)
        K_tstnb = (-D_tstnb / (2 * sigma ** 2)).exp()

        start_time = time.time()

        # Apply preconditioner
        UTgb = U.t() @ gb
        dir = U @ (UTgb / (S + rho)) + 1/rho * (gb - U @ UTgb)

        # Update y
        y = a.clone()
        y[block] -= eta * dir

        iter_time += time.time() - start_time

        Kdir = Kbn.t() @ dir
        Ky = Ka - eta * Kdir
        K_tstdir = K_tstnb @ dir
        K_tsty = K_tsta - eta * K_tstdir

        start_time = time.time()

        # Update z
        z = (1 / (1 + gamma * lambd)) * (z + gamma * lambd * a)
        z[block] -= (1 / (1 + gamma * lambd)) * \
            gamma / block_probs[block_idx] * dir

        iter_time += time.time() - start_time

        Kz = (1 / (1 + gamma * lambd)) * (Kz + gamma * lambd *
                                          Ka - gamma / block_probs[block_idx] * Kdir)
        K_tstz = (1 / (1 + gamma * lambd)) * (K_tstz + gamma *
                                              lambd * K_tsta - gamma / block_probs[block_idx] * K_tstdir)

        start_time = time.time()

        # Update x
        a = tau * z + (1 - tau) * y

        iter_time += time.time() - start_time

        Ka = tau * Kz + (1 - tau) * Ky
        K_tsta = tau * K_tstz + (1 - tau) * K_tsty

        # Update residual
        residual = Ky + lambd * y - b
        test_acc = torch.sum(torch.sign(K_tsty) == b_tst) / b_tst.shape[0]
        wandb.log({'elapsed_time': iter_time,
               'residual': torch.norm(residual) / b_norm,
               'test_acc': test_acc})

    return y