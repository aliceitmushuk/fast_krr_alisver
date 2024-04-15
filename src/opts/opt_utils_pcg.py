import torch
from pykeops.torch import LazyTensor

from ..preconditioners.nystrom import Nystrom
from ..preconditioners.partial_cholesky import Pivoted_Cholesky
from .. preconditioners.falkon import Falkon
from ..kernels import _get_kernel, _get_kernels_start

def _get_kernel_matrices(x, x_tst, kernel_params, b):

    # Get kernels for training and test sets
    _,K,K_tst = _get_kernels_start(x, x_tst, kernel_params)

    return K, K_tst, x.shape[0], torch.norm(b)

def _get_kernel_matrices_inducing(x, x_tst, inducing_pts, kernel_params, b):
     # Get inducing points kernel
        x_inducing_i = LazyTensor(x[inducing_pts][:, None, :])
        x_inducing_j = LazyTensor(x[inducing_pts][None, :, :])
        K_mm = _get_kernel(x_inducing_i, x_inducing_j, kernel_params)

        # Get kernel between full training set and inducing points
        x_i = LazyTensor(x[:, None, :])
        K_nm = _get_kernel(x_i, x_inducing_j, kernel_params)

        # Get kernel for test set
        x_tst_i = LazyTensor(x_tst[:, None, :])
        K_tst = _get_kernel(x_tst_i, x_inducing_j, kernel_params)

        return K_mm, K_nm, K_tst, inducing_pts.shape[0], x.shape[0], torch.norm(b)

def _get_precond(x,n, K,kernel_params,precond_params, device):

    precond = None
    if precond_params is not None:
        if precond_params['type'] == 'nystrom':
            precond_params_sub = {k: v for k, v in precond_params.items() if k != 'type'}
            def K_Lin_Op(v): return K@v
            precond = Nystrom(device, **precond_params_sub)
            precond.update(K_Lin_Op, n)
        elif precond_params['type'] == 'pivoted_cholesky':
            precond_params_sub = {k: v for k, v in precond_params.items() if k != 'type'}
            precond = Pivoted_Cholesky(device,**precond_params_sub)
            precond.update(x,kernel_params)
    return precond

def _get_precond_inducing(K_mm,n,m,lambd,precond_params,device):
     if precond_params['type'] == 'falkon':
          precond = Falkon(device)
          precond.update(K_mm,n,m,lambd)
     return precond

def _init_pcg(a0,K_Lin_Op,b,precond):
    r = b-K_Lin_Op(a0)
    z = _apply_precond(r,precond)
    p = z.clone()
    return r,z,p

def _step_pcg(a,r,z,p,K_Lin_Op,precond):
        Kp = K_Lin_Op(p)
        r0_dot_z0 = torch.dot(r,z)
        alpha = r0_dot_z0/torch.dot(p,Kp)
        a+= alpha*p
        r-= alpha*Kp
        z = _apply_precond(r,precond)
        beta = torch.dot(r,z)/r0_dot_z0
        p = z+beta*p
        return a,r,z,p

def _apply_precond(v, precond):
    if precond is not None:
        return precond.inv_lin_op(v)
    else:
        return v