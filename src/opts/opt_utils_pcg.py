import torch

from ..preconditioners.nystrom import Nystrom
from ..preconditioners.partial_cholesky import Pivoted_Cholesky
from ..kernels import _get_kernels_start

def _get_kernel_matrices(x, x_tst, kernel_params, b):

    # Get kernels for training and test sets
    _,K,K_tst = _get_kernels_start(x, x_tst, kernel_params)

    return K, K_tst, x.shape[0], torch.norm(b)

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