import torch
import numpy as np

from opt_utils_pcg import _get_precond, _get_kernel_matrices, _init_pcg,_step_pcg 



class PCG():
    def __init__(self,precond_params):
        self.precond_params = precond_params

    def run(self, x, b, x_tst, b_tst, kernel_params,lambd,task,
            a0,max_iter,device, logger = None, pcg_tol = 10**-6, Verbose = False):
       K, K_tst, n, b_norm =  _get_kernel_matrices(x,x_tst,kernel_params,b)

       if logger is not None:
            logger_enabled = True
       else:
            logger_enabled = False

       precond = _get_precond(x,n,K,kernel_params,self.precond_params,device)

       a = a0.clone()
       def K_Lin_Op(v): return K@v+lambd*v

       r,z,p = _init_pcg(a,K_Lin_Op,b,precond)

       for i in range(max_iter):
           a,r,z,p = _step_pcg(a,r,z,p,K_Lin_Op,precond)
           
           if logger_enabled:
                logger.compute_log_reset(K_Lin_Op, K_tst, a, b, b_tst, b_norm, task, i, False)
           
           r_norm = torch.linalg.norm(r)

           if Verbose:
              print('Current PCG residual:' + repr(r_norm))

           if torch.linalg.norm(r_norm)<= pcg_tol:
            print('PCG has converged with residual:'+repr(r_norm))
            break
       if r_norm> pcg_tol:
        print('PCG has reached max number of iterations without reaching convergence tolerance:' +repr(r_norm))
           
       return a 