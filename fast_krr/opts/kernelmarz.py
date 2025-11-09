import torch
import math
import random
from fast_krr.opts.optimizer import Optimizer
from fast_krr.opts.utils.bcd import (
    _get_block,
)

#also returns a sum of the squared error
def _get_block_update_w_err_kaczmarz(model, w, L, block):
    
    # Compute the block gradient
    gb = model._get_block_grad(w, block)
    xb_i = LazyTensor(self.x[block][:, None, :])
    Kbn = _get_kernel(xb_i, self.x_j, self.kernel_params)
    resids=Kbn @ w - self.b[block]
    temp1=torch.linalg.solve_triangular(L, resids, upper=False)
    temp2=torch.linalg.solve_triangular(L.t(), temp1, upper=True)
    dir=Kbn.t()@temp2
    return dir, (resids**2).sum()

class KernelMarz(Optimizer):
    def __init__(
        self,
        model,
        block_sz,
        sampling_method="uniform",
        proj_reg=1e-8,
        eta=None,
        p=None,
        min_dists=None,
        precond_params=None,
        accelerated=True,
        rho_stop=1e-4,
    ):
        super().__init__(model, precond_params)
        self.block_sz = block_sz
        self.eta = eta if eta is not None else 4*self.block_sz / self.model.n
        self.p = p if p is not None else 100
        self.accelerated = accelerated
        #stores the chol factors L
        self.cache_chol = []
        self.cache_blocks = []
        #right now only uniform sampling is implemented
        if sampling_method == "uniform":
            self.probs = torch.ones(self.model.n) / self.model.n
        self.probs_cpu = self.probs.cpu().numpy()
        self.i = 0

        if self.accelerated:
            self.dist_new = 0.0
            self.dist_old = 0.0
            #rho=1 means no acceleration, only start to accelerate later
            self.rho = 0.01
            self.m_old = torch.zeros(self.model.n,device=self.model.device)
            self.m_new = torch.zeros(self.model.n,device=self.model.device)
            self.temp = torch.zeros(self.model.n,device=self.model.device)
            self.ratio = 0
            self.rho_stop = rho_stop 


    def step(self):
        # Randomly select block_sz distinct indices
        if self.rho<self.rho_stop and self.i>1000:
            return
        block = _get_block(self.probs, self.probs_cpu, self.block_sz)

        # Get the update direction
        dir,sum_o_sqrerr = _get_block_update_w_err(self.model, self.model.w,L, block)

        if self.accelerated:
            self.temp += dir 
            
            self.m_new = (1-self.rho)/(1+self.rho)*(self.m_old-self.temp)
            self.model.w = self.model.w - self.temp + self.eta*self.m_new
            self.temp[:]=0
            self.m_old = self.m_new.clone()
            self.dist_new += sum_o_sqrerr
            if self.i%self.p==self.p-1:
                cnt=(self.i+1)//self.p
                a_old = cnt**math.log(cnt)
                a_new = (cnt+1)**math.log(cnt+1)
                if cnt>=2:
                    if cnt==2:
                        self.ratio = self.dist_new / self.dist_old
                    else:
                        self.ratio = self.ratio*(a_old/a_new) + self.dist_new / self.dist_old * (1 - a_old/a_new)
                    self.rho = max(0,1 - self.ratio**(1/self.p))
                self.dist_old=self.dist_new
                self.dist_new=0
        else:
            self.model.w -= dir
        self.i+=1
