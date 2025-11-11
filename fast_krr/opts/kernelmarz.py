import torch
import math
import random
from fast_krr.opts.optimizer import Optimizer
from pykeops.torch import LazyTensor
from fast_krr.opts.utils.bcd import (
    _get_block,
)
from fast_krr.kernels.kernel_inits import (
    _get_kernel,
)

def keops_matmul(A_lazy,B_lazy,m,n):
    prod=torch.zeros(m,n)
    for i in range(n):
        prod[:,i]=A_lazy@B_lazy[i]
    return prod
    
#also returns a sum of the squared error
def _get_block_update_w_err_kaczmarz(model, w, L, block):

    xb_i = LazyTensor(model.x[block][:, None, :])
    Kbn = _get_kernel(xb_i, model.x_j, model.kernel_params)
    resids=Kbn @ w - self.b[block]
    temp1=torch.linalg.solve_triangular(L, resids, upper=False)
    temp2=torch.linalg.solve_triangular(L.t(), temp1, upper=True)
    dir=Kbn.t()@temp2
    return dir, (resids**2).sum()
def min_l2_dist(x):
    x_i = LazyTensor( x[:,None,:] )  
    x_j = LazyTensor( y[None,:,:] )

    D_ij = ((x_i - x_j)**2).sum(dim=2)
    K=2
    d_mins_raw=D_ij.Kmin(K,dim=1)   
    #just get rid of the zero element
    d_mins=d_mins_raw.sum(dim=1)
    min_dists=d_mins**0.5
    return min_dists
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
        prob_add=min(1,self.model.n/((self.i+1)*self.block_sz)*math.log(self.model.n))
        if random.random()<prob_add:
            block = _get_block(self.probs, self.probs_cpu, self.block_sz)
            xb_i = LazyTensor(self.model.x[block][:, None, :])
            Kbn = _get_kernel(xb_i, self.model.x_j, self.model.kernel_params)
            print(Kbn)
            Kbn_Kbn_T=keops_matmul(Kbn,Kbn.t(),self.block_sz,self.block_sz)
            L=torch.cholesky(Kbn_Kbn_T+proj_reg*torch.eye(self.block_sz))
            self.cache_chol.append(L)
            self.cache_blocks.append(block)
        else:
            len_cache=len(self.cache_blocks)
            ind=random.randint(0,len_cache-1)
            block=self.cache_blocks[ind]
            L=self.cache_chol[ind]
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
