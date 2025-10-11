import torch

from fast_krr.opts.optimizer import Optimizer
from fast_krr.opts.utils.general import _get_leverage_scores
from fast_krr.opts.utils.bcd import (
    _get_block,
    _get_block_properties,
)

#also returns a sum of the squared error
def _get_block_update_w_err(model, w, block, precond):
    
    # Compute the block gradient
    gb = model._get_block_grad(w, block)
    resids=gb-model.lambd * model.w[block]
    
    # Apply the preconditioner
    dir = _apply_precond(gb, precond)
    return dir, (resids**2).sum()

class ASkotchV3(Optimizer):
    def __init__(
        self,
        model,
        block_sz,
        sampling_method="uniform",
        precond_params=None,
        eta=None
        p=None
        accelerated=True,
    ):
        super().__init__(model, precond_params)
        self.block_sz = block_sz
        self.eta = eta if eta is not None else self.block_sz / (2*self.model.n)
        self.p = p if p is not None else 100
        self.accelerated = accelerated

        # TODO(pratik): check that nu > mu and mu * nu <= 1

        # Compute sampling probabilities
        if sampling_method == "rls":
            leverage_scores = _get_leverage_scores(
                model=self.model,
                size_final=int(self.model.n**0.5),
                lam_final=self.model.lambd,
                rls_oversample_param=5,
            )
            self.probs = leverage_scores / torch.sum(leverage_scores)
        elif sampling_method == "uniform":
            self.probs = torch.ones(self.model.n) / self.model.n
        self.probs_cpu = self.probs.cpu().numpy()
        self.i = 0

        if self.accelerated:
            self.dist_new = 0.0
            self.dist_old = 0.0
            self.rho = 0
            self.m_old = torch.zeros(self.model.n)
            self.m_new = torch.zeros(self.model.n)


    def step(self):
        # Randomly select block_sz distinct indices
        block = _get_block(self.probs, self.probs_cpu, self.block_sz)

        # Compute block preconditioner and learning rate
        block_precond, block_eta, _ = _get_block_properties(
            self.model, self.precond_params, [block], False
        )
        block_precond = block_precond[0]
        block_eta = block_eta[0]

        # Get the update direction
        # Update direction is computed at self.y if accelerated, else at self.model.w
        eval_loc = self.y if self.accelerated else self.model.w
        dir,sum_o_sqrerr = _get_block_update_w_err(self.model, eval_loc, block, block_precond)

        if self.accelerated:
            self.model.w = self.y.clone()
            self.model.w[block] -= block_eta * dir
            self.m_new = (1-rho)/(1+rho)*(self.m_old-dir)
            self.model.w = self.model.w - dir + eta*self.m_new
            
        else:
            self.model.w[block] -= block_eta * dir
