"""
Implementation of BLESS algorithm for approximate ridge leverage score sampling.
Adapted from the original implementation at
https://github.com/guilgautier/DPPy/blob/master/dppy/bless.py.
"""

from collections import namedtuple

import torch


CentersDictionary = namedtuple(
    "CentersDictionary", ("idx", "X", "probs", "lam", "rls_oversample")
)


def _stable_filter(eigenvec, eigenval):
    """Given eigendecomposition of a PSD matrix, compute a reduced (thin) version
    containing only stable eigenvalues."""
    n = eigenvec.shape[0]

    if eigenvec.shape != (n, n) or eigenval.shape != (n,):
        raise ValueError(
            f"Array sizes of {eigenvec.shape} eigenvectors and \
                {eigenval.shape} eigenvalues do not match"
        )

    # Threshold formula, similar to numpy/scipy's pinv2 implementation
    thresh = (
        torch.abs(eigenval).max()
        * max(eigenval.shape)
        * torch.finfo(eigenval.dtype).eps
    )
    stable_eig = ~torch.isclose(
        eigenval, torch.tensor(0.0, dtype=eigenval.dtype), atol=thresh
    )

    if torch.any(eigenval <= -thresh):
        raise ValueError(
            f"Some eigenvalues of a PSD matrix are negative, this should never happen."
            f"Minimum eig: {torch.min(eigenval).item()}"
        )

    m = stable_eig.sum().item()
    eigenvec_thin = eigenvec[:, stable_eig]
    eigenval_thin = eigenval[stable_eig]

    if eigenvec_thin.shape != (n, m) or eigenval_thin.shape != (m,):
        raise ValueError(
            f"Array sizes of {eigenvec_thin.shape} eigenvectors and \
                  {eigenval_thin.shape} eigenvalues do not match"
        )

    return eigenvec_thin, eigenval_thin


def _stable_invert_root(eigenvec, eigenval):
    """Given eigendecomposition of a PSD matrix, compute a representation
    of the pseudo-inverse square root of the matrix using numerically stable operations.
    In particular, eigenvalues which are near-zero and the associated eigenvectors
    are dropped from the pseudo-inverse.
    """
    eigenvec_thin, eigenval_thin = _stable_filter(eigenvec, eigenval)

    eigenval_thin_inv_root = 1 / torch.sqrt(eigenval_thin)

    return eigenvec_thin, eigenval_thin_inv_root


def _estimate_rls_bless(D, X, K_fn, K_diag_fn, lam_new):
    diag_norm = K_diag_fn(X.shape[0]).to(X.device)

    # (m x n) kernel matrix between samples in dictionary and dataset X
    get_row = True if D.X.shape[0] == 1 else False
    K_DU = K_fn(D.X, X, get_row=get_row)
    K_DD = K_fn(D.X, D.X, get_row=get_row)
    K_DD = K_DD @ torch.eye(D.X.shape[0], device=X.device)

    U_DD, S_DD, _ = torch.linalg.svd(K_DD + lam_new * torch.diag(D.probs))
    U_DD, S_root_inv_DD = _stable_invert_root(U_DD, S_DD)

    E = S_root_inv_DD.unsqueeze(1) * U_DD.T

    # compute (X'X + lam*S^(-2))^(-1/2)XX'
    X_precond = (K_DU.T @ E.T).T

    # the diagonal entries of XX'(X'X + lam*S^(-2))^(-1)XX' are just the squared
    # ell-2 norm of the columns of (X'X + lam*S^(-2))^(-1/2)XX'
    rls_estimate = (diag_norm - X_precond.pow(2).sum(dim=0)) / lam_new

    if torch.any(rls_estimate < 0.0):
        raise ValueError(
            "Some estimated RLS is negative, this should never happen. "
            f"Min prob: {torch.min(rls_estimate).item()}"
        )

    return rls_estimate


def _reduce_lambda(
    X_data,
    K_fn,
    K_diag_fn,
    intermediate_dict_bless,
    lam_new,
    rls_oversample_parameter=None,
):
    n, _ = X_data.shape

    if rls_oversample_parameter is None:
        rls_oversample_parameter = intermediate_dict_bless.rls_oversample

    red_ratio = intermediate_dict_bless.lam / lam_new

    if red_ratio < 1.0:
        raise ValueError(f"red_ratio = {red_ratio} is less than 1.0")

    diag = K_diag_fn(n).to(X_data.device)

    # compute upper confidence bound on RLS of each sample,
    # overestimate (oversample) by a rls_oversample factor
    # to boost success probability at the expenses of a larger sample (dictionary)
    ucb = torch.minimum(
        rls_oversample_parameter * diag / (diag + lam_new), torch.tensor(1.0)
    )
    U = torch.rand(n, device=X_data.device) <= ucb
    u = U.sum()

    if u <= 0:
        raise ValueError(
            "No point selected during uniform sampling step, \
                try to increase rls_oversample_bless. "
            "Expected number of points: {:.3f}".format(n * ucb.mean())
        )

    X_U = X_data[U, :]

    rls_estimate = _estimate_rls_bless(
        intermediate_dict_bless, X_U, K_fn, K_diag_fn, lam_new
    )

    # same as before, oversample by a rls_oversample factor
    probs = torch.minimum(rls_oversample_parameter * rls_estimate, ucb[U])
    probs_reject = probs / ucb[U]

    if torch.any(probs < 0.0):
        raise ValueError(
            f"Some estimated probability is negative, this should never happen. "
            f"Min prob: {torch.min(probs)}"
        )

    deff_estimate = probs_reject.sum() / rls_oversample_parameter

    if rls_oversample_parameter * deff_estimate < 1.0:
        raise ValueError(
            "Estimated deff is smaller than 1, \
                you might want to reconsider your kernel. "
            "deff_estimate: {:.3f}".format(rls_oversample_parameter * deff_estimate)
        )

    selected = torch.rand(u, device=X_data.device) <= probs_reject
    s = selected.sum()

    if s.item() <= 0:
        raise ValueError(
            f"No point selected during RLS sampling step, \
                try to increase rls_oversample_bless. "
            f"Expected number of points (rls_oversample_bless*deff): \
                {probs_reject.sum().item():.3f}"
        )

    intermediate_dict_bless_new = CentersDictionary(
        idx=U.nonzero()[selected.nonzero()],
        X=X_U[selected, :],
        probs=probs[selected],
        lam=lam_new,
        rls_oversample=rls_oversample_parameter,
    )

    return intermediate_dict_bless_new


def _bless_size(
    X_data, K_fn, K_diag_fn, size_final, rls_oversample_param, nb_iter_bless=None
):
    n, _ = X_data.shape

    diag_norm = K_diag_fn(n).to(X_data.device)

    lam_init = diag_norm.sum() / (size_final - 1.0)
    lam_max = lam_init

    lam_final = 1
    lam_min = lam_final

    if nb_iter_bless is None:
        nb_iter_bless = torch.ceil(torch.log2(torch.tensor(lam_init))).int().item()

    lam_sequence = torch.exp(
        torch.linspace(
            torch.log(torch.tensor(lam_final)),
            torch.log(torch.tensor(lam_init)),
            nb_iter_bless,
        )
    ).tolist()

    ucb_init = rls_oversample_param * diag_norm / (diag_norm + lam_init)

    selected_init = torch.rand(n, device=X_data.device) <= ucb_init
    selected_init[0] = True  # force at least one sample to be selected
    ucb_init[0] = rls_oversample_param * 1.0

    intermediate_dict_bless = CentersDictionary(
        idx=selected_init.nonzero(),
        X=X_data[selected_init, :],
        probs=torch.ones(selected_init.sum(), device=X_data.device)
        * ucb_init[selected_init],
        lam=lam_init,
        rls_oversample=rls_oversample_param,
    )

    # discard lam_init from the list, we already used it to initialize
    lam_new = lam_sequence.pop()
    deff_hat_new = size_final - 1

    while len(lam_sequence) > 0:
        lam_old = lam_new
        deff_hat_old = deff_hat_new

        lam_new = lam_sequence.pop()
        intermediate_dict_bless = _reduce_lambda(
            X_data, K_fn, K_diag_fn, intermediate_dict_bless, lam_new
        )
        deff_hat_new = (
            len(intermediate_dict_bless.idx) / intermediate_dict_bless.rls_oversample
        )

        # check if the dictionary passed the lower threshold,
        # in which case set it to last valid lambda
        if deff_hat_old <= (size_final - 1) / 2 <= deff_hat_new:
            lam_max = lam_old

        if deff_hat_new >= 2 * (size_final + 1):
            lam_min = lam_new
            break

    return intermediate_dict_bless, lam_max, lam_min
