import torch

from coupling_jacobian import svd


def metrics(Jac, Us=None, Ss=None, Vs=None, p=2, num_sing_vecs=(10, 30, 50),
            svd_method="torch", L=20, E=5, ITS=20, device="cpu", verbose=False):
    """
    Main method for computing coupling metrics between layer-wise Jacobians.

    Jac:            Jacobians across skip connection
    Us, Ss, Vs:     SVDs of Jac
    - if None, they are computed within this function
    p:              order of p-norm for coupling measurement
    num_sing_vecs:  number of top singular vectors to use in computing coupling 
    svd_method:     method for computing svd, see `jacobian.svd`
    - if using "random", `K, L, E, ITS` will be used
    """

    aln_ujv_all_k = {}
    aln_vju_all_k = {}

    if Us is None or Ss is None or Vs is None:
        Us, Ss, Vs = svd(Jac, K=max(num_sing_vecs), L=L, E=E, ITS=ITS, method=svd_method, verbose=verbose)

    S = torch.stack(Ss).cpu()
    U_all = [u.to(device) for u in Us]
    V_all = [v.to(device) for v in Vs]
    J = [j.to(device) for j in Jac]

    for K in num_sing_vecs:
        U, V = [u[:, :, :K] for u in U_all], [v[:, :K, :].permute(0,2,1) for v in V_all]

        ujv_mat_trace = torch.zeros((len(S), len(S), S.shape[1]))
        vju_mat_trace = torch.zeros((len(S), len(S), S.shape[1]))

        ujv_mat_norm = torch.zeros((len(S), len(S), S.shape[1]))
        vju_mat_norm = torch.zeros((len(S), len(S), S.shape[1]))

        for i in range(len(S)):
            for j in range(len(S)):
                uj, ji, vj = U[j], J[i], V[j]
                # ui, vi = U[i], V[i]
                if uj.shape[1] != ji.shape[2] or vj.shape[1] != ji.shape[1]:
                    print("wrong shape")
                    continue

                # S[i] 1D, S_i 2D
                S_i = torch.diag_embed(S[i][:,:K])

                ujv_mat_trace[i, j], ujv_mat_norm[i, j] = diag_sv_trace_similarity(ji, S_i, uj, vj, p=p)
                # vju_mat_trace[i, j], vju_mat_norm[i, j] = diag_sv_trace_similarity(ji, S_i, vj, uj, p=p)

        aln_ujv_all = {}
        aln_ujv_all['trace'] = ujv_mat_trace
        aln_ujv_all['norm'] = ujv_mat_norm

        # aln_vju_all = {}
        # aln_vju_all['trace'] = vju_mat_trace
        # aln_vju_all['norm'] = vju_mat_norm

        aln_ujv_all_k[K] = aln_ujv_all
        # aln_vju_all_k[K] = aln_vju_all

    return aln_ujv_all_k, None


def diag_sv_trace_similarity(J1, S1, U2, V2, p=2):  # swap U2 and V2 for the vju case
    """ 
    TODO: Main coupling metric
    """
    # (B, K, m) @ (B, m, n) @ (B, n, K) -> (B, K, K)
    M = torch.matmul(U2.transpose(1, 2), torch.matmul(J1, V2))

    # trace of each (B, K, K)
    tr = S1.diagonal(dim1=-2, dim2=-1).sum(-1)   # (B,)

    # p-norm of diag
    norm = torch.norm(S1.diagonal(dim1=-2, dim2=-1), p=p, dim=-1)  # (B,)

    # Frobenius norm of difference
    diff = torch.linalg.norm(torch.abs(M) - S1, dim=(-2, -1))  # (B,)

    return diff / tr, diff / norm


def diag_sv_similarity(U1, V1, U2, V2):  # swap U2 and V2 for the vju case
    M = U2.T @ U1 @ V1.T @ V2
    return torch.linalg.norm(torch.abs(M) - torch.eye(M.shape[0]))


def diag_sv_similarity_k(U1, S1, V1, U2, V2):  # swap U2 and V2 for the vju case
    M = U2.T @ U1 @ S1 @ V1.T @ V2
    tr = torch.trace(S1)
    norm = torch.linalg.norm(S1)
    diff = torch.linalg.norm(torch.abs(M) - S1)
    return diff / tr, diff / norm