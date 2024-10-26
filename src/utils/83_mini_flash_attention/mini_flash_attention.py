import math

import torch

import mini_flash_attn as cuda_flash_attn


def flash_attn_fwd(Q: torch.Tensor,
                   K: torch.Tensor,
                   V: torch.Tensor,
                   Bc: int,
                   Br: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    """
    assert Q.shape == K.shape == V.shape
    assert Q.device == K.device == V.device
    n: int = Q.size(0)
    d: int = Q.size(1)
    dtype = V.dtype
    device = V.device

    assert n % Bc == 0 and n % Br == 0
    Tc: int = n // Bc
    Tr: int = n // Br

    softmax_scale: float = 1.0 / math.sqrt(d)

    O: torch.Tensor = torch.empty((n, d,), dtype=dtype, device=device)  # (n, d,)
    L: torch.Tensor = torch.empty((n,), dtype=dtype, device=device)  # (n,)

    for i in range(Tr):
        Qi: torch.Tensor = Q[i * Br:(i + 1) * Br] * softmax_scale  # (Br, d,)
        Oi: torch.Tensor = torch.zeros((Br, d,), dtype=dtype, device=device)  # (Br, d,)

        m: torch.Tensor = torch.full((Br,), -torch.inf, dtype=dtype, device=device)  # (Br,)
        l: torch.Tensor = torch.zeros((Br,), dtype=dtype, device=device)  # (Br,)

        for j in range(Tc):
            Kj: torch.Tensor = K[j * Bc:(j + 1) * Bc]  # (Bc, d,)
            Vj: torch.Tensor = V[j * Bc:(j + 1) * Bc]  # (Bc, d,)
            S: torch.Tensor = Qi @ Kj.transpose(-2, -1)  # (Br, Bc,)

            m_old: torch.Tensor = m  # (Br,)
            m = torch.maximum(torch.max(S, dim=-1)[0], m_old)  # (Br,)

            P: torch.Tensor = torch.exp(S - m.unsqueeze(-1))  # (Br, Bc,)

            l = torch.exp(m_old - m) * l + torch.sum(P, dim=-1)  # (Br,)
            Oi = torch.diag(torch.exp(m_old - m)) @ Oi + P @ Vj  # (Br, Bc,)

        O[i * Br:(i + 1) * Br] = torch.inverse(torch.diag(l)) @ Oi  # (Br, Bc,)
        L[i * Br:(i + 1) * Br] = m + torch.log(l)  # (Br,)

    return O, L


def flash_attn_bwd(Q: torch.Tensor,
                   K: torch.Tensor,
                   V: torch.Tensor,
                   O: torch.Tensor,
                   dO: torch.Tensor,
                   L: torch.Tensor,
                   Bc: int,
                   Br: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    P = softmax(S)
    O = P @ V

    D = dO * O
    dS = P * (dP - D)
    """
    assert Q.shape == K.shape == V.shape == O.shape == dO.shape and Q.shape[:-1] == L.shape
    assert Q.device == K.device == V.device == O.device == dO.device == L.device
    n: int = Q.size(0)
    d: int = Q.size(1)
    dtype = torch.float32
    device = V.device

    assert n % Br == 0 and n % Bc == 0
    Tc: int = n // Bc
    Tr: int = n // Br

    softmax_scale: float = 1.0 / math.sqrt(d)

    D: torch.Tensor = torch.sum(dO * O, dim=-1)  # (n,)

    dQ: torch.Tensor = torch.zeros_like(Q, dtype=dtype, device=device)  # (n, d,)
    dK: torch.Tensor = torch.zeros_like(K, dtype=dtype, device=device)  # (n, d,)
    dV: torch.Tensor = torch.zeros_like(V, dtype=dtype, device=device)  # (n, d,)

    for j in range(Tc):
        Kj: torch.Tensor = K[j * Bc:(j + 1) * Bc, :]  # (Bc, d,)
        Vj: torch.Tensor = V[j * Bc:(j + 1) * Bc, :]  # (Bc, d,)
        dKj: torch.Tensor = torch.zeros((Bc, d,), dtype=dtype, device=device)  # (Bc, d,)
        dVj: torch.Tensor = torch.zeros((Bc, d,), dtype=dtype, device=device)  # (Bc, d,)

        for i in range(Tr):
            Qi: torch.Tensor = Q[i * Br:(i + 1) * Br, :] * softmax_scale  # (Br, d,)
            dOi: torch.Tensor = dO[i * Br:(i + 1) * Br, :]  # (Br, d,)
            Li: torch.Tensor = L[i * Br:(i + 1) * Br]  # (Br,)
            Di: torch.Tensor = D[i * Br:(i + 1) * Br]  # (Br,)

            S: torch.Tensor = Qi @ Kj.transpose(-2, -1) # (Br, Bc,)
            P: torch.Tensor = torch.exp(S - Li.unsqueeze(-1))  # (Br, Bc,)

            dVj = dVj + P.transpose(-2, -1) @ dOi  # (Bc, d,)
            dP: torch.Tensor = dOi @ Vj.transpose(-2, -1)  # (Br, Bc,)
            dS: torch.Tensor = P * (dP - Di.unsqueeze(-1))  # (Br, Bc,)

            dQi: torch.Tensor = dQ[i * Br:(i + 1) * Br, :]  # (Br, d,)
            dQi = dQi + dS @ Kj  # (Br, d,)
            dQ[i * Br:(i + 1) * Br, :] = dQi  # (Br, d,)

            dKj = dKj + dS.transpose(-2, -1) @ Qi  # (Bc, d,)

        dK[j * Bc:(j + 1) * Bc, :] = dKj
        dV[j * Bc:(j + 1) * Bc, :] = dVj

    return dQ, dK, dV


def main() -> None:
    dtype = torch.float32
    device = torch.device('cuda')

    batch_size: int = 1
    num_heads: int = 1
    seqlen: int = 256
    embed_dim: int = 32
    assert batch_size == 1 and num_heads == 1

    Br: int = 16
    Bc: int = 16

    QQ: torch.Tensor = torch.rand(batch_size, num_heads, seqlen, embed_dim).to(dtype=dtype, device=device)
    KK: torch.Tensor = torch.rand(batch_size, num_heads, seqlen, embed_dim).to(dtype=dtype, device=device)
    VV: torch.Tensor = torch.rand(batch_size, num_heads, seqlen, embed_dim).to(dtype=dtype, device=device)

    """
    Ground truth.
    """
    Q1 = QQ.clone()
    K1 = KK.clone()
    V1 = VV.clone()
    Q1.requires_grad_(True)
    K1.requires_grad_(True)
    V1.requires_grad_(True)

    softmax_scale: float = 1.0 / math.sqrt(embed_dim)
    gt = torch.softmax(Q1 @ K1.transpose(-2, -1) * softmax_scale, dim=-1) @ V1
    loss1 = torch.mean(gt)
    loss1.backward()

    """
    Py flash attention. 
    """
    Q2 = QQ.clone()
    K2 = KK.clone()
    V2 = VV.clone()
    Q2.requires_grad_(True)
    K2.requires_grad_(True)
    V2.requires_grad_(True)

    O, L = flash_attn_fwd(Q2.squeeze(), K2.squeeze(), V2.squeeze(), Bc, Br)
    O.retain_grad()
    loss2 = torch.mean(O)
    loss2.backward()
    dO = O.grad
    dQ, dK, dV = flash_attn_bwd(Q2.squeeze(), K2.squeeze(), V2.squeeze(), O, dO, L, Bc, Br)

    """
    CUDA flash attention.
    """
    Q3 = QQ.clone()
    K3 = KK.clone()
    V3 = VV.clone()
    Q3.requires_grad_(True)
    K3.requires_grad_(True)
    V3.requires_grad_(True)

    cuda_flash_attn_out = cuda_flash_attn.fwd(Q3, K3, V3)

    """
    Evaluation.
    """
    rtol: float = 1e-04
    atol: float = 1e-04
    print("cu_flash_attn  O == gt ?", torch.allclose(cuda_flash_attn_out, gt, rtol=rtol, atol=atol))
    print()
    print("py_flash_attn  O == gt ?", torch.allclose(O, gt, rtol=rtol, atol=atol))
    print("py_flash_attn dQ == gt ?", torch.allclose(dQ, Q1.grad, rtol=rtol, atol=atol))
    print("py_flash_attn dK == gt ?", torch.allclose(dK, K1.grad, rtol=rtol, atol=atol))
    print("py_flash_attn dV == gt ?", torch.allclose(dV, V1.grad, rtol=rtol, atol=atol))


if __name__ == '__main__':
    main()
