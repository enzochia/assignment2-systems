import torch
from typing import Tuple, Any


def _flash_attention_2_forward(
    q_ptr: torch.Tensor, k_ptr: torch.Tensor, v_ptr: torch.Tensor,
    o_ptr: torch.Tensor, L_ptr: torch.Tensor,
    q_tile_size: int, k_tile_size: int,
    program_id: int,
    is_causal: bool,
    device: torch.device = torch.device("cuda"),
    dtype_full: torch.dtype = torch.float32,
    eps: float = 1e-8
) -> None:
    d_q = q_ptr.shape[-2]
    d_k = k_ptr.shape[-2]
    d_model = q_ptr.shape[-1]
    d_batch = q_ptr.shape[:-2]
    q_block_ptr = q_tile_size * program_id
    q_block = q_ptr[..., q_block_ptr:(q_block_ptr + q_tile_size), :]
    m = torch.full([*d_batch, q_tile_size], -torch.inf, dtype=dtype_full, device=device)
    l = L_ptr[..., q_block_ptr:(q_block_ptr + q_tile_size)]
    o = torch.zeros([*d_batch, q_tile_size, d_model], dtype=dtype_full, device=device)
    for k_block_idx in range((d_k // k_tile_size) + 1):
        k_block_ptr = k_tile_size * k_block_idx
        if k_block_ptr < d_k:
            k_tile_size=min(k_tile_size, d_k - k_block_ptr)
            k_block = k_ptr[..., k_block_ptr:(k_block_ptr + k_tile_size), :]
            v_block = v_ptr[..., k_block_ptr:(k_block_ptr + k_tile_size), :]
            # [..., q_tile_size, k_tile_size]
            s_i_j = torch.matmul(q_block, k_block.transpose(-2, -1)) / (d_model ** 0.5)
            if is_causal:
                causal_mask = torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool).tril(diagonal=0)
                # don't do inplace masked_fill_
                s_i_j = s_i_j.masked_fill(causal_mask.logical_not(), float("-inf"))
            m_prev = m
            # [..., q_tile_size]
            m = torch.maximum(m_prev, torch.max(s_i_j, dim=-1)[0])
            stabilization_term = torch.exp(m_prev.detach() - m.detach())
            # [..., q_tile_size, k_tile_size]
            p_tilda_i_j = torch.exp(s_i_j - m.detach()[..., None])
            # [..., q_tile_size]
            l = stabilization_term * l + torch.sum(p_tilda_i_j, dim=-1)
            # [..., q_tile_size, d_model]
            o = stabilization_term[..., None] * o + torch.matmul(p_tilda_i_j, v_block)
    # [..., q_tile_size, d_model]
    l += eps
    o = (1 / l)[..., None] * o
    # [..., q_tile_size]
    logsumexp = m + torch.log(l)    
    o_ptr[..., q_block_ptr:(q_block_ptr + q_tile_size), :] = o.to(o_ptr.device, dtype=q_ptr.dtype)
    L_ptr[..., q_block_ptr:(q_block_ptr + q_tile_size)] = logsumexp.to(L_ptr.device)


class FlashAttention2_PyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                Q: torch.Tensor,
                K: torch.Tensor,
                V: torch.Tensor,
                is_causal: bool = False) -> Tuple[Any]:
        dtype_full = torch.float32
        q_context_len = Q.shape[-2]
        q_tile_size = 32
        k_tile_size = 16
        O = torch.empty_like(Q)
        L = torch.zeros(Q.shape[:-1], device=Q.device, dtype=dtype_full)
        for program_id in range((q_context_len // q_tile_size) + 1):
            if program_id * q_tile_size < q_context_len:
                _flash_attention_2_forward(q_ptr=Q, k_ptr=K, v_ptr=V, o_ptr=O, L_ptr=L, 
                                            q_tile_size=min(q_tile_size, q_context_len - program_id * q_tile_size), 
                                            k_tile_size=k_tile_size, program_id=program_id, 
                                            is_causal=is_causal, device=Q.device, dtype_full=dtype_full)
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO: torch.Tensor) -> None:
        Q, K, V, O, L = ctx.saved_tensors
        d_model = Q.shape[-1]

        # [..., context_len_q]
        D = (O * dO).sum(dim=-1)
        # [..., context_len_q, context_len_k]
        S = torch.matmul(Q, K.transpose(-2, -1)) / (d_model ** 0.5)
        if ctx.is_causal:
            causal_mask = torch.ones(Q.shape[-2], K.shape[-2], device=Q.device, dtype=torch.bool).tril(diagonal=0)
            # don't do inplace masked_fill_
            S = S.masked_fill(causal_mask.logical_not(), float("-inf"))
        P = torch.exp(S - L[..., None])
        # [..., context_len_k, d_model]
        dV = torch.matmul(P.transpose(-2, -1), dO)
        # [..., context_len_q, context_len_k]
        dP = torch.matmul(dO, V.transpose(-2, -1))
        dS = P * (dP - D[..., None])
        # [..., context_len_q, d_model]
        dQ = torch.matmul(dS, K) / (d_model ** 0.5)
        # [..., context_len_k, d_model]
        dK = torch.matmul(dS.transpose(-2, -1), Q) / (d_model ** 0.5)
        return dQ, dK, dV, None



        




