import torch
import triton
import triton.language as tl
from typing import Tuple, Any
from .utils import flash_fwd_kernel


class FlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                Q: torch.Tensor,
                K: torch.Tensor,
                V: torch.Tensor,
                is_causal: bool = False) -> Tuple[Any]:
        dtype_full = torch.float32
        q_tile_size = 32
        k_tile_size = 32

        Q = Q.contiguous().view(-1, *Q.shape[-2:])
        K = K.contiguous().view(-1, *K.shape[-2:])
        V = V.contiguous().view(-1, *V.shape[-2:])
        # dtype_full? 
        # O = torch.empty_like(Q)
        O = torch.zeros(Q.shape, device=Q.device, dtype=dtype_full)
        L = torch.zeros(Q.shape[:-1], device=Q.device, dtype=dtype_full)

        flash_fwd_kernel[((Q.shape[-2] +  q_tile_size - 1) // q_tile_size, Q.shape[0])](
            Q_ptr=Q, K_ptr=K, V_ptr=V,
            O_ptr=O, L_ptr=L,
            stride_qb=Q.stride(0), stride_qq=Q.stride(1), stride_qd=Q.stride(2),
            stride_kb=K.stride(0), stride_kk=K.stride(1), stride_kd=K.stride(2),
            stride_vb=V.stride(0), stride_vk=V.stride(1), stride_vd=V.stride(2),
            stride_ob=O.stride(0), stride_oq=O.stride(1), stride_od=O.stride(2),
            stride_lb=L.stride(0), stride_lq=L.stride(1),
            N_QUERIES=Q.shape[-2], N_KEYS=K.shape[-2],
            scale= 1 / (Q.shape[-1] ** 0.5),
            D=Q.shape[-1],
            Q_TILE_SIZE=q_tile_size,
            K_TILE_SIZE=k_tile_size,
            IS_CAUSAL=is_causal,
        )
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        # reshape O
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



        




