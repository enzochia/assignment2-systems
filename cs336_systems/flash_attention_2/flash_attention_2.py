import torch
import triton
import triton.language as tl
from typing import Tuple, Any
from .utils import flash_fwd_kernel, flash_bwd_kernel


class FlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                Q: torch.Tensor,
                K: torch.Tensor,
                V: torch.Tensor,
                is_causal: bool = False) -> Tuple[Any]:
        dtype_full = torch.float32
        q_tile_size, k_tile_size = 32, 32

        batch_sizes = Q.shape[:-2]
        Q = Q.view(-1, *Q.shape[-2:])
        K = K.view(-1, *K.shape[-2:])
        V = V.view(-1, *V.shape[-2:])
        # if something goes wrong with precision, it could be here
        O = torch.zeros_like(Q)
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
        ctx.tile_sizes = (32, 32)
        O = O.view(*batch_sizes, *O.shape[-2:])
        return O

    @staticmethod
    def backward(ctx, dO: torch.Tensor) -> None:
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        q_tile_size, k_tile_size = ctx.tile_sizes
        d_model = Q.shape[-1]

        batch_sizes = dO.shape[:-2]
        dO = dO.view(-1, *Q.shape[-2:])
        # [..., context_len_q]
        D = (O * dO).sum(dim=-1)

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        flash_bwd_kernel[((K.shape[-2] +  k_tile_size - 1) // k_tile_size, Q.shape[0])](
            dO_ptr=dO, Q_ptr=Q, K_ptr=K,
            V_ptr=V, O_ptr=O, L_ptr=L,
            dQ_ptr=dQ, dK_ptr=dK, dV_ptr=dV,
            D_ptr=D,
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

        dQ = dQ.view(*batch_sizes, *dO.shape[-2:])
        dK = dK.view(*batch_sizes, *dO.shape[-2:])
        dV = dV.view(*batch_sizes, *dO.shape[-2:])

        return dQ, dK, dV, None