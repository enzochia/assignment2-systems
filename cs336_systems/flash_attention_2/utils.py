import torch
import triton
import triton.language as tl


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
            m = torch.maximum(m_prev, torch.max(s_i_j.to(dtype_full), dim=-1)[0])
            stabilization_term = torch.exp(m_prev.detach() - m.detach())
            # [..., q_tile_size, k_tile_size]
            p_tilda_i_j = torch.exp(s_i_j - m.detach()[..., None])
            # [..., q_tile_size]
            l = stabilization_term * l + torch.sum(p_tilda_i_j, dim=-1)
            # [..., q_tile_size, d_model]
            o = stabilization_term[..., None] * o + torch.matmul(p_tilda_i_j.to(v_block.dtype), v_block)
    # [..., q_tile_size]
    l += eps
    o = (1 / l)[..., None] * o
    # [..., q_tile_size]
    logsumexp = m + torch.log(l)    
    o_ptr[..., q_block_ptr:(q_block_ptr + q_tile_size), :] = o.to(o_ptr.device, dtype=q_ptr.dtype)
    L_ptr[..., q_block_ptr:(q_block_ptr + q_tile_size)] = logsumexp.to(L_ptr.device)



@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    tl.device_print("##########################################################################")
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )


    dtype_full = tl.float32
    eps=1e-8

    m = tl.full([Q_TILE_SIZE], float("-inf"), dtype=dtype_full)
    l = tl.zeros([Q_TILE_SIZE], dtype=dtype_full)
    o = tl.zeros([Q_TILE_SIZE, D], dtype=dtype_full)
    # [Q_TILE_SIZE, D]
    q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero").to(dtype_full)

    for k_tile_idx in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # [K_TILE_SIZE, D]
        k_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero").to(dtype_full)
        v_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        # [Q_TILE_SIZE, K_TILE_SIZE]
        score_tile = tl.dot(q_tile, tl.trans(k_tile)) * scale
        if IS_CAUSAL:
            q_seq_idx = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE
            k_seq_idx = tl.arange(0, K_TILE_SIZE) + k_tile_idx * K_TILE_SIZE
            causal_mask = q_seq_idx[:, None] >= k_seq_idx[None, :]
            score_tile = tl.where(causal_mask, score_tile, float("-inf"))

        m_prev = m
        # [Q_TILE_SIZE]
        m = tl.maximum(m_prev, tl.max(score_tile, axis=-1))
        stabilization_term = tl.exp(m_prev - m)
        # [Q_TILE_SIZE, K_TILE_SIZE]
        p_tile = tl.exp(score_tile - m[:, None])
        # [Q_TILE_SIZE]
        l = stabilization_term * l + tl.sum(p_tile, axis=-1)
        # [Q_TILE_SIZE, D]
        o = stabilization_term[:, None] * o 
        o = tl.dot(p_tile.to(V_block_ptr.type.element_ty), v_tile, acc=o)

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
    # [Q_TILE_SIZE]
    l += eps
    # [Q_TILE_SIZE, D]
    o = (1 / l)[:, None] * o
    # [Q_TILE_SIZE]
    logsumexp = m + tl.log(l)
    tl.store(O_block_ptr, o.to(O_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, logsumexp, boundary_check=(0,))








        