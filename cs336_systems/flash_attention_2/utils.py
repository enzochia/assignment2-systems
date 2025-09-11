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
    IS_CAUSAL: tl.constexpr = False,
):
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
    # test it here, whether it's faster this way or directly load as transpose of K
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
    q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    for k_tile_idx in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # [K_TILE_SIZE, D]
        k_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
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
        o = tl.dot(p_tile.to(v_tile.dtype), v_tile, acc=o)
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


@triton.jit
def flash_bwd_kernel(
    dO_ptr, Q_ptr, K_ptr,
    V_ptr, O_ptr, L_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    D_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES: tl.constexpr, 
    N_KEYS: tl.constexpr,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr = False,
):
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )    
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    dtype_full = tl.float32
    eps=1e-8

    k_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    v_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
    dk_tile = tl.zeros([K_TILE_SIZE, D], dtype=dtype_full)
    dv_tile = tl.zeros([K_TILE_SIZE, D], dtype=dtype_full)
    for q_tile_idx in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero").to(dtype_full)
        # o_tile = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")
        L_tile = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        # dq_tile = tl.load(dQ_block_ptr, boundary_check=(0, 1), padding_option="zero")
        do_tile = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
        d_tile = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
        # [Q_TILE_SIZE, K_TILE_SIZE]
        score_tile = tl.dot(q_tile, tl.trans(k_tile)) * scale
        if IS_CAUSAL:
            q_seq_idx = tl.arange(0, Q_TILE_SIZE) + q_tile_idx * Q_TILE_SIZE
            k_seq_idx = tl.arange(0, K_TILE_SIZE) + key_tile_index * K_TILE_SIZE
            causal_mask = q_seq_idx[:, None] >= k_seq_idx[None, :]
            score_tile = tl.where(causal_mask, score_tile, float("-inf"))
        p_tile = tl.exp(score_tile - L_tile[:, None])
        # [K_TILE_SIZE, D]
        dv_tile += tl.dot(tl.trans(p_tile), do_tile)
        # [Q_TILE_SIZE, K_TILE_SIZE]
        dp_tile = tl.dot(do_tile, tl.trans(v_tile))
        ds_tile = p_tile * (dp_tile - d_tile[:, None]) * scale
        dk_tile += tl.dot(tl.trans(ds_tile), q_tile)

        #### calculation for atomic add: start ####
        # a tensor of memory pointers to the dQ tile to be added to
        idx_row = q_tile_idx * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        idx_col = tl.arange(0, D)
        dq_ptr_tensor = dQ_ptr + batch_index * stride_qb + \
                        idx_row[:, None] * stride_qq + idx_col[None, :] * stride_qd
        # [Q_TILE_SIZE, D]
        dq_tile_increment = tl.dot(ds_tile, k_tile)
        tl.atomic_add(dq_ptr_tensor, dq_tile_increment.to(dQ_ptr.type.element_ty))
        #### calculation for atomic add: end ####

        Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE_SIZE, 0))
        L_block_ptr = tl.advance(L_block_ptr, (Q_TILE_SIZE,))
        dO_block_ptr = tl.advance(dO_block_ptr, (Q_TILE_SIZE, 0))
        D_block_ptr = tl.advance(D_block_ptr, (Q_TILE_SIZE,))

    tl.store(dK_block_ptr, dk_tile.to(dK_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(dV_block_ptr, dv_tile.to(dV_ptr.type.element_ty), boundary_check=(0, 1))






