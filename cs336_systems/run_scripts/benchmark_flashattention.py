import torch
import logging
import timeit
import itertools
import triton
import pandas as pd
from cs336_systems.flash_attention_2 import FlashAttention2_PyTorch, FlashAttention2

def test_timing_flash_forward_backward(seq_len, num_head, d_head):
    q, k, v = torch.randn(
        3, num_head, seq_len, d_head, device='cuda', dtype=torch.bfloat16, requires_grad=True
    )

    flash = torch.compile(FlashAttention2.apply)

    def flash_forward_backward():
        o = flash(q, k, v, True)
        loss = o.sum()
        loss.backward()
    results = triton.testing.do_bench(flash_forward_backward, rep=10000, warmup=1000)
    print(results)


seq_len_list = [2 ** x for x in range(10, 11)]
num_head_list = [2 ** x for x in range(4, 5)]
d_head_list = [2 ** x for x in range(8, 9)]

for seq_len, num_head, d_head in itertools.product(seq_len_list, num_head_list, d_head_list):
    test_timing_flash_forward_backward(seq_len, num_head, d_head)