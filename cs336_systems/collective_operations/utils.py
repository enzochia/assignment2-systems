import timeit
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tests.common import _setup_process_group, _cleanup_process_group
from cs336_basics.optim import AdamW


def benchmark_all_reduce(rank, world_size, backend,
                         warmup_iters, benchmark_iters, 
                         size_mb, result_queue):
    device = _setup_process_group(rank, world_size, backend)

    try:
        dtype_full = torch.float32
        num_elements = size_mb * 1024 * 1024 // 4
        device = "cpu" if backend == "gloo" else device
        
        for _ in range(warmup_iters):
            data = torch.randn(num_elements, device=device, dtype=dtype_full)
            dist.all_reduce(data, async_op=False)

        if backend == "nccl":
            torch.cuda.synchronize()
        run_time_list = []
        for _ in range(benchmark_iters):
            data = torch.randn(num_elements, device=device, dtype=dtype_full)
            start_time = timeit.default_timer()
            dist.all_reduce(data, async_op=False)
            if backend == "nccl":
                torch.cuda.synchronize()
            run_time_list.append(timeit.default_timer() - start_time)

        run_time = sum(run_time_list) / len(run_time_list)
        gathered_run_time_list = [None] * world_size
        dist.all_gather_object(gathered_run_time_list, run_time)

        if rank == 0:
            result_queue.put(sum(gathered_run_time_list) / len(gathered_run_time_list))
    finally:
        _cleanup_process_group()


def ddp_train(rank, world_size, backend, ModelClass, data_x, data_y, num_steps, result_queue):
    device = _setup_process_group(rank, world_size, backend)
    device = "cpu" if backend == "gloo" else device
    try:
        torch.manual_seed(rank)
        toy_model = ModelClass().to(device)
        for param in toy_model.parameters():
            dist.broadcast(param.data, src=0)
        
        batch_size = data_x.shape[0]
        local_batch_size = batch_size // world_size
        data_x = data_x[(local_batch_size * rank):(local_batch_size * (rank + 1)), ...].to(device)
        prob_label = data_y[(local_batch_size * rank):(local_batch_size * (rank + 1)), ...].to(device)

        optimizer = AdamW(toy_model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()

        toy_model.train()
        for _ in range(num_steps):
            optimizer.zero_grad()
            logits = toy_model(data_x)
            loss = loss_fn(logits, prob_label)
            loss.backward()
            for param in toy_model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=False)
                # actually do not need this when async_op=False
                # torch.cuda.synchronize()
                param.grad /= world_size
            optimizer.step()
        
        if rank == 0:
            state_dict = {k: v.detach().cpu() for k, v in toy_model.state_dict().items()}
            result_queue.put(state_dict)
    finally:
        _cleanup_process_group()

def single_process_train(ModelClass, data_x, data_y, num_steps, world_size, device):
        torch.manual_seed(0)
        toy_model = ModelClass().to(device)
        data_x = data_x.to(device)
        data_y = data_y.to(device)

        optimizer = AdamW(toy_model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()

        toy_model.train()
        for _ in range(num_steps):
            optimizer.zero_grad()
            logits = toy_model(data_x)
            loss = loss_fn(logits, data_y)
            loss.backward()
            optimizer.step()
        state_dict = {k: v.detach() for k, v in toy_model.state_dict().items()}
        return state_dict