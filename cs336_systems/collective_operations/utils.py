import timeit
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank)

def benchmark_all_reduce(rank, world_size, backend,
                         warmup_iters, benchmark_iters, 
                         size_mb, result_queue):
    setup(rank, world_size, backend)
    dtype_full = torch.float32
    num_elements = size_mb * 1024 * 1024 // 4
    device = None
    if backend == "nccl":
        device = torch.device("cuda")
    elif backend == "gloo":
        device = torch.device("cpu")
    else:
        raise ValueError("Only nccl and gloo are supported.")
    
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



