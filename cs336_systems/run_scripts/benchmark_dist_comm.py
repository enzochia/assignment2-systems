import torch
import torch.multiprocessing as mp
import pandas as pd
from cs336_systems.collective_operations import benchmark_all_reduce

if __name__ == "__main__":
    world_size_list = [2, 4]
    size_mb_list = [1, 10, 100, 1000]
    backend_dict = {"gloo": "CPU", "nccl": "GPU"}
    # backend_dict = {"nccl": "GPU"}
    warmup_iters, benchmark_iters = 5, 10
    results = []
    for backend in backend_dict:
        if torch.cuda.is_available() and backend == "nccl":
            print(f"{torch.cuda.device_count()} GPUs are found.")
            world_size_list = [x for x in world_size_list if x <= torch.cuda.device_count()]
        for size_mb in size_mb_list:
            for world_size in world_size_list:
                print(f"Benchmarking for all-reduce {size_mb}MB tensors on {world_size}{backend_dict[backend]}s.")
                ctx = mp.get_context("spawn")
                result_queue = ctx.Queue()
                
                mp.spawn(fn=benchmark_all_reduce, 
                         args=(world_size, backend, warmup_iters, 
                               benchmark_iters, size_mb, result_queue), 
                         nprocs=world_size, join=True)
                
                results.append(
                    {
                        "backend": backend_dict[backend],
                        "world_size": world_size,
                        "size_mb": size_mb,
                        "run_time": result_queue.get()
                    }
                )

    df = pd.DataFrame(results)
    print("\n--- Benchmark Results ---")
    print(df.to_string())
    # pd.to_markdown("all_reduce_run_time.md", index=False)



