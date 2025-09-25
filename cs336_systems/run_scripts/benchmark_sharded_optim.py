import torch
import torch.multiprocessing as mp
import pandas as pd
import logging
from multiprocessing import Manager
from cs336_systems.sharded_optim import sharded_train, ShardedOptimizer
from cs336_basics.nn import TransformerLM
from cs336_systems.benchmarking import BenchmarkingConfig as LMConfig
from dataclasses import asdict
from transformers import HfArgumentParser

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()

if __name__ == "__main__":
    parser = HfArgumentParser(LMConfig)
    conf = parser.parse_args_into_dataclasses()[0]
    conf.device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Training on {conf.device} with configuration:")
    logging.info(f"{asdict(conf)}")

    dtype_full = torch.float32
    batch_sizes = (8, 16)
    data_x = torch.randint(0, conf.vocab_size, (conf.batch_size, conf.context_length), device="cpu")
    data_y = torch.randint(0, conf.vocab_size, (conf.batch_size, conf.context_length), device="cpu")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    world_size = 4
    warmup_iters = 2
    benchmark_iters = 4
    print(f"Testing the sharded Optimizer on {world_size} {conf.device}.")

    manager = Manager()
    result_queue = manager.Queue()
    mp.spawn(fn=sharded_train, 
             args=(world_size, backend, TransformerLM, ShardedOptimizer, data_x, data_y, 
                   warmup_iters, benchmark_iters, result_queue, conf, True),
             nprocs=world_size, join=True)
    run_time_dict = result_queue.get() 
    logging.info(run_time_dict)