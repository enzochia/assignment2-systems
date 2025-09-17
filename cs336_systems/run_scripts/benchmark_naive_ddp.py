import torch
import torch.multiprocessing as mp
import pandas as pd
import logging
from multiprocessing import Manager
from cs336_systems.collective_operations import ddp_train, single_process_train
from cs336_basics.nn import TransformerLM
from cs336_basics.config import Config as LMConfig
from dataclasses import asdict
from transformers import HfArgumentParser
from tests.common import ToyModelWithTiedWeights

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
    backend = "gloo"
    world_size = 4
    warmup_iters = 2
    benchmark_iters = 4
    print(f"A naive DDP test on {world_size} {conf.device}.")

    manager = Manager()
    result_queue = manager.Queue()
    mp.spawn(fn=ddp_train, 
             args=(world_size, backend, TransformerLM, data_x, data_y, warmup_iters, 
                   benchmark_iters, result_queue, conf, True),
             nprocs=world_size, join=True)
    ddp_state_dict = result_queue.get()
    run_time_dict = result_queue.get() 
    logging.info(run_time_dict)

    single_process_state_dict = single_process_train(TransformerLM, data_x, data_y, 
                                                     warmup_iters + benchmark_iters, 
                                                     world_size, conf)

    for key in single_process_state_dict.keys():
        assert key in ddp_state_dict
        assert torch.allclose(single_process_state_dict[key].to(conf.device), 
                              ddp_state_dict[key].to(conf.device),
                              atol=1e-5)
    print(f"PASSED.")

