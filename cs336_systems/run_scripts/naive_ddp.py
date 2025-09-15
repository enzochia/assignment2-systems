import torch
import torch.multiprocessing as mp
import pandas as pd
from multiprocessing import Manager
from cs336_systems.collective_operations import ddp_train, single_process_train
from tests.common import ToyModelWithTiedWeights

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_full = torch.float32
    batch_sizes = (8, 16)
    data_x = torch.randn(*batch_sizes, 10, device="cpu", dtype=dtype_full)
    data_y = torch.randn(*batch_sizes, 5, device="cpu", dtype=dtype_full)
    backend = "nccl"
    world_size = 4
    num_steps = 10
    print(f"A naive DDP test on {world_size} {device}.")

    manager = Manager()
    result_queue = manager.Queue()
    mp.spawn(fn=ddp_train, 
             args=(world_size, backend, ToyModelWithTiedWeights, data_x, data_y, num_steps, result_queue),
             nprocs=world_size, join=True)
    ddp_state_dict = result_queue.get()

    single_process_state_dict = single_process_train(ToyModelWithTiedWeights, data_x, data_y, 
                                                     num_steps, world_size, device)

    for key in single_process_state_dict.keys():
        assert key in ddp_state_dict
        assert torch.allclose(single_process_state_dict[key].to(device), ddp_state_dict[key].to(device))
    print(f"PASSED.")

