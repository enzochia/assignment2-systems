import timeit
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from dataclasses import asdict
from tests.common import _setup_process_group, _cleanup_process_group


def sharded_train(rank, world_size, backend, ModelClass, 
                  OptimizerClass, data_x, data_y, warmup_iters, 
                  benchmark_iters, result_queue, conf,
                  do_flattened_comm=True):
    device = _setup_process_group(rank, world_size, backend)
    device = "cpu" if backend == "gloo" else device
    try:
        torch.manual_seed(rank)
        toy_model = ModelClass(
            vocab_size=conf.vocab_size,
            context_length=conf.context_length,
            d_model=conf.d_model,
            num_layers=conf.num_layers,
            num_heads=conf.num_heads,
            d_ff=conf.d_ff,
            rope_theta=conf.rope_theta,
            device=conf.device
        ).to(conf.device)
        for param in toy_model.parameters():
            dist.broadcast(param.data, src=0)
        
        batch_size = data_x.shape[0]
        local_batch_size = batch_size // world_size
        data_x = data_x[(local_batch_size * rank):(local_batch_size * (rank + 1)), ...].to(device)
        prob_label = data_y[(local_batch_size * rank):(local_batch_size * (rank + 1)), ...].to(device)

        if conf.benchmark_memory:
            torch.cuda.memory._record_memory_history(max_entries=1000000)
            torch.cuda.memory._dump_snapshot(conf.benchmark_memory_path)

        optimizer = OptimizerClass(toy_model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()

        run_time_list = []
        toy_model.train()
        for iter in range(warmup_iters + benchmark_iters):
            full_step_start_time = timeit.default_timer()
            optimizer.zero_grad()
            logits = toy_model(data_x)
            loss = loss_fn(logits.view(-1, conf.vocab_size), prob_label.view(-1))

            if conf.benchmark_memory:
                torch.cuda.memory._dump_snapshot(conf.benchmark_memory_path)

            loss.backward()

            if conf.benchmark_memory:
                torch.cuda.memory._dump_snapshot(conf.benchmark_memory_path)
            # enable this line when benchmarking on CUDA
            torch.cuda.synchronize()
            optim_step_start_time = timeit.default_timer()
            optimizer.step()
            # # enable this line when benchmarking on CUDA
            torch.cuda.synchronize()
            optim_step_time = timeit.default_timer() - optim_step_start_time
            if conf.benchmark_memory:
                torch.cuda.memory._dump_snapshot(conf.benchmark_memory_path)
            full_step_time = timeit.default_timer() - full_step_start_time
            if iter >= warmup_iters:
                run_time_list.append({"full_step_time": full_step_time,
                                      "optim_step_time": optim_step_time})
        run_time_dict = {"full_step_time": sum(x["full_step_time"] for x in run_time_list) / len(run_time_list),
                         "optim_step_time": sum(x["optim_step_time"] for x in run_time_list) / len(run_time_list)}
        
        if rank == 0:
            result_queue.put(run_time_dict)
    finally:
        _cleanup_process_group()

