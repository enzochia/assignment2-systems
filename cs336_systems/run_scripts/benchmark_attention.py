import torch
import logging
import timeit
import itertools
import pandas as pd
from dataclasses import asdict
from transformers import HfArgumentParser
from cs336_systems.benchmarking import BenchmarkingConfig

# from cs336_basics.optimizer import AdamW
# from cs336_basics.model import BasicsTransformerLM
# from cs336_basics.nn_utils import clip_gradient
from cs336_basics.nn import scaled_dot_product_attention
from cs336_systems.benchmarking import get_random_benchmarking_qkv


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()


parser = HfArgumentParser(BenchmarkingConfig)
benchmarking_config = parser.parse_args_into_dataclasses()[0]
logging.info(f"Benchmarking on device {benchmarking_config.device} with config:")
logging.info(f"{asdict(benchmarking_config)}")

batch_size = 8
d_model_dims = [16, 32, 64, 128]
seq_len_dims = [256, 1024, 4096, 8192, 16384]
benchmarking_config.batch_size = batch_size
results = []

# ~33% of improvements on both runtime and memory usage
scaled_dot_product_attention = torch.compile(scaled_dot_product_attention)

for seq_len, d_model in itertools.product(seq_len_dims, d_model_dims):
    config_str = f"d_model={d_model}, seq_len={seq_len}"
    logging.info(f"--- Benchmarking configuration: {config_str} ---")
    benchmarking_config.context_length = seq_len
    benchmarking_config.d_model = d_model
    q, k, v = get_random_benchmarking_qkv(benchmarking_config)
    resutls = []
    try:
        for _ in range(benchmarking_config.warmup_iters):
            o = scaled_dot_product_attention(q, k, v)
            loss = o.sum()
            loss.backward()

        q.grad, k.grad, v.grad = None, None, None

        start_time = timeit.default_timer()
        torch.cuda.synchronize()
        for _ in range(benchmarking_config.benchmarking_iters):
            o = scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        forward_time = (timeit.default_timer() - start_time) / benchmarking_config.benchmarking_iters

        peak_memory_bytes = torch.cuda.max_memory_allocated(benchmarking_config.device)
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)

        backward_time_list = []
        for _ in range(benchmarking_config.benchmarking_iters):
            o = scaled_dot_product_attention(q, k, v)
            loss = o.sum()
            start_time = timeit.default_timer()
            torch.cuda.synchronize()
            loss.backward()
            torch.cuda.synchronize()
            backward_time_list.append(timeit.default_timer() - start_time)
        backward_time = sum(backward_time_list) / benchmarking_config.benchmarking_iters
    except torch.cuda.OutOfMemoryError:
        logging.error(f"OOM at config: {config_str}")
        torch.cuda.empty_cache()
        forward_time = "OOM"
        backward_time = "OOM"
        peak_memory_mb = "OOM"

    results.append(
        {
            "seq_len": seq_len,
            "d_model": d_model,
            "forward_time": forward_time,
            "backward_time": backward_time,
            "peak_memory": peak_memory_mb,
        }
    )
torch.cuda.memory._record_memory_history(enabled=None)

df = pd.DataFrame(results)
print("\n--- Benchmark Results ---")
print(df.to_string())
