import torch
import timeit
from cs336_systems.benchmarking import BenchmarkingConfig

# from cs336_basics.optimizer import AdamW
# from cs336_basics.model import BasicsTransformerLM
# from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optim import AdamW
from cs336_basics.nn import TransformerLM, cross_entropy


def get_random_benchmarking_data(benchmarking_config: BenchmarkingConfig) -> tuple[torch.Tensor]:
    text_rand = torch.randint(
        0, benchmarking_config.vocab_size, (benchmarking_config.batch_size, benchmarking_config.context_length + 1)
    )
    text_input = text_rand[:, :-1].to(benchmarking_config.device)
    text_output = text_rand[:, 1:].to(benchmarking_config.device)
    return text_input, text_output


def forward_benchmarking(
    # model: BasicsTransformerLM,
    model: TransformerLM,
    benchmarking_config: BenchmarkingConfig,
    text_input: torch.Tensor,
    text_output: torch.Tensor
) -> torch.Tensor:
    start_time = timeit.default_timer()
    benchmarking_config.sync_func()
    if benchmarking_config.benchmark_memory:
        torch.cuda.memory._dump_snapshot(benchmarking_config.benchmark_memory_path)
    logits = model(text_input)
    if benchmarking_config.benchmark_memory:
        torch.cuda.memory._dump_snapshot(benchmarking_config.benchmark_memory_path)
    loss = cross_entropy(logits, text_output)
    benchmarking_config.sync_func()
    run_time = timeit.default_timer() - start_time
    return run_time, loss


def backward_benchmkarking(
    optimizer: AdamW, 
    loss: torch.Tensor, 
    benchmarking_config: BenchmarkingConfig
) -> None:
    start_time = timeit.default_timer()
    benchmarking_config.sync_func()
    if benchmarking_config.benchmark_memory:
        torch.cuda.memory._dump_snapshot(benchmarking_config.benchmark_memory_path)
    optimizer.zero_grad()
    loss.backward()
    if benchmarking_config.benchmark_memory:
        torch.cuda.memory._dump_snapshot(benchmarking_config.benchmark_memory_path)
    benchmarking_config.sync_func()
    run_time = timeit.default_timer() - start_time
    return run_time
