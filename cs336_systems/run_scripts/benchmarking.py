import torch
import logging
from dataclasses import asdict
from transformers import HfArgumentParser
from cs336_systems.benchmarking import BenchmarkingConfig
from cs336_basics.optimizer import AdamW
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import clip_gradient
from cs336_systems.benchmarking import get_random_benchmarking_data, forward_benchmarking, backward_benchmkarking


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()


parser = HfArgumentParser(BenchmarkingConfig)
benchmarking_config = parser.parse_args_into_dataclasses()[0]
logging.info(f"Benchmarking on device {benchmarking_config.device} with config:")
logging.info(f"{asdict(benchmarking_config)}")

model = BasicsTransformerLM(
    vocab_size=benchmarking_config.vocab_size,
    context_length=benchmarking_config.context_length,
    d_model=benchmarking_config.d_model,
    num_layers=benchmarking_config.num_layers,
    num_heads=benchmarking_config.num_heads,
    d_ff=benchmarking_config.d_ff,
    rope_theta=benchmarking_config.rope_theta,
).to(benchmarking_config.device)

# saves >25% running time for both forward and backward pass on cuda. breaks on mps.
model = torch.compile(model)

optimizer = AdamW(params=model.parameters())


text_input, text_output = get_random_benchmarking_data(benchmarking_config)

run_time = torch.zeros(3, benchmarking_config.benchmarking_iters)

for it in range(benchmarking_config.warmup_iters):
    logging.info(f"Warm-up iter #{it}")
    _, loss = forward_benchmarking(model, benchmarking_config, text_input, text_output)
    _ = backward_benchmkarking(optimizer, loss, benchmarking_config)
    clip_gradient(model.parameters(), 1)
    optimizer.step()

for it in range(benchmarking_config.benchmarking_iters):
    logging.info(f"Benchmarking iter #{it}")
    forward_runtime, loss = forward_benchmarking(model, benchmarking_config, text_input, text_output)
    backward_runtime = backward_benchmkarking(optimizer, loss, benchmarking_config)
    clip_gradient(model.parameters(), 1)
    optimizer.step()
    run_time[:, it] = torch.tensor([forward_runtime, backward_runtime, forward_runtime + backward_runtime])
run_time = run_time.mean(dim=-1, keepdim=False)
logging.info(
    f"Average time spent: {run_time[0].item():2f} sec on forward pass, and {run_time[1].item():2f} sec on backward pass."
)
