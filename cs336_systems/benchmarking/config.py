import torch
from dataclasses import dataclass, field
from collections.abc import Callable


@dataclass
class BenchmarkingConfig:
    device: torch.device | None = field(
        default=torch.device("cuda")
        if torch.cuda.is_available()
        else (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    )

    # Model parameters
    d_model: int | None = field(default=768)
    d_ff: int | None = field(default=3072)
    num_layers: int | None = field(default=12)
    num_heads: int | None = field(default=12)
    rope_theta: int | None = field(default=10000)

    # Training parameters
    batch_size: int | None = field(default=64)
    context_length: int | None = field(default=1024)
    vocab_size: int | None = field(default=50257)
    use_mixed_precision: bool | None = field(default=False)
    precision: str | None = field(default="bf16")
    train_context_dtype: torch.dtype | None = field(default=torch.bfloat16)

    # Logging parameters
    wandb_logging: bool | None = field(default=False)
    wandb_project: str | None = field(default=None)
    wandb_run_name: str | None = field(default=None)

    # Benchmarking parameters
    warmup_iters: int | None = field(default=5)
    benchmarking_iters: int | None = field(default=10)
    sync_func: Callable | None = field(default=None)
    to_sync: str | None = field(default="forward")

    def __post_init__(self):
        assert self.to_sync in {"forward", "backward", "both"}
        if self.wandb_logging:
            assert self.wandb_project is not None, "wandb_project is required when wandb_logging is True."
            assert self.wandb_run_name is not None, "wandb_run_name is required when wandb_logging is True."
        if self.device == torch.device("cuda"):
            self.sync_func = torch.cuda.synchronize
        elif self.device == torch.device("mps"):
            self.sync_func = torch.mps.synchronize
        if self.precision == "fp16":
            self.train_context_dtype = torch.float16
        elif self.precision == "bf16":
            self.train_context_dtype = torch.bfloat16
        elif self.precision == "fp32":
            self.train_context_dtype = torch.float32
        else:
            raise ValueError("Wrong input for precision.")
