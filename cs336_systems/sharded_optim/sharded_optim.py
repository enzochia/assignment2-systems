import torch
import torch.distributed as dist
from torch import optim
from cs336_basics.optim import AdamW
from typing import Type, Any, Optional, Callable


class ShardedOptimizer(optim.Optimizer):
    """
    This is a ZeRO stage 1 implementation, not FSDP.
    """
    def __init__(self, 
                 params, 
                 optimizer_cls: Type[optim.Optimizer] = optim.Optimizer,
                 **kwargs: Any):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.param_list = list(params)
        param_group_list = [{"params": [
            param for idx, param in enumerate(self.param_list) if idx % self.world_size == self.rank
        ]}]
        super().__init__(param_group_list, {})
        self.optimizer = optimizer_cls(param_group_list, **kwargs)
        self.handle_list = []


    def step(self,
             closure: Optional[Callable] = None,
             **kwargs: Any):
        self.optimizer.step(closure, **kwargs)
        self.synchronize_parameters()
        self.wait_for_all_params()

    def add_param_group(self, 
                        param_group: dict[str, Any]):
        super().add_param_group(param_group)

    def synchronize_parameters(self):
        with torch.no_grad():
            for idx, param in enumerate(self.param_list):
                rank = idx % self.world_size
                self.handle_list.append(dist.broadcast(param, src=rank, async_op=True))

    def wait_for_all_params(self):
        for handle in self.handle_list:
            handle.wait()
        self.handle_list.clear


