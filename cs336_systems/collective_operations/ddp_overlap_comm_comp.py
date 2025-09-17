import torch
import torch.distributed as dist


class DDPOverlapCommComp(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.handles = []
        for param in self.module.parameters():
            dist.broadcast(param, src=0)
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self.all_reduce_grad)

    def all_reduce_grad(self, param):
        with torch.no_grad():
            param.grad /= dist.get_world_size()
            self.handles.append(dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True))

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
