import torch
import torch.distributed as dist


class DDPOverlapCommComp(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.handles = []
        for param in self.module.parameters():
            with torch.no_grad():
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
        self.handles.clear()


class Buckets():
    def __init__(self, bucket_size_mb: float):
        self.bucket_size_mb = bucket_size_mb * (2 ** 20)
        self.bucket_list = [[]]
        self.handle_list = []
        self.param_to_bucket = {}
        self.bucket_param_count = [{"param_count": 0,
                                    "backward_count": 0}]
    
    def add_param(self, module):
        bucket_mb_count = 0
        for param in module.parameters():
            if param.requires_grad:
                if bucket_mb_count + param.data.nbytes <= self.bucket_size_mb:
                    bucket_mb_count += param.data.nbytes
                else:
                    bucket_mb_count = param.data.nbytes
                    self.bucket_list.append([])
                    self.bucket_param_count.append({"param_count": 0,
                                                    "backward_count": 0})
                self.bucket_list[-1].append(param)
                self.bucket_param_count[-1]["param_count"] += 1
                self.param_to_bucket[param] = len(self.bucket_list) - 1
                param.register_post_accumulate_grad_hook(self.try_to_reduce)

    def try_to_reduce(self, param):
        bucket_id = self.param_to_bucket[param]
        self.bucket_param_count[bucket_id]["backward_count"] += 1
        if self.bucket_param_count[bucket_id]["backward_count"] == \
           self.bucket_param_count[bucket_id]["param_count"]:
            self.bucket_param_count[bucket_id]["backward_count"] = 0
            self.all_reduce_bucket(bucket_id)
    
    def all_reduce_bucket(self, bucket_id):
        with torch.no_grad():
            grad_tensor_template = [param.grad for param in self.bucket_list[bucket_id] if param.grad is not None]
            flattened_param = torch._utils._flatten_dense_tensors(tensors=grad_tensor_template)
            dist.all_reduce(flattened_param, op=dist.ReduceOp.SUM, async_op=False)
            unflattened_params = torch._utils._unflatten_dense_tensors(flattened_param, tensors=grad_tensor_template)
            for param, param_tensor in zip(
                (param for param in self.bucket_list[bucket_id] if param.grad is not None),  
                unflattened_params):
                param.grad.copy_(param_tensor / dist.get_world_size())

    def finish_gradient_synchronization(self):
        for handle in self.handle_list:
            handle.wait()
        self.handle_list.clear()


class DDPOverlapBucketed(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.buckets = Buckets(bucket_size_mb)
        for param in self.module.parameters():
            with torch.no_grad():
                dist.broadcast(param, src=0)
        self.buckets.add_param(self.module)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        self.buckets.finish_gradient_synchronization()