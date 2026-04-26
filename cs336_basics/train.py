import torch 
import torch.nn as nn
from einops import rearrange, reduce, einsum
from jaxtyping import Float, Int, Bool
from torch import Tensor
from collections.abc import Callable, Iterable
from typing import Optional, BinaryIO, IO
import typing
import math
import numpy as np
import numpy.typing as npt
import os

def cross_entropy(logits: Float[Tensor, "... vocab_size"], targets: Int[Tensor, "..."]):
    
    # flatten to 2D, no matter what the input tensors are
    logits = logits.reshape(-1, logits.shape[-1]) # reshape the tensor such that last dimension is logits.shape[-1], ie vocab size, and combine everything else to the other dimension (meaning of first -1)
    uber_batch_size = logits.shape[0]
    
    # flatten to 1D
    targets = targets.reshape(-1) # flatten to a 1D tensor
    
    # numerical stability
    max_values = torch.max(logits, dim = -1, keepdim = True).values
    logits = logits - max_values
    
    first_term = -logits[torch.arange(uber_batch_size), targets]        # (uber_batch_size,)
    second_term = torch.log(torch.sum(torch.exp(logits), dim = -1))     # (uber_batch_size,)
    ce_loss = torch.sum(first_term + second_term) / uber_batch_size
    return ce_loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, weight_decay, eps):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)
        

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
      
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:   # For each param
                if p.grad is None:
                    continue

                state = self.state[p]

                # Initialization
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                
                # Updates for step t. t = 0, ..., T-1
                t = state["t"]      # int
                lr_adapt = lr * (1 - beta2**(t+1))**0.5 * (1 - beta1**(t+1))**(-1)
                p.data -= lr * weight_decay * p.data
                state["m"] = beta1 * state["m"] + (1 - beta1) * p.grad
                state["v"] = beta2 * state["v"] + (1 - beta2) * p.grad**2
                p.data -= lr_adapt * state["m"] / (state["v"]**0.5 + eps)
                state["t"] = t + 1
                
        return loss

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]    # Get LR
            for p in group["params"]:   # For each param
                if p.grad is None:
                    continue

                state = self.state[p]   # Get state associated with p
                t = state.get("t", 0)   # GEt iteration number from the state, or 0
                grad = p.grad.data      # Get gradients of loss wrt p
                p.data -= lr / math.sqrt(t +1) * grad
                state["t"] = t + 1
        return loss

def learning_rate_scheduler(
        t: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int, 
        cosine_cycle_iters: int,
):
    if t < warmup_iters:
        lr = t / warmup_iters * max_learning_rate
    elif t <= cosine_cycle_iters:
        lr = min_learning_rate + (max_learning_rate - min_learning_rate) * (1 + math.cos((t - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) / 2
    else:
        lr = min_learning_rate
    return lr

def gradient_clipping(params, max_l2_norm):
    eps = 10**(-6)
    total_l2_norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params if p.grad is not None))
    if total_l2_norm >= max_l2_norm:
        scaling_factor = max_l2_norm / (total_l2_norm + eps)
        for p in params: 
            if p.grad is not None:
                p.grad = p.grad * scaling_factor
    return total_l2_norm
    
def data_loader(dataset: np.ndarray, batch_size: int, context_length: int, device: str):
    
    dataset_size = len(dataset)
    
    inputs = []
    targets = []
    for _ in range(batch_size):
        start_index = np.random.randint(0, dataset_size - context_length)
        inputs.append(dataset[start_index:start_index + context_length])
        targets.append(dataset[start_index + 1:start_index + context_length + 1])
    
    inputs = torch.from_numpy(np.stack(inputs)).long().to(device)   # (batch_size, context_length)
    targets = torch.from_numpy(np.stack(targets)).long().to(device)   # (batch_size, context_length)
    
    return (inputs, targets)


def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    states = {}
    states["model"] = model.state_dict()
    states["optimizer"] = optimizer.state_dict()
    states["iteration"] = iteration
    torch.save(states, out)

def load_checkpoint(
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
):
    states = torch.load(src)
    model.load_state_dict(states["model"])
    if optimizer is not None:
        optimizer.load_state_dict(states["optimizer"])
        return states["iteration"]


    
