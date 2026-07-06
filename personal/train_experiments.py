from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

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


def small_training_ex(lr: float):
    weights =torch.nn.Parameter(5 * torch.randn(10,10))
    opt = SGD([weights], lr=lr)
    print(f"Learning rate: {lr}")
    for t in range(100):
        opt.zero_grad()
        loss = (weights**2).mean()
        if t+1 == 1 or (t+1)%10 ==0: 
            print(f"step: {t+1}   loss: {loss.cpu().item()}")
        loss.backward()
        opt.step()
    print("\n--------------\n")


if __name__ == "__main__":
    
    small_training_ex(1e1)
    small_training_ex(1e2)
    small_training_ex(1e3)