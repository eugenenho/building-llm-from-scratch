import torch 
import torch.nn as nn
from einops import rearrange, reduce, einsum
from jaxtyping import Float, Int, Bool
from torch import Tensor

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
    
