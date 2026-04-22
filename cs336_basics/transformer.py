import torch 
import torch.nn as nn
from einops import rearrange, reduce, einsum
from jaxtyping import Float, Int, Bool
from torch import Tensor

class linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        
        # initialize 
        std = (2 / (out_features + in_features))**0.5
        self.w = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty((out_features, in_features), device=device, dtype=dtype), mean =0, std = std, a = -3 * std, b = 3 * std))

    def forward(self, x: Float[Tensor, "... in_features"]):
        return einsum(x, self.w, "... in_features, out_features in_features -> ... out_features")
    

class embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype |None = None):
        super().__init__()

        # Initialize
        std = 1
        self.embed = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype), mean = 0, std = std, a = -3, b = 3))

    def forward(self, token_ids: Int[Tensor, "batch_size seq_length"]) -> Float[Tensor, "batch_size seq_length embedding_dim"]:
        return self.embed[token_ids]
    

class rmsnorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        # Initialize
        std = (2 / d_model)**0.5
        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(d_model), mean = 0, std = std, a = -3*std, b = 3*std
            )
        )

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        # Upcast to float32
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # RMSNorm calculation
        rms = (einsum(x*x, "... d_model -> ...")/self.d_model + self.eps)**0.5
        rms_broadcasted = rearrange(rms, "... -> ... 1")                
        result = einsum(x/rms_broadcasted, self.g, " ... d_model, ... d_model -> ... d_model")
        return result.to(in_dtype)
        

class positionwise_feedforward(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()

        # Initialize
        self.d_model = d_model
        if not d_ff:
            if round(8/3 * d_model / 64) > 0:
                self.d_ff = round(8/3 * d_model / 64) * 64
            else:
                self.d_ff = 64
        else:
            self.d_ff = d_ff
        
        print(f"dff: {self.d_ff}")
        std = (2 / (self.d_ff + self.d_model))**0.5
        self.w1 = nn.Parameter(         # (d_ff, d_model)
            torch.nn.init.trunc_normal_(
                torch.empty(self.d_ff, self.d_model), mean = 0, std = std, a = -3*std, b = 3*std
            )
        )
        self.w3 = nn.Parameter(          # (d_ff, d_model)
            torch.nn.init.trunc_normal_(
                torch.empty(self.d_ff, self.d_model), mean = 0, std = std, a = -3*std, b = 3*std
            )
        )
        self.w2 = nn.Parameter(          # (d_model, d_ff)
            torch.nn.init.trunc_normal_(
                torch.empty(self.d_model, self.d_ff), mean = 0, std = std, a = -3*std, b = 3*std
            )
        )

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        x_w1 = einsum(x, self.w1, "... d_model, d_ff d_model -> ... d_ff")
        sigmoid_x_w1 = einsum(x_w1, torch.sigmoid(x_w1), "... d_ff, ... d_ff -> ... d_ff")
        x_w3 = einsum(x, self.w3, "... d_model, d_ff d_model -> ... d_ff")
        swiglu = einsum(sigmoid_x_w1, x_w3, "... d_ff, ... d_ff -> ... d_ff")
        return einsum(swiglu, self.w2, "... d_ff, d_model d_ff -> ... d_model")

class rope(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        
        # Precompute all sin and cos values of 
        """
            i ranges from (0, max_seq_len)
            k ranges from (0, d_k/2)
            theta_ik = i / theta**(-2k/d_k)
            theta_ik_matrix shape is (max_seq_len, d_k / 2) 
        """
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        d_k_half = d_k // 2
        k = torch.arange(d_k_half)
        k = -2 / d_k * k
        k = theta**k
        i = torch.arange(max_seq_len)
        matrix_ik = einsum(i, k, "i_dim, k_dim -> i_dim k_dim")
        matrix_sin = torch.sin(matrix_ik)
        matrix_cos = torch.cos(matrix_ik)
        print(f"matrix_ik_sin shape: {matrix_sin.shape}")
        print(f"matrix_ik_cos shape: {matrix_cos.shape}")
        
        self.register_buffer("matrix_sin", matrix_sin, persistent=False)
        self.register_buffer("matrix_cos", matrix_cos, persistent=False)
    
    def forward(self, x: Float[Tensor, "... seq_len d_k"], token_positions: Int[Tensor, "... seq_len"]):
        
        # Rearrange x
        x_new = rearrange(x, "... seq_len (d_k_half two) -> ... seq_len d_k_half two", d_k_half = self.d_k//2, two = 2)
        sin_values = self.matrix_sin[token_positions] # (... seq_len d_k_half)
        cos_values = self.matrix_cos[token_positions] # (... seq_len d_k_half)
        a = x_new[..., 0] # ( ... seq_len d_k_half 1)
        b = x_new[..., 1] # ( ... seq_len d_k_half 1)

        a_new = a * cos_values - b * sin_values 
        b_new = a * sin_values + b * cos_values       
        x_new = torch.stack([a_new, b_new], dim = -1)
        x_new = rearrange(x_new, "... seq_len d_k_half two -> ... seq_len (d_k_half two)", d_k_half = self.d_k//2, two = 2)

        assert x.shape == x_new.shape
        return x_new

def softmax(x: Float[Tensor, "..."], target_dim=int):
        max_values = torch.max(x, dim = target_dim, keepdim = True).values
        stable_x = x - max_values
        exp_stable_x = torch.exp(stable_x)
        denominator = torch.sum(exp_stable_x, dim = target_dim, keepdim=True)
        return exp_stable_x / denominator


def scaled_dot_product_attention(
        Q: Float[Tensor, "batch_size ... seq_len d_k"], 
        K: Float[Tensor, "batch_size ... seq_len d_k"], 
        V: Float[Tensor, "batch_size ... seq_len d_v"], 
        mask: Bool[Tensor, "seq_len seq_len"] | None = None,
):
    d_k = Q.shape[-1]
    qk = einsum(Q, K, "batch_size ... seq_len_q d_k, batch_size ... seq_len_k d_k -> batch_size ... seq_len_q seq_len_k") * d_k**(-0.5)
    
    masked_qk = qk.masked_fill(~mask, float('-inf')) # if mask is True, this value should stay. if mask is False, this should be 0 after softmax, i.e. -inf now
    

    softmax_qk = softmax(masked_qk, target_dim = -1)
    return einsum(softmax_qk, V, "batch_size ... seq_len_q seq_len_k, batch_size ... seq_len_k d_v -> batch_size ... seq_len_q d_v")
