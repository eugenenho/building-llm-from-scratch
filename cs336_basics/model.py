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
        silu = einsum(x_w1, torch.sigmoid(x_w1), "... d_ff, ... d_ff -> ... d_ff")
        x_w3 = einsum(x, self.w3, "... d_model, d_ff d_model -> ... d_ff")
        swiglu = einsum(silu, x_w3, "... d_ff, ... d_ff -> ... d_ff")
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
        # print(f"matrix_ik_sin shape: {matrix_sin.shape}")
        # print(f"matrix_ik_cos shape: {matrix_cos.shape}")
        
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

        # assert x.shape == x_new.shape
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

class multihead_self_attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int | None = None, theta: float | None = None,):
        super().__init__()
        
        # Initialization
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = int(d_model / num_heads)
        assert d_model % num_heads == 0
        self.d_v = self.d_k
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        std = (2 / (num_heads * self.d_k + d_model))**0.5
        self.w_q = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(num_heads * self.d_k, d_model), mean = 0, std = std, a = -3*std, b = 3*std
            )
        )
        self.w_k = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(num_heads * self.d_k, d_model), mean = 0, std = std, a = -3*std, b = 3*std
            )
        )
        self.w_v = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(num_heads * self.d_v, d_model), mean = 0, std = std, a = -3*std, b = 3*std
            )
        )
        self.w_o = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(d_model, num_heads * self.d_v), mean = 0, std = std, a = -3*std, b = 3*std
            )
        )
        if self.theta is not None:
            self.rope = rope(theta = self.theta, d_k = self.d_k, max_seq_len = self.max_seq_len)
        
    def forward(self, x: Float[Tensor, "... seq_len d_model"], token_positions: Int[Tensor, " ... sequence_length"] | None = None,)-> Float[Tensor, "... seq_len d_model"]:
        
        # Combine into one large matrix so Q, K, V can be calculated with one mat mul
        # print(f"\n\n x shape:{x.shape}")
        # print(f"self.w_q shape: {self.w_q.shape}")
        w_qkv = torch.cat([self.w_q, self.w_k, self.w_v], dim = 0)                          # (3 * num_heads * self.d_k, d_model)
        # print(f"wqkv shape: {w_qkv.shape}")
        matmulresult = einsum(x, w_qkv, "... d_model, three_dim d_model -> ... three_dim")  # where three_dim = 3 * num_heads * self.d_k
        # print(f"matmulresult shape: {matmulresult.shape}")
        Q, K, V = torch.chunk(matmulresult, 3, dim=-1)
        # print(f"Q, K, V shape: {Q.shape}, {K.shape}, {V.shape}")
                
        seq_len = x.shape[-2]
        
        
        # if not self.theta: self.theta = 10000
        # if not self.max_seq_len: self.max_seq_len = seq_len
        # if not token_positions: self.token_positions = torch.arange(seq_len)
        
        Q = rearrange(Q, "... seq_len (num_heads d_k) -> ... seq_len num_heads d_k", num_heads = self.num_heads, d_k = self.d_k)
        Q = rearrange(Q, "... seq_len num_heads d_k -> ... num_heads seq_len d_k", num_heads = self.num_heads, d_k = self.d_k)
        K = rearrange(K, "... seq_len (num_heads d_k) -> ... seq_len num_heads d_k", num_heads = self.num_heads, d_k = self.d_k)
        K = rearrange(K, "... seq_len num_heads d_k -> ... num_heads seq_len d_k", num_heads = self.num_heads, d_k = self.d_k)
        V = rearrange(V, "... seq_len (num_heads d_v) -> ... seq_len num_heads d_v", num_heads = self.num_heads, d_v = self.d_v)
        V = rearrange(V, "... seq_len num_heads d_v -> ... num_heads seq_len d_v", num_heads = self.num_heads, d_v = self.d_v)
        
        if self.theta is not None:
            if token_positions is None: token_positions = torch.arange(seq_len, device = Q.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        

        # print(f"After rearrange, Q, K, V shape: {Q.shape}, {K.shape}, {V.shape}")
        mask = torch.ones(seq_len, seq_len, device = Q.device).tril().bool()   # (seq_len, seq_len) matrix. True at the bottom triangle. True indicating, signal passing
        attention = scaled_dot_product_attention(Q, K, V, mask) # "... num_heads seq_len d_v"
        # print(f"attention shape: {attention.shape}")
        attention = rearrange(attention, "... num_heads seq_len d_v -> ... seq_len num_heads d_v")
        attention = rearrange(attention, "... seq_len num_heads d_v -> ... seq_len (num_heads d_v)")
        # print(f"AFter rearange, attention shape: {attention.shape}")
        result = einsum(attention, self.w_o, "... seq_len num_heads_d_v, d_model num_heads_d_v -> ... seq_len d_model")
        # print(f"result shape: {result.shape}")
        return result


class transformer_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # Create MHA and FFN objects
        self.rmsnorm1 = rmsnorm(d_model = d_model)
        self.mha = multihead_self_attention(d_model = d_model, num_heads = num_heads, max_seq_len = max_seq_len, theta = theta)
        self.rmsnorm2 = rmsnorm(d_model = d_model)
        self.ffn = positionwise_feedforward(d_model = d_model, d_ff = d_ff)
    
    def forward(self, x: Float[Tensor, "batch_size ... seq_len d_model"]) -> Float[Tensor, "batch_size ... seq_len d_model"]:
        
        x = x + self.mha(self.rmsnorm1(x))
        x = x + self.ffn(self.rmsnorm2(x))
        return x


class transformer_lm(nn.Module):
    def __init__(
            self, 
            d_model: int, 
            num_heads: int, 
            d_ff: int, 
            context_length: int, 
            rope_theta: float, 
            vocab_size: int, 
            num_layers: int
    ):
        super().__init__()

        self.context_length = context_length

        # Create vocab embedding
        self.token_embedding = embedding(num_embeddings = vocab_size, embedding_dim = d_model)
        
        # Create transformer blocks
        self.blocks = nn.ModuleList([
            transformer_block(d_model = d_model, num_heads = num_heads, d_ff = d_ff, max_seq_len = context_length, theta = rope_theta) for _ in range(num_layers)
        ])
            
        # Create final norm
        self.final_norm = rmsnorm(d_model = d_model)

        # Create LM Head
        self.lm_head = linear(in_features = d_model, out_features = vocab_size)

    def forward(self, x: Int[Tensor, "batch_size ... seq_len"]) -> Float[Tensor, "batch_size ... seq_len vocab_size"]:
        
        # get token embeddings
        x = self.token_embedding(x) # (batch_size, seq_len, d_model)
        
        # transformer blocks
        for block in self.blocks:
            x = block(x)

        # final norm layer
        x = self.final_norm(x)  # (batch_size, ... , seq_len, d_model)

        # LM Head
        logits = self.lm_head(x)     # (batch_size, ... , seq_len, vocab_size)

        return logits 

            

        
        

        
        