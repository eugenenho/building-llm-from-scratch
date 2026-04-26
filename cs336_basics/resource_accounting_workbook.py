import torch 
import torch.nn as nn
from einops import rearrange, reduce, einsum
from jaxtyping import Float, Int, Bool
from torch import Tensor




class resource_account:
    def __init__(self, name: str, vocab_size: int, context_length: int, num_layer: int, d_model: int, num_heads: int, d_ff: int | None = None):
        self.name = name
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layer = num_layer
        self.d_model = d_model
        self.num_heads = num_heads
        if not d_ff:
            self.d_ff = round(8/3 * d_model / 64) * 64
        else:
            self.d_ff = d_ff

        assert d_model % num_heads == 0
        self.d_k = int(d_model / num_heads)

        # Assumptions
        self.batch_size = 1
        self.seq_len = self.context_length
        self.d_v = self.d_k

    def calc_params(self):
        
        params = {}
        params["token_embeddings"] = self.vocab_size * self.d_model
        
        for i in range(self.num_layer):
            params[f"block.{i}.rmsnorm1"] = self.d_model
            params[f"block.{i}.mha.w_q"] = self.d_model * self.d_model
            params[f"block.{i}.mha.w_k"] = self.d_model * self.d_model
            params[f"block.{i}.mha.w_v"] = self.d_model * self.d_model
            params[f"block.{i}.mha.w_o"] = self.d_model * self.d_model
            params[f"block.{i}.rmsnorm2"] = self.d_model
            params[f"block.{i}.ffn.w1"] = self.d_ff * self.d_model
            params[f"block.{i}.ffn.w2"] = self.d_ff * self.d_model
            params[f"block.{i}.ffn.w3"] = self.d_ff * self.d_model
        
        params["final_norm"] = self.d_model
        params["lm_head"] = self.d_model * self.vocab_size

        print(f"Parameters for {self.name}\n")
        print(f"Total count: {sum(params.values()):,}\n")
        print(f"Total memory for params (assuming single precision, ie float32): {4 * sum(params.values()):,} bytes")
        # for k, v in params.items():
        #     print(f"{k}: {v:,}")
        return params
    
    def calc_flops_forward(self, batch_size: int | None = None):
        
        if batch_size is None:
            batch_size = self.batch_size

        matmuls = {}

        matmuls[("block.mha", "(x, w_qkv, 'batch_size seq_len d_model, d_model 3x_d_model -> batch_size d_model seq_len 3x_d_model')")] = (self.num_layer, self.num_layer * 2 * batch_size * self.seq_len * self.d_model * 3 * self.d_model)
        matmuls[("block.mha", "(Q, K, 'batch_size num_heads seq_len1 d_k, batch_size num_heads seq_len2 d_k -> batch_size num_heads seq_len1 seq_len2')")] = (self.num_layer, self.num_layer * 2 * batch_size * self.num_heads * self.seq_len**2 * self.d_k)
        matmuls[("block.mha", "(softmax(QK), V, 'batch_size num_heads seq_len1 seq_len2, batch_size num_heads seq_len1 d_v -> batch_size num_heads seq_len1 d_v')")] = (self.num_layer, self.num_layer * 2 * batch_size * self.num_heads * self.seq_len * self.d_v * self.seq_len)
        matmuls[("block.mha", "(attn, w_o, 'batch_size seq_len (num_heads d_v), d_model (num_heads d_v) -> batch_size seq_len d_model')")] = (self.num_layer, self.num_layer * 2 * batch_size * self.seq_len * self.d_model * (self.num_heads * self.d_v))
        matmuls[("block.ffn", "(x, w1, 'batch_size seq_len d_model, d_ff d_model -> batch_size seq_len d_ff')")] = (self.num_layer, self.num_layer * 2 * batch_size * self.seq_len * self.d_ff * self.d_model)
        matmuls[("block.ffn", "(x, w3, 'batch_size seq_len d_model, d_ff d_model -> batch_size seq_len d_ff')")] = (self.num_layer, self.num_layer * 2 * batch_size * self.seq_len * self.d_ff * self.d_model)
        matmuls[("block.ffn", "(swiglu, w2, 'batch_size seq_len d_ff, d_model d_ff -> batch_size seq_len d_model')")] = (self.num_layer, self.num_layer * 2 * batch_size * self.seq_len * self.d_model * self.d_ff)
    
        matmuls[("lm_head", "(x, w, 'batch_size seq_len d_model, d_model vocab_size -> batch_size seq_len vocab_size)")] = (1, 2 * batch_size * self.seq_len * self.vocab_size * self.d_model)

        print("--------------------------\n")
        print(f"Matmuls and FLOPs for {self.name}\n")
        print(f"Total matmuls: {sum(v[0] for v in matmuls.values()):,}\n")
        print(f"Total Flops: {sum(v[1] for v in matmuls.values()):,}\n")
        for k, v in matmuls.items():
            print(f"{str(k):<150}: {v[0]:>3} matmuls, with {v[1]:>20,} flops")

        return matmuls
    
    def calc_memory(self):
        
        # This is first calculated assuming batch_size = 1
        batch_size = self.batch_size
        
        # Calculate P
        P = 2 * self.vocab_size * self.d_model + self.d_model + 2 * self.num_layer * self.d_model + 12 * self.num_layer * self.d_model**2
        # Calculate G
        G = P
        # Calculate O
        O = 2 * P
        # Calculate A
        A = (56 / 3) * self.num_layer * batch_size * self.context_length * self.d_model + 2 * self.num_layer * batch_size * self.num_heads * self.context_length**2 + batch_size * self.context_length * (self.d_model + 2 * self.vocab_size)
        # Calculate total elements
        total_elements = P + G + O + A
        # Calculate byte total estiamte
        total_bytes = 4 * total_elements
        fixed_bytes = 4 * 4 * P
        per_batch_bytes = 4 * A


        
        print(f"Memory calculation: {self.name}, assuming batch_size = 1")
        print(f"total elements needed: {total_elements:<20,} total bytes: {total_bytes:>10,}")

        # Assignment-specific calc
        # Assuming 80 GB of memory, max batch size
        memory_limit = 85899345920
        max_batch_size = (memory_limit - fixed_bytes) / per_batch_bytes
        print(f"assuming total memory limit of 80GB, max batch size is: {max_batch_size}")

        # For H200
        # Assuming 140 GB of memory, max batch size 
        memory_limit = 150323855872
        max_batch_size = (memory_limit - fixed_bytes) / per_batch_bytes
        print(f"assuming total memory limit of 140GB, max batch size is: {max_batch_size}")
        
    def calc_flops_total(self):
        # Assuming backward pass takes 2x flops as forward pass
        
        forward_flops_result = self.calc_flops_forward()
        forward_flops = sum(v[1] for v in forward_flops_result.values())
        backward_flops = 2 * forward_flops
        params = self.calc_params()
        optimizer_flops = 13 * sum(params.values())
        
        total_per_step_flops = forward_flops + backward_flops + optimizer_flops
        print(f"FLOPS: for {self.name}, assuming batch_size = 1")
        print(f"forward flops: {forward_flops:,} flops")
        print(f"backward flops: {backward_flops:,} flops")
        print(f"optimizer flops: {optimizer_flops:,} flops")
        print(f"total_per_step flops: {total_per_step_flops:,} flops")

        # Assuming
        """
            H100 GPU theoretical peak FLOP throughput: 495 teraFLOP/s for “float32” 
            MFU: 50%
            1x H100 GPU
            400K steps
            batch_size = 1024
            How long would it take?
        """
        h100_theo_FLOPS = 4.95e14                   # FLOPS (flop/s)
        mfu = 0.5
        h100_throughput = h100_theo_FLOPS * mfu     # FLOPS (flop/s)

        forward_flops_result = self.calc_flops_forward(batch_size=1024)
        forward_flops = sum(v[1] for v in forward_flops_result.values())
        backward_flops = 2 * forward_flops
        params = self.calc_params()
        optimizer_flops = 13 * sum(params.values())
        
        total_per_step_flops = forward_flops + backward_flops + optimizer_flops
        total_flops_for_training = total_per_step_flops * 400
        time_needed = total_flops_for_training / h100_throughput / 3600   # hours

        print(f"FLOPS: for {self.name}, assuming batch_size = 1024")
        print(f"forward flops: {forward_flops:,} flops")
        print(f"backward flops: {backward_flops:,} flops")
        print(f"optimizer flops: {optimizer_flops:,} flops")
        print(f"total_per_step flops: {total_per_step_flops:,} flops")
        print(f"total flops needed for training: {total_flops_for_training:,} flops")
        print(f"hours needed for training: {time_needed:,} hours")



        
        
    
if __name__ == "__main__":

    gpt_2_small = resource_account(
        name = "gpt-2 small",
        vocab_size = 50257,
        context_length	= 1024,
        num_layer = 12,
        d_model = 768,
        num_heads = 12,
        d_ff = None
    )
    gpt_2_medium = resource_account(
        name = "gpt-2 medium",
        vocab_size = 50257,
        context_length	= 1024,
        num_layer = 24,
        d_model = 1024,
        num_heads = 16,
        d_ff = None
    )

    gpt_2_large = resource_account(
        name = "gpt-2 large",
        vocab_size = 50257,
        context_length	= 1024,
        num_layer = 36,
        d_model = 1280,
        num_heads = 20,
        d_ff = None
    )

    gpt_2_xl = resource_account(
        name = "gpt-2 xl",
        vocab_size = 50257,
        context_length	= 1024,
        num_layer = 48,
        d_model = 1600,
        num_heads = 25,
        d_ff = 4288
    ) 
   
    gpt_2_xl_lc = resource_account(
        name = "gpt-2 xl long context",
        vocab_size = 50257,
        context_length	= 16384,
        num_layer = 48,
        d_model = 1600,
        num_heads = 25,
        d_ff = 4288
    ) 
   
    ts_model = resource_account(
        name = "tinystories_model",
        vocab_size = 10000,
        context_length	= 256,
        num_layer = 4,
        d_model = 512,
        num_heads = 16,
        d_ff = 1344
    ) 

    gpt_2_small.calc_params()
    small_flops = gpt_2_small.calc_flops_forward()
    
    print("-------------------------------------------------\n")
    gpt_2_medium.calc_params()
    medium_flops = gpt_2_medium.calc_flops_forward()
    print("-------------------------------------------------\n")
    gpt_2_large.calc_params()
    large_flops = gpt_2_large.calc_flops_forward()
    print("-------------------------------------------------\n")
    gpt_2_xl.calc_params()
    xl_flops = gpt_2_xl.calc_flops_forward()
    print("-------------------------------------------------\n")
    gpt_2_xl_lc.calc_params()
    xl_lc_flops = gpt_2_xl_lc.calc_flops_forward()
    print("-------------------------------------------------\n")
    
    small_total_flop = sum(v[1] for v in small_flops.values())
    small_sum_block_mha = sum(v[1] for k, v in small_flops.items() if k[0] == "block.mha")
    small_sum_block_ffn = sum(v[1] for k, v in small_flops.items() if k[0] == "block.ffn")
    small_sum_lm_head = sum(v[1] for k, v in small_flops.items() if k[0] == "lm_head")

    print(f"Small: total forward flops: {small_total_flop:,}")
    print(f"{'Block.MHA:':<12} {small_sum_block_mha:<20,} flops  | {small_sum_block_mha/small_total_flop*100:>6.2f}% of total forward flops")
    print(f"{'Block.FFN:':<12} {small_sum_block_ffn:<20,} flops  | {small_sum_block_ffn/small_total_flop*100:>6.2f}% of total forward flops")
    print(f"{'LM Head:':<12} {small_sum_lm_head:<20,} flops  | {small_sum_lm_head/small_total_flop*100:>6.2f}% of total forward flops")
    
    medium_total_flop = sum(v[1] for v in medium_flops.values())
    medium_sum_block_mha = sum(v[1] for k, v in medium_flops.items() if k[0] == "block.mha")
    medium_sum_block_ffn = sum(v[1] for k, v in medium_flops.items() if k[0] == "block.ffn")
    medium_sum_lm_head = sum(v[1] for k, v in medium_flops.items() if k[0] == "lm_head")

    print(f"\nMedium: total forward flops: {medium_total_flop:,}")
    print(f"{'Block.MHA:':<12} {medium_sum_block_mha:<20,} flops  | {medium_sum_block_mha/medium_total_flop*100:>6.2f}% of total forward flops")
    print(f"{'Block.FFN:':<12} {medium_sum_block_ffn:<20,} flops  | {medium_sum_block_ffn/medium_total_flop*100:>6.2f}% of total forward flops")
    print(f"{'LM Head:':<12} {medium_sum_lm_head:<20,} flops  | {medium_sum_lm_head/medium_total_flop*100:>6.2f}% of total forward flops")

    large_total_flop = sum(v[1] for v in large_flops.values())
    large_sum_block_mha = sum(v[1] for k, v in large_flops.items() if k[0] == "block.mha")
    large_sum_block_ffn = sum(v[1] for k, v in large_flops.items() if k[0] == "block.ffn")
    large_sum_lm_head = sum(v[1] for k, v in large_flops.items() if k[0] == "lm_head")

    print(f"\nLarge: total forward flops: {large_total_flop:,}")
    print(f"{'Block.MHA:':<12} {large_sum_block_mha:<20,} flops  | {large_sum_block_mha/large_total_flop*100:>6.2f}% of total forward flops")
    print(f"{'Block.FFN:':<12} {large_sum_block_ffn:<20,} flops  | {large_sum_block_ffn/large_total_flop*100:>6.2f}% of total forward flops")
    print(f"{'LM Head:':<12} {large_sum_lm_head:<20,} flops  | {large_sum_lm_head/large_total_flop*100:>6.2f}% of total forward flops")

    xl_total_flop = sum(v[1] for v in xl_flops.values())
    xl_sum_block_mha = sum(v[1] for k, v in xl_flops.items() if k[0] == "block.mha")
    xl_sum_block_ffn = sum(v[1] for k, v in xl_flops.items() if k[0] == "block.ffn")
    xl_sum_lm_head = sum(v[1] for k, v in xl_flops.items() if k[0] == "lm_head")

    print(f"\nXL: total forward flops: {xl_total_flop:,}")
    print(f"{'Block.MHA:':<12} {xl_sum_block_mha:<20,} flops  | {xl_sum_block_mha/xl_total_flop*100:>6.2f}% of total forward flops")
    print(f"{'Block.FFN:':<12} {xl_sum_block_ffn:<20,} flops  | {xl_sum_block_ffn/xl_total_flop*100:>6.2f}% of total forward flops")
    print(f"{'LM Head:':<12} {xl_sum_lm_head:<20,} flops  | {xl_sum_lm_head/xl_total_flop*100:>6.2f}% of total forward flops")

    xl_lc_total_flop = sum(v[1] for v in xl_lc_flops.values())
    xl_lc_sum_block_mha = sum(v[1] for k, v in xl_lc_flops.items() if k[0] == "block.mha")
    xl_lc_sum_block_ffn = sum(v[1] for k, v in xl_lc_flops.items() if k[0] == "block.ffn")
    xl_lc_sum_lm_head = sum(v[1] for k, v in xl_lc_flops.items() if k[0] == "lm_head")

    print(f"\nXL Long Context: total forward flops: {xl_lc_total_flop:,}")
    print(f"{'Block.MHA:':<12} {xl_lc_sum_block_mha:<20,} flops  | {xl_lc_sum_block_mha/xl_lc_total_flop*100:>6.2f}% of total forward flops")
    print(f"{'Block.FFN:':<12} {xl_lc_sum_block_ffn:<20,} flops  | {xl_lc_sum_block_ffn/xl_lc_total_flop*100:>6.2f}% of total forward flops")
    print(f"{'LM Head:':<12} {xl_lc_sum_lm_head:<20,} flops  | {xl_lc_sum_lm_head/xl_lc_total_flop*100:>6.2f}% of total forward flops")

    gpt_2_xl.calc_memory()
    gpt_2_xl.calc_flops_total()

    ts_model.calc_memory()
