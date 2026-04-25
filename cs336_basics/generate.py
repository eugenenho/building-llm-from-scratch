import torch
from cs336_basics.model import transformer_lm, softmax
from cs336_basics.train import AdamW
from cs336_basics.train import data_loader, learning_rate_scheduler, gradient_clipping, save_checkpoint, load_checkpoint, cross_entropy
from cs336_basics.tokenizer import Tokenizer
import argparse
import yaml
from datetime import datetime
import numpy as np
from pathlib import Path
import wandb
import math
from jaxtyping import Float, Int, Bool
from torch import Tensor


def sample_top_p(
    probs: Float[Tensor, "... vocab_size"],
    top_p: float,
):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)                                # (... vocab_size) for both
    cumul_probs = torch.cumsum(sorted_probs, dim=-1)                                                        # (... vocab_size)
    rshifted_probs = torch.cat([torch.zeros_like(cumul_probs[..., :1]), cumul_probs[..., :-1]], dim = -1)   # (... vocab_size)
    mask = rshifted_probs < top_p                                                                           # (... vocab_size)
    sorted_probs = sorted_probs * mask                                                                      # (... vocab_size)
    sorted_probs = sorted_probs / sorted_probs.sum(dim = -1, keepdim=True)
    sampled_in_sorted = torch.multinomial(input=sorted_probs, num_samples=1)                                # returns (... 1) tensor, sampled index for each batch, if batches exist
    sampled_original_indices = sorted_indices.gather(dim = -1, index = sampled_in_sorted)                   # (... 1) tensor. for each batch, sampled index (according to original index, not sorted). sorted_indices: (... vocab_size), sampled_in_sorted (... 1) 
    return sampled_original_indices    

def generate(
    model: torch.nn.Module,
    prompt_ids: Float[Tensor, "... seq_len"],
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    eot_id: int | None = None,        
):
    """
    Takes as inputs:
        - model
        - tokenizer
        - prompt_ids: tensor of token IDs for the prompt
        - max_tokens
        - eot_id (end of text token ID)
        - temperature
        - top_p
        - seed

    Returns:
        - output_ids: tensor of token IDs for generated output
    """
    torch.manual_seed(seed)
    was_training = model.training
    model.eval()
    
    context_length = model.context_length
    current_seq = prompt_ids              # (... seq_len)
    prompt_len = prompt_ids.shape[-1]       # seq_len of the prompt
    
    with torch.no_grad():
        for _ in range(max_tokens):
            
            model_input = current_seq[...,-context_length:]                         # Always just grab the latest context_length size chunk
            logits = model(model_input)                                             # (... seq_len vocab_size)
            logits_last_token = logits[..., -1, :]                                  # (... vocab_size)
            probs = softmax(logits_last_token / temperature, target_dim=-1)         # (... vocab_size). temperature-treated probs
            next_token = sample_top_p(probs, top_p)                          # (... vocab_size). token_ids for the sampled vocab
            
            # Note: assumes one prompt generation at a time, even though other parts of code allowed for batch
            if eot_id is not None and eot_id == next_token.item():
                break
            
            current_seq = torch.cat([current_seq, next_token], dim= -1)
        
    output = current_seq[..., prompt_len:]
    if was_training: model.train()
    return output
        
if __name__ == "__main__":
    
    """
    Inputs needed to generate:
        checkpoint: str
        prompt: str
        max_tokens: int
        temperature (tau): float
        top_p: float
        seed: int
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="model checkpoint to load")
    parser.add_argument("--prompt", type=str, required=True, help="user prompt")
    parser.add_argument("--max-tokens", type=int, default=256, help="max number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.5, help="temperature for softmax")
    parser.add_argument("--top-p", type=float, default=0.9, help="top p sampling for inference")
    parser.add_argument("--seed", type=int, default=42, help="torch seed for generation determinism")
    args_dict = vars(parser.parse_args())
    
    checkpoint_path = args_dict["checkpoint"]
    config_path = Path(checkpoint_path).parent / "config.yaml"
    prompt = args_dict["prompt"]
    max_tokens = args_dict["max_tokens"]
    temperature = args_dict["temperature"]
    top_p = args_dict["top_p"]
    seed = args_dict["seed"]

    nested_hparams = yaml.safe_load(open(config_path)) # dict (nested)
    hparams = {}
    hparams = {k:v for _, group in nested_hparams.items() for k, v in group.items()}    # dict (Flattened)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # PART 2: LOAD THE MODEL
    
    # Create model and optimizer objects
    model = transformer_lm(
            d_model = hparams["d_model"], 
            num_heads = hparams["num_heads"], 
            d_ff = hparams["d_ff"], 
            context_length = hparams["context_length"], 
            rope_theta = hparams["rope_theta"], 
            vocab_size = hparams["vocab_size"], 
            num_layers = hparams["num_layers"],
    )
    model.to(device)
    load_checkpoint(src = checkpoint_path, model = model)

    
    # PART 3: PROCESS THE PROMPT
    tokenizer = Tokenizer.from_files(                                    # for sampling, for vocab_size
        vocab_filepath = hparams["vocab_path"], 
        merges_filepath = hparams["merges_path"], 
        special_tokens=hparams["special_tokens"]
    ) 
    eot_id = tokenizer.reverse_vocab["<|endoftext|>".encode("utf-8")]                   # end of text token: hard coded!
    prompt_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device) # tensor (seq_len)
    prompt_ids = prompt_ids.unsqueeze(0)                                                # tensor (1, seq_len). batch_size == 1
    prompt_len = prompt_ids.shape[-1]                                                   # seq_len of prompt
    
    # check if context length limit is violated
    if prompt_len > model.context_length:
        raise ValueError(f"prompt exceeds the context window limit of the model")
        
    # PART 4: Generation
    output_ids = generate(                          # tensor (1, seq_len). assuming batch_size = 1
        model=model, 
        prompt_ids = prompt_ids,
        max_tokens = max_tokens,
        temperature = temperature,
        top_p = top_p,
        seed = seed,
        eot_id = eot_id,
    )
    output_ids = output_ids.squeeze(0)              # tensor (seq_len)
    output = tokenizer.decode(output_ids.tolist())
    print(f"Prompt: {prompt}\n\n")
    print(f"Output: {output}")



