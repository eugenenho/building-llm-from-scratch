import torch
from cs336_basics.model import transformer_lm
from cs336_basics.train import AdamW
from cs336_basics.train import data_loader, learning_rate_scheduler, gradient_clipping, save_checkpoint, load_checkpoint, cross_entropy
from cs336_basics.tokenizer import Tokenizer
import argparse
import yaml
from datetime import datetime
import numpy as np




if __name__ == "__main__":
    
    # PART 1: GET HYPERPARAMETERS
    """ 
    Full set of hyperparameters needed for training:(example)
        model:
            batch_size: 32
            context_length: 256
            d_model: 512
            num_layers: 4
            num_heads: 16 
            rope_theta: 10000
            d_ff: 1344

        training:
            device: 
            steps: 5000
            warmup_tiers: 250
            cosine_cycle_iters: 5000
            max_lr: 1e-3
            min_lr: 1e-4
            weight_decay: 0.01
            betas:
                - 0.9
                - 0.95
            max_grad_norm: 1.0

        data:
            train_data_path: data/TinyStoriesV2-GPT4-train-encoded.npy
            valid_data_path: data/TinyStoriesV2-GPT4-valid-encoded.npy
        
        tokenizer:
            vocab_path: outputs_tinystories/vocab.json
            merges_path: outputs_tinystories/merges.txt 
            special_tokens: ["<|endoftext|>",]

        checkpoint:
            output_dir: checkpoints/tinystories
            run_name: null
            save_every: 1000        
    
    """
        
    # Step 1: Read from YAML first
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="config file path")
    known_args, _ = parser.parse_known_args()
    config_path = known_args.config
    nested_hparams = yaml.safe_load(open(config_path))
    hparams = {}
    hparams = {k:v for _, group in nested_hparams.items() for k, v in group.items()}
    
    # Step 2: Check for overrides from CLI
    parser.add_argument("--run-name", default=argparse.SUPPRESS, type=str, help="name of the run")
    parser.add_argument("--lr-max", default=argparse.SUPPRESS, type=float, help="max learning rate for the lr scheduler")
    parser.add_argument("--lr-min", default=argparse.SUPPRESS, type=float, help="min learning rate for the lr scheduler")
    parser.add_argument("--steps", default=argparse.SUPPRESS, type=int, help="total steps for training")
    parser.add_argument("--batch-size", default=argparse.SUPPRESS, type=int, help="batch size for training")
    parser.add_argument("--context-length", default=argparse.SUPPRESS, type=int, help="max sequence length for the model")
    args = parser.parse_args()
    cli_overrides = vars(args) # dict of only what the user passed on CLI, overriding YAML

    for k, v in cli_overrides.items():
        if k in hparams and hparams[k] != v:
            print(f"CLI override: {k} = {v} (from YAML: {hparams[k]})")
            hparams[k] = v  
    
    # Step 3: Check if any of the fields are empty
    missing_okay = ["run_name", "d_ff"]
    for k, v in hparams.items():
        if v is None and k not in missing_okay: 
            raise ValueError(f"Critical hyperparameter missing: {k: v}")

    if hparams["d_ff"] is None:
        hparams["d_ff"] = round(8/3 * hparams["d_model"] / 64) * 64 
    
    if hparams["run_name"] is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        hparams["run_name"] = f"{timestamp}_lrmax{hparams["lr_max"]:.0e}_bs{hparams["batch_size"]}_steps{hparams["steps"]}"
    
    # PART 2: CREATE A MODEL
    model = transformer_lm(
            d_model = hparams["d_model"], 
            num_heads = hparams["num_heads"], 
            d_ff = hparams["d_ff"], 
            context_length = hparams["context_length"], 
            rope_theta = hparams["rope_theta"], 
            vocab_size = hparams["vocab_size"], 
            num_layers = hparams["num_layers"],
    )
    optimizer = AdamW(
        params = model.parameters, 
        lr = hparams["lr_min"], 
        betas = hparams["betas"], 
        weight_decay= hparams["weight_decay"], 
        eps = hparams["adamw_eps"]
    )

    # PART 3: Load data
    dataset = np.load(hparams["train_data_path"], mmap_mode='r')    
    tokenizer = Tokenizer.from_files(vocab_filepath = hparams["vocab_path"], merges_filepath = hparams["merges_path"], special_tokens=hparams["special_tokens"]) # for sampling
    
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
      device = "mps"
    else:
      device = "cpu"

    # Training loop
    
    for t in hparams["steps"]:
        
        # Get data
        inputs, targets = data_loader(                               # tensors: (batch_size, context_length)
            dataset = dataset, 
            batch_size=hparams["batch_size"], 
            context_length=hparams["context_length"], 
            device = device
        )       

        # Update learning rate
        

        # Forward pass
        logits = model(x = inputs)                                  # output: Float[Tensor, "batch_size ... seq_len vocab_size"]

        loss = cross_entropy(logits = logits, targets = targets)    # int
        loss.backward()
        
