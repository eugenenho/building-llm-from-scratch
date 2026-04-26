import torch
from cs336_basics.model import transformer_lm
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

if __name__ == "__main__":
    
    # PART 1: GET HYPERPARAMETERS
    """ 
    Full set of hyperparameters needed for training:(example)
        model:
            context_length: 256
            d_model: 512
            vocab_size: null
            num_layers: 4
            num_heads: 16 
            rope_theta: 10000
            d_ff: 1344

        training:
            batch_size: 32
            steps: 5000
            warmup_iters: 250
            cosine_cycle_iters: 5000
            lr_max: 1e-3
            lr_min: 1e-4
            weight_decay: 0.01
            betas:
                - 0.9
                - 0.95
            adamw_eps: 1e-8
            max_l2_norm: 1.0

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
    nested_hparams = yaml.safe_load(open(config_path)) # dict
    hparams = {}
    hparams = {k:v for _, group in nested_hparams.items() for k, v in group.items()}
    
    # Step 2: Check for overrides from CLI
    list_args = ["run_name", "lr_max", "lr_min", "steps", "batch_size", "context_length"]
    parser.add_argument("--run-name", default=argparse.SUPPRESS, type=str, help="name of the run")
    parser.add_argument("--lr-max", default=argparse.SUPPRESS, type=float, help="max learning rate for the lr scheduler")
    parser.add_argument("--lr-min", default=argparse.SUPPRESS, type=float, help="min learning rate for the lr scheduler")
    parser.add_argument("--steps", default=argparse.SUPPRESS, type=int, help="total steps for training")
    parser.add_argument("--batch-size", default=argparse.SUPPRESS, type=int, help="batch size for training")
    parser.add_argument("--context-length", default=argparse.SUPPRESS, type=int, help="max sequence length for the model")
    args = parser.parse_args()
    cli_overrides = vars(args) # dict of only what the user passed on CLI, overriding YAML

    for k, v in cli_overrides.items():
        if k == "config":
            continue
        if k not in list_args:
            raise ValueError(f"non-acceptable flag: {k}")
        if k in hparams and hparams[k] != v:
            print(f"CLI override: {k} = {v} (from YAML: {hparams[k]})")
            hparams[k] = v  
    
    # Step 3: Check if any of the fields are empty
    missing_okay = ["run_name", "d_ff", "vocab_size"]
    for k, v in hparams.items():
        if v is None and k not in missing_okay: 
            raise ValueError(f"Critical hyperparameter missing: {k}: {v}")

    if hparams["d_ff"] is None:
        hparams["d_ff"] = round(8/3 * hparams["d_model"] / 64) * 64 
    
    if hparams["run_name"] is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        hparams["run_name"] = f"{timestamp}_lrmax{hparams['lr_max']:.0e}_bs{hparams['batch_size']}_steps{hparams['steps']}"
     
    # Step 4: Compute and add hparams not in yaml originally
    tokenizer = Tokenizer.from_files(                   # for sampling, for vocab_size
        vocab_filepath = hparams["vocab_path"], 
        merges_filepath = hparams["merges_path"], 
        special_tokens=hparams["special_tokens"]
    ) 
    
    vocab_size_from_tokenizer = len(tokenizer.vocab)
    if hparams["vocab_size"] is not None and hparams["vocab_size"] != vocab_size_from_tokenizer:
        print(f"vocab_size entered in YAML is incorrect. YAML value: {hparams['vocab_size']}. True value based on the loaded tokenizer: {vocab_size_from_tokenizer}. Updating to the correct value")
    hparams["vocab_size"] = vocab_size_from_tokenizer

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
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
    model.to(device)
    optimizer = AdamW(
        params = model.parameters(), 
        lr = 0, 
        betas = hparams["betas"], 
        weight_decay= hparams["weight_decay"], 
        eps = hparams["adamw_eps"]
    )

    # PART 3: ALL OTHER SET UP
    
    ## DEBUGGING ##
    # for k, v in hparams.items():
    #     print(f"{k}: {v}  - dtype: {type(v)}")

    # Checkpointing set up
    run_dir = Path(hparams["output_dir"]) / hparams["run_name"]
    run_dir.mkdir(parents = True, exist_ok = True)
    
    # Dumping config in the run_dir
    nested_hparams_for_dump = {
        group_name: {k: hparams[k] for k in group.keys() if k in hparams}
        for group_name, group in nested_hparams.items()
    }
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(nested_hparams_for_dump, f, sort_keys=False)
    
    # Logging set up
    wandb.init(
      project="building-llm-from-scratch-1",        # groups runs in the dashboard
      name=hparams["run_name"],                     # this run's name
      config=hparams,                               # auto-logged as the run's config
    )
    
    # Data loading
    train_dataset = np.load(hparams["train_data_path"], mmap_mode='r')
    val_dataset = np.load(hparams["val_data_path"], mmap_mode='r')


    # PART 4: TRAINING LOOP
    for t in range(hparams["steps"]):
        
        # Get data
        inputs, targets = data_loader(                               # tensors: (batch_size, context_length)
            dataset = train_dataset, 
            batch_size=hparams["batch_size"], 
            context_length=hparams["context_length"], 
            device = device
        )       

        # Update learning rate
        lr = learning_rate_scheduler(
            t=t, 
            max_learning_rate=hparams["lr_max"],
            min_learning_rate=hparams["lr_min"],
            warmup_iters=hparams["warmup_iters"],
            cosine_cycle_iters=hparams["cosine_cycle_iters"],
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Forward, backward, step
        optimizer.zero_grad()
        logits = model(x = inputs)                                  # output: Float[Tensor, "batch_size ... seq_len vocab_size"]
        loss = cross_entropy(logits = logits, targets = targets)    # torch
        loss.backward()
        gradient_clipping(model.parameters(), max_l2_norm=hparams["max_l2_norm"])
        optimizer.step()

        # Progress logging
        wandb.log({"training_loss": loss.item(), "lr": lr}, step=t)

        # Check for NaN or inf/-inf => if so, diverged. log and break
        if not math.isfinite(loss.item()): 
            print(f"DIVERGED at step {t}, loss={loss.item()}")
            wandb.run.summary["diverged"] = True
            wandb.run.summary["divergence_step"] = t
            wandb.run.summary["lr_at_divergence"] = lr
            break

        # Checkpointing
        if (t + 1) % hparams["save_every"] == 0:
            ckpt_path = run_dir / f"step_{t}.pt"
            save_checkpoint(model = model, optimizer = optimizer, iteration = t, out = ckpt_path)

        # Validation
        if (t + 1) % hparams["eval_every"] == 0:
            
            model.eval()                    # turns off dropout, etc
            with torch.no_grad():           # no autograd stuff, saves memory and time

                val_losses = []                
                for _ in range(hparams["val_iters"]):
                    # Get data
                    val_inputs, val_targets = data_loader(                               # tensors: (batch_size, context_length)
                        dataset = val_dataset, 
                        batch_size=hparams["batch_size"], 
                        context_length=hparams["context_length"], 
                        device = device
                    ) 
                    val_logits = model(x = val_inputs)
                    val_losses.append(cross_entropy(logits = val_logits, targets = val_targets).item())
                val_loss = sum(val_losses) / len(val_losses)
                val_ppl = math.exp(val_loss)
            model.train()
            wandb.log({"val_loss": val_loss, "val_ppl": val_ppl}, step=t)
        
            print(f"Step: {t}   Training loss: {loss.item()}    Validation loss: {val_loss}    Vallidation perplexity: {val_ppl}     lr: {lr}")
            

    final_path = run_dir / f"step_{t}_final.pt"
    save_checkpoint(model = model, optimizer = optimizer, iteration = t, out = final_path)
    wandb.run.summary["diverged"] = False
    wandb.finish()


