from cs336_basics.tokenizer import Tokenizer
import time
import numpy as np
import os

def encoding_owt():
    # OWT
    vocab_path = "outputs_owt/vocab.json"
    merges_path = "outputs_owt/merges.txt"
    special_tokens = ["<|endoftext|>",]
    data_path_train = "data/owt_train.txt"
    data_path_valid = "data/owt_valid.txt"
    output_path_train = "data/owt-train-encoded.npy"
    output_path_valid = "data/owt-valid-encoded.npy"
    
    # Existence check
    required_inputs = [vocab_path, merges_path, data_path_valid, data_path_train]
    missing = []
    for input_path in required_inputs:
        if not os.path.exists(input_path): 
            missing.append(input_path)
    if missing:
        raise SystemExit(f"Missing input file: {missing}. Run the download steps in the README first")    

    tokenizer = Tokenizer.from_files(vocab_filepath = vocab_path, merges_filepath = merges_path, special_tokens = special_tokens)
    
    # Validation dataset 
    t1 = time.time()
    with open(data_path_valid, "r") as fv:
        valid_tok_ids = list(tokenizer.encode_iterable(fv))
    
    valid_tok_ids_np = np.array(valid_tok_ids, dtype = np.uint16)
    np.save(output_path_valid, valid_tok_ids_np)
    t2 = time.time()
    print(f"Validation set encoded and saved. Time taken: {t2 - t1:.3f}s")
    

    # Training dataset 
    t3 = time.time()
    with open(data_path_train, "r") as ft:
        train_tok_ids = list(tokenizer.encode_iterable(ft))
    
    train_tok_ids_np = np.array(train_tok_ids, dtype = np.uint16)
    np.save(output_path_train, train_tok_ids_np)
    t4 = time.time()
    print(f"Training set encoded and saved. Time taken: {t4 - t3:.3f}s")


if __name__ == "__main__":
    """
        Run this code to encode validation and training data (OWT) to .npy
        Need to have the raw dataset files (from the repo README)
        Note: Encoding the OWT train set (~5 GB) holds the full token stream in memory
              before writing, so it needs a machine with substantial free RAM and takes several minutes.
    """
    
    encoding_owt()
