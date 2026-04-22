from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_bpe import load_chunks, find_chunk_boundaries
import regex as re
import time
import numpy as np


def tiny_story_experiment():
    # TinyStories  
    ts_vocab_path = "outputs_tinystories/vocab.json"
    ts_merges_path = "outputs_tinystories/merges.txt"
    ts_special_tokens = ["<|endoftext|>",]
    ts_data_path = "data/TinyStoriesV2-GPT4-train.txt"
    doc_split_pattern = "|".join(re.escape(token) for token in special_tokens)
    
    tokenizer = Tokenizer.from_files(vocab_filepath = ts_vocab_path, merges_filepath = ts_merges_path, special_tokens = ts_special_tokens)
    
    chunks_info = load_chunks(ts_data_path, 10) # chunks: list of strings

    (file_path, start, end) = chunks_info[0]
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    docs = re.split(doc_split_pattern, chunks_info[0]) # list of strings

    text_bytes = 0
    len_tok_ids = 0
    
    t1 = time.time()
    for i in range(10):
        original = docs[i]                      # string
        encoded = tokenizer.encode(docs[i])     # list of ints
        decoded = tokenizer.decode(encoded)     # string
        assert original == decoded
        # print(f"Sample {i+1}: \nText: {original}\nTokenIDs: {encoded}\nDecoded Again: {decoded}\n\n ----------------\n")

        text_bytes += len(original.encode("utf-8")) # text encoded into bytes, then size measured by counting bytes
        len_tok_ids += len(encoded)
    
    t2 = time.time()
    encoding_time = t2 - t1
    print(f"encoding time: {encoding_time:.3f}s")
    throughput = text_bytes / encoding_time
    print(f"throughput: {throughput} bytes/sec")

    estimated_time_pile = float(825 * 1024**3) / throughput # bytes/sec
    print(f"estimated time to encode the Pile dataset (825GB of text): {estimated_time_pile} seconds, or {estimated_time_pile / 3600} hours")
    
    comp_ratio = text_bytes / len_tok_ids
    print(f"Compression ratio of Tiny Stories BPE tokenizer: {comp_ratio}")

def owt_experiment():
    # OWT
    vocab_path = "outputs_owt/vocab.json"
    merges_path = "outputs_owt/merges.txt"
    special_tokens = ["<|endoftext|>",]
    data_path = "data/owt_train.txt"
    doc_split_pattern = "|".join(re.escape(token) for token in special_tokens)
    
    tokenizer = Tokenizer.from_files(vocab_filepath = vocab_path, merges_filepath = merges_path, special_tokens = special_tokens)
    
    chunks_info = load_chunks(data_path, 10) # chunks: list of strings

    (file_path, start, end) = chunks_info[0]
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
   
    docs = re.split(doc_split_pattern, chunk) # list of strings

    text_bytes = 0
    len_tok_ids = 0
    
    t1 = time.time()
    for i in range(10):
        original = docs[i]                      # string
        encoded = tokenizer.encode(docs[i])     # list of ints
        decoded = tokenizer.decode(encoded)     # string
        assert original == decoded
        print(f"Sample {i+1}: \nText: {original}\nTokenIDs: {encoded}\nDecoded Again: {decoded}\n\n ----------------\n")

        text_bytes += len(original.encode("utf-8")) # text encoded into bytes, then size measured by counting bytes
        len_tok_ids += len(encoded)
    
    t2 = time.time()
    encoding_time = t2 - t1
    print(f"encoding time: {encoding_time:.3f}s")
    throughput = text_bytes / encoding_time
    print(f"throughput: {throughput} bytes/sec")

    estimated_time_pile = float(825 * 1024**3) / throughput # bytes/sec
    print(f"estimated time to encode the Pile dataset (825GB of text): {estimated_time_pile} seconds, or {estimated_time_pile / 3600} hours")
    
    comp_ratio = text_bytes / len_tok_ids
    print(f"Compression ratio of OWT BPE tokenizer: {comp_ratio}")


def mixed_experiment():
    """
        Load the Tiny Stories tokenizer, then
        Encode the samples from OWT
    """
    
    # Tokenizer: from Tiny Stories
    vocab_path = "outputs_tinystories/vocab.json"
    merges_path = "outputs_tinystories/merges.txt"
        
    # Samples: from OWT
    data_path = "data/owt_train.txt"
    special_tokens=["<|endoftext|>",]
    doc_split_pattern = "|".join(re.escape(token) for token in special_tokens)
    
    tokenizer = Tokenizer.from_files(vocab_filepath = vocab_path, merges_filepath = merges_path, special_tokens = special_tokens)
    
    chunks_info = load_chunks(data_path, 10) # chunks: list of strings

    (file_path, start, end) = chunks_info[0]
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
   
    docs = re.split(doc_split_pattern, chunk) # list of strings

    text_bytes = 0
    len_tok_ids = 0
    
    t1 = time.time()
    for i in range(10):
        original = docs[i]                      # string
        encoded = tokenizer.encode(docs[i])     # list of ints
        decoded = tokenizer.decode(encoded)     # string
        assert original == decoded
        print(f"Sample {i+1}: \nText: {original}\nTokenIDs: {encoded}\nDecoded Again: {decoded}\n\n ----------------\n")

        text_bytes += len(original.encode("utf-8")) # text encoded into bytes, then size measured by counting bytes
        len_tok_ids += len(encoded)
    
    t2 = time.time()
    encoding_time = t2 - t1
    print(f"encoding time: {encoding_time:.3f}s")
    throughput = text_bytes / encoding_time
    print(f"throughput: {throughput} bytes/sec")

    estimated_time_pile = float(825 * 1024**3) / throughput # bytes/sec
    print(f"estimated time to encode the Pile dataset (825GB of text): {estimated_time_pile} seconds, or {estimated_time_pile / 3600} hours")
    
    comp_ratio = text_bytes / len_tok_ids
    print(f"Compression ratio of BPE tokenizer: {comp_ratio}")


def encoding_tinystories():
    # TinyStories  
    vocab_path = "tinystories_train_output/vocab.json"
    merges_path = "tinystories_train_output/merges.txt"
    special_tokens = ["<|endoftext|>",]
    data_path_train = "data/TinyStoriesV2-GPT4-train.txt"
    data_path_valid = "data/TinyStoriesV2-GPT4-valid.txt"
    output_path_train = "data/TinyStoriesV2-GPT4-train-encoded.npy"
    output_path_valid = "data/TinyStoriesV2-GPT4-valid-encoded.npy"
    doc_split_pattern = "|".join(re.escape(token) for token in special_tokens)
    
    tokenizer = Tokenizer.from_files(vocab_filepath = vocab_path, merges_filepath = merges_path, special_tokens = special_tokens)
    

    # Validation dataset 
    t1 = time.time()
    with open(data_path_valid, "r") as fv:
        valid_tok_ids = list(tokenizer.encode_iterable(fv))
    
    valid_tok_ids_np = np.array(valid_tok_ids, dtype = np.uint16)
    np.save(output_path_valid, valid_tok_ids_np)
    t2 = time.time()
    print(f"Validation set encoded and saved. Time taken: {t2 - t1:.3f}s")

    # Sanity check : passed
    # print("\n\n\n")
    # arr = np.load("data/TinyStoriesV2-GPT4-valid-encoded.npy")
    # print(arr.shape, arr.dtype, arr[:20])
    # print(f"decoded content: {tokenizer.decode(arr.tolist())}")

    # Training dataset 
    t3 = time.time()
    with open(data_path_train, "r") as ft:
        train_tok_ids = list(tokenizer.encode_iterable(ft))
    
    train_tok_ids_np = np.array(train_tok_ids, dtype = np.uint16)
    np.save(output_path_train, train_tok_ids_np)
    t4 = time.time()
    print(f"Training set encoded and saved. Time taken: {t4 - t3:.3f}s")


def encoding_owt():
    # OWT
    vocab_path = "outputs_owt/vocab.json"
    merges_path = "outputs_owt/merges.txt"
    special_tokens = ["<|endoftext|>",]
    data_path_train = "data/owt_train.txt"
    data_path_valid = "data/owt_valid.txt"
    output_path_train = "data/owt-train-encoded.npy"
    output_path_valid = "data/owt-valid-encoded.npy"
    doc_split_pattern = "|".join(re.escape(token) for token in special_tokens)
    
    tokenizer = Tokenizer.from_files(vocab_filepath = vocab_path, merges_filepath = merges_path, special_tokens = special_tokens)
    
    # Validation dataset 
    t1 = time.time()
    with open(data_path_valid, "r") as fv:
        valid_tok_ids = list(tokenizer.encode_iterable(fv))
    
    valid_tok_ids_np = np.array(valid_tok_ids, dtype = np.uint16)
    np.save(output_path_valid, valid_tok_ids_np)
    t2 = time.time()
    print(f"Validation set encoded and saved. Time taken: {t2 - t1:.3f}s")

    # Sanity check : passed
    # print("\n\n\n")
    # arr = np.load("data/owt-valid-encoded.npy")
    # print(arr.shape, arr.dtype, arr[:20])
    # print(f"decoded content: {tokenizer.decode(arr[:20].tolist())}")

    # Training dataset 
    t3 = time.time()
    with open(data_path_train, "r") as ft:
        train_tok_ids = list(tokenizer.encode_iterable(ft))
    
    train_tok_ids_np = np.array(train_tok_ids, dtype = np.uint16)
    np.save(output_path_train, train_tok_ids_np)
    t4 = time.time()
    print(f"Training set encoded and saved. Time taken: {t4 - t3:.3f}s")


if __name__ == "__main__":
#    tiny_story_experiment()
    # owt_experiment()
    # mixed_experiment()
    # encoding_tinystories()
    encoding_owt()




 










    # TEST CASE #1
    # vocab = {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
    # merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
    # test_str = "the cat ate"
    # tokenizer = Tokenizer(vocab = vocab, merges = merges)
    # print(tokenizer.encode(test_str))

    