from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_bpe import load_chunks, find_chunk_boundaries
import regex as re
import time

if __name__ == "__main__":


    # TinyStories
    
    # Load TinyStories vocab and merges
        
    ts_vocab_path = "tinystories_train_output/vocab.json"
    ts_merges_path = "tinystories_train_output/merges.txt"
    ts_special_tokens = ["<|endoftext|>",]
    ts_iterable_path = "data/TinyStoriesV2-GPT4-train.txt"
    special_tokens=["<|endoftext|>"]
    doc_split_pattern = "|".join(re.escape(token) for token in special_tokens)
    
    tokenizer = Tokenizer.from_files(vocab_filepath = ts_vocab_path, merges_filepath = ts_merges_path, special_tokens = ts_special_tokens)
    
    chunks = load_chunks(ts_iterable_path, 10) # chunks: list of strings
    docs = re.split(doc_split_pattern, chunks[0]) # list of strings

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

 
 
    # with open(ts_iterable_path, "r") as f:
    #     for token_id in tokenizer.encode_iterable(iterable = f):
    #         print(token_id)




    # TEST CASE #1
    # vocab = {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
    # merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
    # test_str = "the cat ate"
    # tokenizer = Tokenizer(vocab = vocab, merges = merges)
    # print(tokenizer.encode(test_str))

    