import regex as re
import os
import time
from typing import BinaryIO
from collections import Counter
from collections import defaultdict  
from multiprocessing import Pool
from functools import partial

# chunking
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


## Usage
def load_chunks(file_path: str):
    
    with open(file_path, "rb") as f:
        num_processes = os.cpu_count()
        
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)
        return chunks
            
# Pre-tokenization

def pre_tokenize(chunk: str, doc_split_pattern: str, pretok_pattern: str):
    docs = re.split(doc_split_pattern, chunk) # list of strings
    pretok_chunk = [match.group() for doc in docs for match in re.finditer(pretok_pattern, doc)] # list of strings
    return Counter(pretok_chunk)

    

# Function
def train_bpe_function(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
):
    
    # Initilalization / hyperparams
    vocab_effective_size = 256
    vocab = {i: bytes([i]) for i in range(vocab_effective_size)}
    merges = []
    min_frequency = 2
    PRETOK_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    doc_split_pattern = "|".join(re.escape(token) for token in special_tokens)

    for i, token in enumerate(special_tokens):
        vocab[vocab_effective_size] = token.encode("utf-8")
        vocab_effective_size += 1
    
    
    # Open file, and get chunks back
    chunks = load_chunks(input_path) # chunks: list of strings
    
    # # Pre-tokenize each doc separately, then add back
    # pretok_chunk = []
    # for chunk in chunks: 
    #     docs = re.split(doc_split_pattern, chunk) # list of strings
    #     pretok_chunk.extend(pre_tokenize(docs = docs, pattern = PRETOK_PATTERN))

   
    # # Count occurence of pre-tokenized tokens
    # counts_naive = Counter(pretok_chunk)
    # # print(counts_naive)
    
    # counts = {tuple(bytes([b]) for b in item.encode("utf-8")): count for item, count in counts_naive.items()} # dict[tuple(bytes,...), int]
    # # print(counts)

    # parallel version
    t1= time.time()
    with Pool(processes = os.cpu_count()) as pool:
        naive_counters = pool.map(partial(pre_tokenize, doc_split_pattern = doc_split_pattern, pretok_pattern = PRETOK_PATTERN), chunks) #list of Counters
    naive_counters_sum = sum(naive_counters, Counter())
    counts = {tuple(bytes([b]) for b in item.encode("utf-8")): count for item, count in naive_counters_sum.items()} # dict[tuple(bytes,...), int]
    
    # get pair_freq counts
    pair_freq=defaultdict(int)
    for word, count in counts.items():
        if len(word) < 2: 
            continue
        for pair in zip(word[:-1], word[1:]):
            pair_freq[pair] += count
    t2 = time.time()
    
    # Run a while loop
    while (vocab_effective_size < vocab_size):
        
        # find max pair 
        if not pair_freq: 
            print("while exit reason: no more pair_freq")
            break                 # Exit condition: no more pairs available
        top_pair = max(pair_freq.items(), key=lambda x: (x[1], x[0]))
        if top_pair[1] < min_frequency: 
            print("while exit reason: min frequency condition")
            break   # Exit condition: min_frequency condition
        
        # debugging only
        # max_count = max(pair_freq.values())   
        # top_pairs = [(pair, count) for pair, count in pair_freq.items() if count == max_count]  
        # print(f"merge #: {len(merges)},    top pair: {top_pair},    other top pairs: {top_pairs}")
        
        # add to vocab
        new_vocab = b''.join(top_pair[0])
        merges.append(top_pair[0])
        vocab[vocab_effective_size] = new_vocab
        vocab_effective_size += 1
        pair_freq.pop(top_pair[0]) # remove from pair_freq, as this top_pair[0], a tuple, will no longer exist

        # merge       
        for word, count in list(counts.items()):
            i = 0
            indices = []
            len_word = len(word)
            pairs = list(zip(word[:-1], word[1:])) # list of tuples
            if len_word < 2: 
                continue
            for pair in pairs:
                if pair == top_pair[0]:
                    indices.append(i)
                i += 1
            
            # update the word
            len_indices = len(indices)           
            if len_indices > 0:
                q = 0
                segment = word[:indices[q]] + (new_vocab,)
            
                while q + 1 < len_indices:
                    segment = segment + word[indices[q] + 2: indices[q+1]] + (new_vocab,)
                    q += 1
                segment = segment + word[indices[q]+2:len_word+1]
                # if len(indices) > 1: print(f"word: {word}, updated word: {segment}, indices: {indices}")
                counts[segment] = counts.pop(word) # update counts dict

                # update pair_freq: decrement non-existent pairs
                decremented_indices =[]
                for q in range(len_indices):
                    
                    index_one_ahead = indices[q]-1
                    index_one_behind = indices[q]+1

                    if (index_one_ahead >= 0) and (index_one_ahead not in decremented_indices):
                        pair_freq[pairs[index_one_ahead]] -= count
                        decremented_indices.append(index_one_ahead)
                    
                    if (index_one_behind < len_word - 1) and (index_one_behind not in decremented_indices):
                        pair_freq[pairs[index_one_behind]] -= count
                        decremented_indices.append(index_one_behind)
                
                # update pair_Freq: increment newly formed pairs
                for segment_pair in zip(segment[:-1], segment[1:]):
                    if new_vocab in segment_pair:
                        pair_freq[segment_pair] += count
                
                


    t3 = time.time()

    print(f"vocab_size: {vocab_effective_size}")
    print(f"pre-tok time: {t2 - t1:.3f}s")
    print(f"merge time: {t3 - t2:.3f}s")
    
    with open("output.txt", "w") as f:                                                                       
        f.write(str(vocab) + "\n")                                                                           
        f.write(str(merges) + "\n")  
    return vocab, merges
            
            

            # . Looking at your merge loop — for each merge, you iterate over all entries in counts (line 161). As the vocabulary grows, this gets expensive. Think about: do you need to scan every word on every merge, or could you track which words contain a given pair more efficiently?





    
# Main run
if __name__ == "__main__":
    
    file_path = "data/TinyStoriesV2-GPT4-train.txt"
    train_bpe_function(file_path, vocab_size= 10000, special_tokens=["<|endoftext|>",])

    
 