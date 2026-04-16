import regex as re
import os
from typing import BinaryIO
from collections import Counter
from collections import defaultdict  

# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# print(re.findall(PAT, "some text that I'll pre-tokenize"))



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
        num_processes = 4
        
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        print(boundaries)

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)
        return chunks
            

# Function

def train_bpe_function(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
):
    
    # Initialize vocab & merges
    vocab_effective_size = 256
    vocab = {i: bytes([i]) for i in range(vocab_effective_size)}
    merges = []
    min_frequency = 2

    for i, token in enumerate(special_tokens):
        vocab[vocab_effective_size] = token.encode("utf-8")
        vocab_effective_size += 1
    
    for k, v in vocab.items():
        print(f"{k}: {v},  {v.decode("utf-8", errors="ignore")}")
    
    # Open file, and get chunks back
    chunks = load_chunks(input_path)
    
    # Pre-tokenize each doc separately, then add back
    pretok_chunk = []
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pattern = "|".join(re.escape(token) for token in special_tokens)
    
    for chunk in chunks:
        docs = re.split(pattern, chunk)        
        pretok_chunk.extend([match.group() for doc in docs for match in re.finditer(PAT, doc)])
        print(pretok_chunk)
    
    # only for initial debugging
    # pretok_chunk = ["low", "low", "low", "low", "low", "lower", "lower", "widest","widest", "widest", "newest","newest","newest","newest","newest","newest",]
    # print(pretok_chunk)
      
    
    # Count occurence of pre-tokenized tokens
    counts_naive = Counter(pretok_chunk)
    # print(counts_naive)
    
    counts = {tuple(bytes([b]) for b in item.encode("utf-8")): count for item, count in counts_naive.items()} # dict[tuple(bytes,...), int]
    # print(counts)
    
    
    # Run a while loop
    while (vocab_effective_size < vocab_size):
        
        # get pair_freq counts
        # pair_freq = get_pair_freq(counts, pair_freq)
        pair_freq=defaultdict(int)
        for word, count in counts.items():
            if len(word) < 2: 
                continue
            for pair in zip(word[:-1], word[1:]):
                pair_freq[pair] += count
                
        # print("-----------")    
        # print(f"counts: {counts}")
        # print(f"pair_freq in a sorted way: {sorted(pair_freq.items(), key=lambda x: x[1], reverse=True)}")

        # find max pair 
        if not pair_freq: 
            print("while exit reason: no more pair_freq")
            break                 # Exit condition: no more pairs available
        top_pair = max(pair_freq.items(), key=lambda x: (x[1], x[0]))
        if top_pair[1] < min_frequency: 
            print("while exit reason: min frequency condition")
            break   # Exit condition: min_frequency condition
        # print(f"top pair: {top_pair}")
        # print(f"top pair: {top_pair[0]}")

        # add to vocab
        new_vocab = b''.join(top_pair[0])
        merges.append(top_pair[0])
        vocab[vocab_effective_size] = new_vocab
        # print(f"latest vocab: {vocab[vocab_effective_size]}")
        vocab_effective_size += 1

        # merge
        # counts = merge(counts, top_pair)
        
        for word, count in list(counts.items()):
            i = 0
            indices = []
            if len(word) < 2: 
                continue
            for pair in zip(word[:-1], word[1:]):
                if pair == top_pair[0]:
                    indices.append(i)
                    # updated_word = word[:i] + (new_vocab,) + word[i+2:]
                    # counts[updated_word] = counts.pop(word)
                    # print(f"word: {word}, updated word: {updated_word}")
                i += 1
            
            # update the word
            working_word = word
            offset = 0
            if len(indices) > 0:
                for index in indices:
                    updated_word = working_word[:index - offset] + (new_vocab,) + working_word[index+2-offset:]
                    offset += 1
                # print(f"word: {word}, updated word: {updated_word}")
                counts[updated_word] = counts.pop(word)
        # print(counts)
    print(f"vocab_size: {vocab_effective_size}")
    
    with open("output.txt", "w") as f:                                                                       
        f.write(str(vocab) + "\n")                                                                           
        f.write(str(merges) + "\n")  
    return vocab, merges
    #print(f"vocab: {vocab}")
    #print(f"merges: {merges}")
            
            





    
# Main run
if __name__ == "__main__":
    
    file_path = "data/TinyStoriesV2-GPT4-valid.txt"
    train_bpe_function(file_path, vocab_size= 10000, special_tokens=["<|endoftext|>",])

    
 