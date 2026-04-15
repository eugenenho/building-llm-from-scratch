import regex as re
import os
from typing import BinaryIO
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# PAT = r"""<\|endoftext\|>|'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
print(re.findall(PAT, "some text that I'll pre-tokenize"))


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
    
    # Initialize vocab
    vocab_effective_size = 256
    vocab = {i: bytes([i]) for i in range(vocab_effective_size)}
    
    for i, token in enumerate(special_tokens):
        vocab[vocab_effective_size] = token.encode("utf-8")
        vocab_effective_size += 1
    
    for k, v in vocab.items():
        print(f"{k}: {v},  {v.decode("utf-8", errors="ignore")}")
    
    # Open file, and get chunks back
    chunks = load_chunks(input_path)
    
    # Process each chunk    
    for chunk in chunks:
        
        # Run pre-tokenization on your chunk and store the counts for each pre-token   
        pattern = "|".join(re.escape(token) for token in special_tokens)
        docs = re.split(pattern, chunk)

        # Pre-tokenize each doc
        for doc in docs:
           
            matches = [match.group() for match in re.finditer(PAT, doc)]
            matches = ["low", "low", "low", "low", "low", "lower", "lower", "widest","widest", "widest", "newest","newest","newest","newest","newest","newest",]
            # Merge: count <--- pick up here
            
            # count
            counts = Counter(matches)
            print(counts)

            counts2 = {tuple(bytes([b]) for b in item.encode("utf-8")): count for item, count in counts.items()}
            print(counts2)

            # get pairwise counts
            frequency={}
            print("-----------")
            for item, count in counts2.items():
                if len(item) < 2: 
                    continue
                temp_count = {pair:count for pair in zip(item[:-1], item[1:])}
                print(temp_count)

                # check if a pair already exists
                for pair, count in temp_count.items():
                    if pair in frequency:
                        frequency[pair] += count
                    else:
                        frequency[pair] = count
                
                
                print("------")
            print("-----------")    
            print(sorted(frequency.items(), key=lambda x: x[1], reverse=True))
                
            ### later move to global location (across all docs)
            
            # find the top pair. highest frequency, and then lexicographically greater
            top_pair = max(frequency.items(), key=lambda x: (x[1], x[0]))
            print(f"top pair: {top_pair}")
            print(f"top pair: {top_pair[0]}")

            # Merge
            
            # add to vocab
            new_vocab = b''.join(top_pair[0])
            vocab[vocab_effective_size] = new_vocab
            print(f"latest vocab: {vocab[vocab_effective_size]}")
            vocab_effective_size += 1
            
            # replace
            for item, count in counts2.items():
                if len(item) < 2: 
                    continue
                
                i = 0
                for pair in zip(item[:-1], item[1:]):
                    
                    print(f"pair: {pair}, top pair: {top_pair[0]}")
                    if (pair == top_pair[0]):
                        print(f"new item udpate before: {item}")
                        item = item[:i] + (new_vocab,) + item[i+2:]
                        print(f"new item udpate after: {item}")
                    i += 1
            print(f"counts2 again: {counts2}")
                        


            
            
            
            
            
            
            

            print(matches)
            break

        break
            





    
# Main run
if __name__ == "__main__":
    
    file_path = "data/TinyStoriesV2-GPT4-valid.txt"
    train_bpe_function(file_path, vocab_size= 300, special_tokens=["<|endoftext|>",])

    
 