from typing import Iterable, Iterator

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: str = None):
        
        # assume: JSON
        with open(vocab_filepath, "r") as f1:
            vocab_dump = f1.read() # dict[str, str] but needs to be dict[int, bytes]
        vocab = {int(k): v.encode("utf-8") for k, v in vocab_dump.items()}
        
        # assume: txt
        merges = [] # needs to be list[tuple[bytes, bytes]]
        with open(merges_filepath, "r") as f2:
            for line in f2: 
                list_of_words = line.rstrip().split() # [str, str]
                merges.append((list_of_words[0].encode("utf-8"), list_of_words[1].encode("utf-8")))

        print("testing from_files: \n vocab: {vocab} \n\n\n merges: {merges}")       

        return cls(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        
        raise NotImplementedError
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        raise NotImplementedError


    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError
    



