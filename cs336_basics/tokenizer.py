from typing import Iterable, Iterator
import regex as re

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens = None):
        self.vocab = vocab
        self.merges = merges # list [tuple [bytes, bytes]]
        self.special_tokens = special_tokens # list[str]
        self.reverse_vocab = {v:k for k, v in self.vocab.items()}
        self.merges_index = {value: index for index, value in enumerate(merges)}
        
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] = None):
        
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

        # print("testing from_files: \n vocab: {vocab} \n\n\n merges: {merges}")       

        return cls(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        
        def encode_doc(doc: str) -> list[int]:               
            # Pretokenize the string first: string -> list[pretokenized str, ...]
            PRETOK_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            pretok_text = [match.group() for match in re.finditer(PRETOK_PATTERN, doc)] # list of strings
            vocab_index_list =[]
            # print(f"encoding start. pretoke_text: {pretok_text}")

            # Loop over each pre-tokenized string, and merge, until we get to vocab index
            for pretoken in pretok_text:
                # print(f"pretoken: {pretoken}")
                pretoken_byte = tuple(bytes([b]) for b in pretoken.encode("utf-8")) # tuple of bytes, e.g. [b'c', b'h', b'a', b'r', b'a', b'c', b't', b'e', b'r']
               
                
                # For this pretoken_byte: while loop until
                #       1. the entire tuple becomes one byte and can't be combined any further (while condition), or
                #       2. no pair of bytes that appears in merges
                while len(pretoken_byte) > 1:

                    # Initialization for each loop
                    min_value = len(self.merges)    # greater than absolute largest merge index by 1
                    min_index = []
                          

                    # Find the highest priority / earliest merge           
                    for pretoken_index in range(len(pretoken_byte) - 1):        # Iterate through the whole pretoken
                        span = (pretoken_byte[pretoken_index], pretoken_byte[pretoken_index + 1]) # tuple of bytes               
                        rel_merge = self.merges_index.get(span, float('inf'))   # this span's relevant merge index. infinity if it doesn't show up in merges
                        if rel_merge < min_value:           
                            min_value = rel_merge
                            min_index = [pretoken_index]
                        elif rel_merge == min_value:                            # same merge index as existing min index, which means, it's the same byte pair
                            min_index.append(pretoken_index)
                    
                    # Exit condition #2: 
                    # This means, there wasn't any pair/span in the pretoken_byte that is on the merges list. exit while
                    if not min_index:   
                        break           
                                    
                    
                    # Updating the pretoken_byte
                    pretoken_index = 0
                    pretoken_byte_new = []                                          # list, to save compute
                    while pretoken_index < len(pretoken_byte) - 1:
                        if pretoken_index not in min_index:                         # regular byte, just add and increment index by 1
                            pretoken_byte_new.append(pretoken_byte[pretoken_index])
                            pretoken_index += 1
                        else:                                                       # first byte of the pair that appears in min_index merges. increment index by 2
                            pretoken_byte_new.append(b''.join((pretoken_byte[pretoken_index], pretoken_byte[pretoken_index + 1])))
                            pretoken_index += 2             ## SKIP AHEAD two positions
                    
                    if pretoken_index == len(pretoken_byte) - 1:        # in case we exited while with the index at the very last position, dangling byte
                            pretoken_byte_new.append(pretoken_byte[pretoken_index])
                    
                    pretoken_byte = tuple(pretoken_byte_new)
                
                # Once you exit the while loop, you have a tuple of bytes that can no longer be merged.
                # Add this to the final vocab index list. This concludes the loop for this pretoken. Move on to the next pretoken
                vocab_index_list.extend(self.reverse_vocab[byte] for byte in pretoken_byte)
            
            # Once you exit the for loop, you have gone through all the pretokens
            return vocab_index_list
                           


            #### OLD CODE. NAIVE IMPLEMENTAION

            #     # Loop until no more merges are possible, which means:
            #     #   1 - reached the end of the self.merges list
            #     #   2 - no more tuples of bytes to merge
                
            #     merges_index = 0
            #     merges_len = len(self.merges)

            #     while merges_index < merges_len and len(pretoken_byte) > 1:
                    
            #         # for each item in self.merges, sweep across pretoken_byte to see if there's a match.
            #         # If there's a match, add the 
            #         pretoken_index = 0
            #         pretoken_byte_new = ()
            #         # print(f"    pretoken_byte: {pretoken_byte}, merges_index: {merges_index}, merges: {self.merges[merges_index]}")

            #         while pretoken_index < len(pretoken_byte) - 1:
            #             span = (pretoken_byte[pretoken_index], pretoken_byte[pretoken_index + 1]) # tuple of bytes
                        

            #             if span == self.merges[merges_index]:
            #                 pretoken_byte_new += (b''.join(span),) # add the merged span
            #                 # print(f"        pretoken_index: {pretoken_index}, MERGED. span: {span}, pretokenbytenew: {pretoken_byte_new}")
            #                 pretoken_index += 2                 # skip one index
                            
            #             else:
            #                 pretoken_byte_new += (pretoken_byte[pretoken_index],)
            #                 # print(f"        pretoken_index: {pretoken_index}, NO MERGE. span: {span}, pretokenbytenew: {pretoken_byte_new}")
            #                 pretoken_index += 1                 # regular incrementing
                    
            #         # print(f"        pretoekn_index: {pretoken_index}, pretoken_len: {len(pretoken_byte)}")
            #         if pretoken_index == len(pretoken_byte) - 1:        # in case we exited while with the index at the very last position, dangling byte
            #             pretoken_byte_new += (pretoken_byte[pretoken_index],)
            #         # print(f"            pretoken_byte_new dangling situation: {pretoken_byte_new}")

            #         pretoken_byte = pretoken_byte_new
            #         merges_index += 1
                        
            #     # Once you exit the while loop, you have a tuple of bytes that can no longer be merged.
            #     # Add this to the final vocab index list. This concludes the loop for this pretoken. Move on to the next pretoken
            #     vocab_index_list.extend(self.reverse_vocab[byte] for byte in pretoken_byte)
            # # print(f"vocab index list: {vocab_index_list}")
            # return vocab_index_list

        # Special token processing
        vocab_index_list = []
        if self.special_tokens:
            ordered_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            doc_split_pattern = "|".join(re.escape(token) for token in ordered_special_tokens)
            doc_split_pattern = "(" + doc_split_pattern + ")"
            docs = re.split(doc_split_pattern, text) # list of text strings, separated by special tokens but containing them as items
            for doc in docs:
                if not doc:
                    continue
                if doc in self.special_tokens:
                    vocab_index_list.extend([self.reverse_vocab[doc.encode("utf-8")]])
                else:
                    vocab_index_list.extend(encode_doc(doc))
            
        else:
            vocab_index_list.extend(encode_doc(text))
        
        return vocab_index_list
    
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)


    def decode(self, ids: list[int]) -> str:
    
        byte_list = [self.vocab[id] for id in ids]
        byte_concat = b''.join(byte_list)
        output_string = byte_concat.decode("utf-8", errors='replace')
        return output_string

