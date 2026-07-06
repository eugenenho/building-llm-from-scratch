from cs336_basics.tokenizer import Tokenizer

if __name__ == "__main__":
    vocab = {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
    merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
    test_str = "the cat ate"
    tokenizer = Tokenizer(vocab = vocab, merges = merges)
    print(tokenizer.encode(test_str))
    