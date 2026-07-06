


test_string = "hello! こんにちは! 도고 왜 그러시오"
utf8_encoded = test_string.encode("utf-8")
utf16_encoded = test_string.encode("utf-16")
utf32_encoded = test_string.encode("utf-32")

print(f"utf8_encoded: {utf8_encoded}")
print(f"utf16_encoded: {utf16_encoded}")
print(f"utf32_encoded: {utf32_encoded}")

print(f"utf8_encoded type: {type(utf8_encoded)}")
print(f"utf16_encoded type: {type(utf16_encoded)}")
print(f"utf32_encoded type: {type(utf32_encoded)}")


print(f"utf8_encoded list: {list(utf8_encoded)}")
print(f"utf16_encoded list: {list(utf16_encoded)}")
print(f"utf32_encoded list: {list(utf32_encoded)}")


print(f"test string len: {len(test_string)}")
print(f"utf8_encoded len: {len(utf8_encoded)}")
print(f"utf16_encoded len: {len(utf16_encoded)}")
print(f"utf32_encoded len: {len(utf32_encoded)}")

def decode_utf_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

print(decode_utf_bytes_to_str_wrong("hello ".encode("utf-8")))

undecodable_b = int('10100000', 2).to_bytes(2, 'big')
undecodable_b= undecodable_b + undecodable_b
print(undecodable_b)
print(f"undecodable: '{undecodable_b.decode()}'")
