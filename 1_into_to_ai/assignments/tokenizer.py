class Tokenizer:
    def encode(self, text):
        encoded_text = []
        splitted_text = list(text)

        for word in splitted_text:
            encoded_word = []
            for char in word:
                encoded_word.append(ord(char))
            encoded_text.extend(encoded_word)
        return encoded_text

    def decode(self, encoded_text):
        decoded_text = []

        for word in encoded_text:
            decoded_text.append(chr(word))

        return "".join(decoded_text)


text = "hi there"
tokenizer = Tokenizer()
print(tokenizer.encode(text))
print(tokenizer.decode([104, 105, 32, 116, 104, 101, 114, 101]))
