import os


def create_vocab(file_path: str):
    """
    Create the vocabulary for the text transformer, as described by Kamienny et al. (2022) in End-to-end symbolic
    regression with transformers.
    :param file_path: Path and filename of the vocabulary
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        file.write(f"+\n")
        file.write(f"-\n")
        for mantissa in range(10000):
            file.write(f"{mantissa}\n")
        for exponent in range(-100, 101):
            file.write(f"E{exponent}\n")


if __name__ == "__main__":
    create_vocab("../../data/text_transformer_vocab.txt")
