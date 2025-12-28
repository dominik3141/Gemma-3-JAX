import argparse
import os
import sys
import sentencepiece as spm


class GemmaTokenizer:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Tokenizer model file not found at: {model_path}")

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

    def tokenize(self, text: str, add_bos: bool = True) -> list:
        # Returns a standard Python list for easy printing
        ids = self.sp.EncodeAsIds(text)
        if add_bos:
            # 2 is the Beginning of Sentence token for Gemma
            ids = [2] + ids
        return ids

    def decode(self, ids: list) -> str:
        return self.sp.DecodeIds(ids)


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize a string using the Gemma tokenizer."
    )
    parser.add_argument("text", type=str, help="The text string to tokenize.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="tokenizer.model",
        help="Path to the tokenizer.model file (default: tokenizer.model)",
    )

    args = parser.parse_args()

    try:
        tokenizer = GemmaTokenizer(args.model_path)
        token_ids = tokenizer.tokenize(args.text)

        print(f"\nString: '{args.text}'")
        print(f"Token IDs: {token_ids}")
        print(f"Count: {len(token_ids)}\n")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
