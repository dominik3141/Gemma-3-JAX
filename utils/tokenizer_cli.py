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
        ids = self.sp.EncodeAsIds(text)
        if add_bos:
            ids = [2] + ids
        return ids

    def decode(self, ids: list) -> str:
        return self.sp.DecodeIds(ids)


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize text or decode IDs using the Gemma tokenizer."
    )
    parser.add_argument(
        "input",
        type=str,
        help="The text string to tokenize, or space-separated IDs to decode.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="data/tokenizer.model",
        help="Path to the tokenizer.model file (default: data/tokenizer.model)",
    )
    parser.add_argument(
        "--decode",
        action="store_true",
        help="Switch mode to DECODE (input should be space-separated integers).",
    )

    args = parser.parse_args()

    try:
        tokenizer = GemmaTokenizer(args.model_path)

        if args.decode:
            # Parse the input string into a list of integers
            # Example input: "2 109 2351" -> [2, 109, 2351]
            try:
                ids = [int(x) for x in args.input.split()]
            except ValueError:
                print(
                    "Error: In decode mode, input must be a list of space-separated integers."
                )
                sys.exit(1)

            decoded_text = tokenizer.decode(ids)
            print(f"\nIDs: {ids}")
            print(f"Decoded String: '{decoded_text}'\n")

        else:
            # Default Encode mode
            token_ids = tokenizer.tokenize(args.input)
            print(f"\nString: '{args.input}'")
            print(f"Token IDs: {token_ids}")
            print(f"Count: {len(token_ids)}\n")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
