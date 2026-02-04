import argparse
import os

from setup import download_weights


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage weights for RL training.")
    parser.add_argument("--model-size", default="27b", choices=["1b", "27b"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    download_weights(args.model_size)
    sentinel = (
        "model_stacked_pt.safetensors"
        if args.model_size == "1b"
        else "model.safetensors.index.json"
    )
    if not os.path.exists(os.path.join(f"data/gemma-3-{args.model_size}", sentinel)):
        raise RuntimeError("Weights not present after staging.")
    print("Stage complete.")


if __name__ == "__main__":
    main()
