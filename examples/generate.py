from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_model
from tokenizers import Tokenizer

from barunlm import BarunConfig, BarunLM

REPO_ID = "harrrshall/BarunLM-35M"
EXPECTED_SHA256 = {
    "model.safetensors": "f2a7c88b9f2c2e3584809081407ab136795d82e30e89b730e007781c45d01447",
    "barun_config.json": "9b3a1d71baa95a198744d250f9629231738d942570b8685c44307fd83dd33565",
    "tokenizer.json": "70ded9605fccd09c2340ca7e225361eab0ae8b4dbbb0d6e26343ab5183979db6",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with BarunLM-35M.")
    parser.add_argument("--prompt", default="The future of small language models is")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--model-dir", type=Path)
    return parser.parse_args()


def resolve_model_dir(path: Path | None) -> Path:
    if path is not None:
        return path.resolve()
    return Path(
        snapshot_download(
            REPO_ID,
            allow_patterns=["model.safetensors", "barun_config.json", "tokenizer.json"],
        )
    )


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative")

    model_dir = resolve_model_dir(args.model_dir)
    for name, expected in EXPECTED_SHA256.items():
        actual = sha256(model_dir / name)
        if actual != expected:
            raise RuntimeError(f"{name} SHA-256 mismatch: expected {expected}, got {actual}")

    config = BarunConfig.from_json(model_dir / "barun_config.json")
    model = BarunLM(config)
    missing, unexpected = load_model(model, model_dir / "model.safetensors", strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    if model.parameter_counts()["total"] != 35_072_768:
        raise RuntimeError("unexpected parameter count")

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model.to(device=device, dtype=dtype).eval()

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    if not prompt_ids:
        raise ValueError("prompt must encode to at least one token")
    if len(prompt_ids) + args.max_new_tokens > config.max_seq_len:
        raise ValueError("prompt and continuation exceed the 2,048-token context")

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    print(tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main()
