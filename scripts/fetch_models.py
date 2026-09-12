#!/usr/bin/env python3
"""Download the AI image models. Never runs automatically.

Models are large and the system works completely without them — the procedural
generators need no weights at all — so fetching them is always a deliberate,
separate step, run when you decide to and not on someone's first boot.

    python scripts/fetch_models.py --backend cuda      # HuggingFace snapshot
    python scripts/fetch_models.py --backend openvino  # convert to OpenVINO IR
    python scripts/fetch_models.py --backend sdcpp --url <gguf-url>
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import load_config
from app.core.paths import Paths

CHUNK = 1024 * 1024
KB = 1024
# A rough figure so the disk check can warn before a long download fails.
APPROXIMATE_SIZE_GB = {"cuda": 3.0, "openvino": 3.5, "sdcpp": 2.0}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < KB:
            return f"{n:.1f} {unit}"
        n /= KB
    return f"{n:.1f} TB"


def check_space(target: Path, needed_gb: float) -> bool:
    target.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(target).free / 1024**3
    if free_gb < needed_gb * 1.5:
        print(f"✖ only {free_gb:.1f} GB free at {target}; about {needed_gb:.1f} GB is needed")
        return False
    return True


def fetch_huggingface(model_id: str, target: Path, *, openvino: bool) -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("✖ huggingface_hub is not installed.")
        print(
            "  pip install huggingface_hub" + (" optimum[openvino]" if openvino else " diffusers")
        )
        return 1

    print(f"→ downloading {model_id} into {target}")
    print("  This is several gigabytes and will take a while. Ctrl-C is safe;")
    print("  the download resumes from where it stopped.")
    try:
        path = snapshot_download(repo_id=model_id, cache_dir=str(target))
    except Exception as exc:
        print(f"✖ download failed: {exc}")
        return 1
    print(f"✔ model in {path}")

    if not openvino:
        return 0

    try:
        from optimum.intel import OVStableDiffusionPipeline
    except ImportError:
        print("✖ optimum-intel is not installed. pip install 'optimum[openvino]'")
        return 1

    converted = target / "openvino" / model_id.replace("/", "--")
    print(f"→ converting to OpenVINO IR in {converted} (this also takes a while)")
    try:
        pipeline = OVStableDiffusionPipeline.from_pretrained(model_id, export=True)
        pipeline.save_pretrained(str(converted))
    except Exception as exc:
        print(f"✖ conversion failed: {exc}")
        return 1
    print(f"✔ converted model in {converted}")
    return 0


def fetch_url(url: str, target: Path) -> int:
    target.mkdir(parents=True, exist_ok=True)
    name = url.rstrip("/").split("/")[-1] or "model.gguf"
    destination = target / name
    if destination.exists():
        print(f"✔ {destination} already exists")
        return 0

    print(f"→ downloading {url}")
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as response:  # noqa: S310 - operator-supplied URL
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            with partial.open("wb") as handle:
                while chunk := response.read(CHUNK):
                    handle.write(chunk)
                    done += len(chunk)
                    if total:
                        print(
                            f"\r  {human(done)} / {human(total)}" f" ({done * 100 // total}%)",
                            end="",
                            flush=True,
                        )
        print()
    except Exception as exc:
        partial.unlink(missing_ok=True)
        print(f"\n✖ download failed: {exc}")
        return 1

    partial.replace(destination)
    print(f"✔ {destination} ({human(destination.stat().st_size)})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Download AI image models")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None, help="override visual.ai.model_id")
    parser.add_argument(
        "--backend",
        default=None,
        choices=["cuda", "openvino", "sdcpp"],
        help="which backend to prepare (default: whatever is configured)",
    )
    parser.add_argument("--url", default=None, help="direct URL, for --backend sdcpp")
    parser.add_argument("--force", action="store_true", help="ignore the disk-space check")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = Paths.from_config(cfg)
    paths.ensure()

    model = args.model or cfg.visual.ai.model_id
    backend = args.backend or cfg.visual.ai.backend
    if backend in {"auto", "none"}:
        print(f"visual.ai.backend is {backend!r}; say which one to prepare:")
        print("  --backend cuda | openvino | sdcpp")
        return 2

    print(f"backend : {backend}")
    print(f"model   : {model}")
    print(f"into    : {paths.models}")
    print()

    if not args.force and not check_space(paths.models, APPROXIMATE_SIZE_GB.get(backend, 3.0)):
        print("  (pass --force to try anyway)")
        return 1

    if backend == "sdcpp":
        if not args.url:
            print("✖ --backend sdcpp needs --url pointing at a .gguf or .safetensors file.")
            print("  Quantised SD-Turbo weights are published on HuggingFace; pick one and")
            print("  pass its download URL. The stable-diffusion.cpp binary itself must")
            print("  already be on PATH as 'sd'.")
            return 2
        return fetch_url(args.url, paths.models / "sdcpp")

    return fetch_huggingface(model, paths.models, openvino=(backend == "openvino"))


if __name__ == "__main__":
    raise SystemExit(main())
