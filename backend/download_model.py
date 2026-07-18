"""Download Anything V5 (SD 1.5) + IP-Adapter into data/models/."""

from __future__ import annotations

from huggingface_hub import hf_hub_download, snapshot_download

from backend.paths import ip_adapter_dir, models_dir

FILENAME = "AnythingV5V3_v5PrtRE.safetensors"
REPO = "ckpt/anything-v5.0"


def download_checkpoint() -> None:
    models = models_dir()
    models.mkdir(parents=True, exist_ok=True)
    dest = models / FILENAME
    if dest.is_file() and dest.stat().st_size > 1_000_000_000:
        print(f"Already present: {dest}")
        return
    print(f"Downloading {REPO}/{FILENAME} ...")
    path = hf_hub_download(
        repo_id=REPO,
        filename=FILENAME,
        local_dir=str(models),
    )
    print(f"Saved: {path}")


def download_ip_adapter() -> None:
    ip_dir = ip_adapter_dir()
    marker = ip_dir / "models" / "ip-adapter_sd15.bin"
    if marker.is_file() and marker.stat().st_size > 10_000_000:
        print(f"Already present: {marker}")
        return
    print("Downloading h94/IP-Adapter (SD1.5 + image encoder) ...")
    path = snapshot_download(
        repo_id="h94/IP-Adapter",
        allow_patterns=[
            "models/image_encoder/*",
            "models/ip-adapter_sd15.bin",
            "models/ip-adapter_sd15_light.bin",
            "models/ip-adapter-plus_sd15.bin",
        ],
        local_dir=str(ip_dir),
    )
    print(f"Saved: {path}")


def main() -> None:
    download_checkpoint()
    download_ip_adapter()


if __name__ == "__main__":
    main()
