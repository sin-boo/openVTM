"""Confirm prompt-tag pose drive: same ref, same seed, visible expression change."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from transformers import CLIPVisionModelWithProjection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test_pose_ui import pose_params_to_tags, rest_pose_vector  # noqa: E402
from utils.params import PARAM_NAMES  # noqa: E402

MODEL = ROOT / "models" / "AnythingV5V3_v5PrtRE.safetensors"
IP = ROOT / "models" / "ip-adapter"
REF = ROOT / "refs" / "train_char_1.png"
OUT = ROOT / "outputs" / "pose_confirm"
BASE = "masterpiece, best quality, 1girl, solo, anime style"
NEG = (
    "lowres, bad anatomy, bad hands, worst quality, low quality, blurry, "
    "jpeg artifacts, watermark, different character"
)
SEED = 42


def set_pose(**kwargs: float) -> list[float]:
    v = rest_pose_vector()
    d = dict(zip(PARAM_NAMES, v))
    d.update(kwargs)
    # Keep aliases in sync like the UI
    d["FaceAngleX"] = d["ParamAngleX"]
    d["FaceAngleY"] = d["ParamAngleY"]
    d["FaceAngleZ"] = d["ParamAngleZ"]
    d["MouthOpen"] = d["ParamMouthOpenY"]
    d["EyeOpenLeft"] = d["ParamEyeLOpen"]
    d["EyeOpenRight"] = d["ParamEyeROpen"]
    return [d[n] for n in PARAM_NAMES]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    dtype = torch.float16
    enc = CLIPVisionModelWithProjection.from_pretrained(
        str(IP), subfolder="models/image_encoder", torch_dtype=dtype
    )
    pipe = StableDiffusionPipeline.from_single_file(
        str(MODEL), torch_dtype=dtype, safety_checker=None, image_encoder=enc
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.load_ip_adapter(str(IP), subfolder="models", weight_name="ip-adapter-plus_sd15.bin")
    pipe.set_ip_adapter_scale(0.85)
    ref = Image.open(REF).convert("RGB")

    cases = {
        "rest": set_pose(),
        "mouth_open": set_pose(ParamMouthOpenY=1.0),
        "eyes_closed": set_pose(ParamEyeLOpen=0.0, ParamEyeROpen=0.0),
        "turn_left": set_pose(ParamAngleX=-1.0),
    }

    imgs = {}
    for name, pose in cases.items():
        tags = pose_params_to_tags(pose)
        prompt = f"{BASE}, {tags}"
        print(f"{name}: {prompt}")
        g = torch.Generator(device="cpu").manual_seed(SEED)
        img = pipe(
            prompt=prompt,
            negative_prompt=NEG,
            ip_adapter_image=ref,
            num_inference_steps=22,
            guidance_scale=5.0,
            width=512,
            height=512,
            generator=g,
        ).images[0]
        path = OUT / f"confirm__{name}.png"
        img.save(path)
        imgs[name] = img
        print(f"  saved {path.name} mean={float(np.asarray(img).mean()):.1f}")

    print("\nMAE vs rest:")
    rest = np.asarray(imgs["rest"], np.float32)
    ok = True
    for name, img in imgs.items():
        if name == "rest":
            continue
        m = float(np.mean(np.abs(rest - np.asarray(img, np.float32))))
        print(f"  {name}: {m:.2f}")
        if m < 3.0:
            ok = False
            print(f"  FAIL: {name} almost identical to rest")

    # Contact sheet
    w, h = imgs["rest"].size
    sheet = Image.new("RGB", (w * 4, h))
    for i, name in enumerate(("rest", "mouth_open", "eyes_closed", "turn_left")):
        sheet.paste(imgs[name], (i * w, 0))
    sheet_path = OUT / "confirm_strip.png"
    sheet.save(sheet_path)
    print(f"\nstrip: {sheet_path}")
    if not ok:
        raise SystemExit(1)
    print("PASS: pose sliders via prompt tags change the image with reference identity.")


if __name__ == "__main__":
    main()
