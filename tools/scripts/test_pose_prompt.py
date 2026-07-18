"""Confirm visible pose change WITH reference identity via prompt tags + optional delta."""

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

from training.pose_adapter import PoseAdapter  # noqa: E402
from utils.params import NUM_PARAMS, PARAM_NAMES  # noqa: E402

MODEL = ROOT / "models" / "AnythingV5V3_v5PrtRE.safetensors"
IP = ROOT / "models" / "ip-adapter"
CKPT = ROOT / "models" / "finetuned" / "checkpoints" / "pose_adapter_step_015000.pt"
REF = ROOT / "refs" / "train_char_1.png"
OUT = ROOT / "outputs" / "pose_prompt_ab"
BASE = "masterpiece, best quality, 1girl, solo, anime style"
NEG = "lowres, bad anatomy, worst quality, low quality, blurry, duplicate"
SEED = 42
STEPS = 22
CFG = 5.5
IP_SCALE = 1.15
RES = 512


def rest_pose() -> list[float]:
    v = {n: 0.0 for n in PARAM_NAMES}
    for n in ("ParamEyeLOpen", "ParamEyeROpen", "EyeOpenLeft", "EyeOpenRight"):
        v[n] = 1.0
    return [v[n] for n in PARAM_NAMES]


def pose_to_prompt(pose: list[float]) -> str:
    d = dict(zip(PARAM_NAMES, pose))
    tags = [BASE, "looking at viewer"]
    ax = d["ParamAngleX"]  # yaw L/R
    ay = d["ParamAngleY"]  # pitch U/D
    az = d["ParamAngleZ"]
    if ax > 0.45:
        tags.append("face turned to the right, looking right, three-quarter view from the right")
    elif ax < -0.45:
        tags.append("face turned to the left, looking left, three-quarter view from the left")
    if ay > 0.45:
        tags.append("looking up")
    elif ay < -0.45:
        tags.append("looking down, head tilted down")
    if az > 0.35:
        tags.append("head tilted")
    mouth = max(d["ParamMouthOpenY"], d["MouthOpen"])
    if mouth > 0.65:
        tags.append("open mouth, talking")
    elif mouth > 0.25:
        tags.append("slightly open mouth")
    else:
        tags.append("closed mouth")
    smile = d.get("MouthSmile", 0.0) + d.get("ParamMouthForm", 0.0)
    if smile > 0.4:
        tags.append("smile")
    eye = min(d["ParamEyeLOpen"], d["ParamEyeROpen"], d["EyeOpenLeft"], d["EyeOpenRight"])
    if eye < 0.2:
        tags.append("closed eyes")
    elif eye < 0.55:
        tags.append("half-closed eyes")
    else:
        tags.append("eyes open")
    return ", ".join(tags)


def make_poses() -> dict[str, list[float]]:
    rest = rest_pose()
    mouth = rest[:]
    mouth[PARAM_NAMES.index("ParamMouthOpenY")] = 1.0
    mouth[PARAM_NAMES.index("MouthOpen")] = 1.0
    angle = rest[:]
    for n in ("ParamAngleX", "FaceAngleX"):
        angle[PARAM_NAMES.index(n)] = -1.0
    eyes = rest[:]
    for n in ("ParamEyeLOpen", "ParamEyeROpen", "EyeOpenLeft", "EyeOpenRight"):
        eyes[PARAM_NAMES.index(n)] = 0.0
    look_up = rest[:]
    for n in ("ParamAngleY", "FaceAngleY"):
        look_up[PARAM_NAMES.index(n)] = 0.8
    return {
        "rest": rest,
        "mouth_open": mouth,
        "turn_left": angle,
        "eyes_closed": eyes,
        "look_up": look_up,
    }


@torch.inference_mode()
def gen_ip_prompt(pipe, ref, prompt: str, device, dtype) -> Image.Image:
    g = torch.Generator(device="cpu").manual_seed(SEED)
    out = pipe(
        prompt=prompt,
        negative_prompt=NEG,
        ip_adapter_image=ref,
        num_inference_steps=STEPS,
        guidance_scale=CFG,
        width=RES,
        height=RES,
        generator=g,
    ).images[0]
    return out


@torch.inference_mode()
def gen_prompt_plus_delta(
    pipe, adapter, ref, prompt: str, pose: list[float], neutral: list[float],
    target_rms: float, device, dtype,
) -> Image.Image:
    tok = pipe.tokenizer
    text_inputs = tok([prompt], padding="max_length", max_length=tok.model_max_length,
                      truncation=True, return_tensors="pt")
    text_emb = pipe.text_encoder(text_inputs.input_ids.to(device))[0].to(dtype=dtype)
    pose_t = torch.tensor([pose], dtype=dtype, device=device)
    neu_t = torch.tensor([neutral], dtype=dtype, device=device)
    delta = adapter(pose_t, train=False) - adapter(neu_t, train=False)
    rms = delta.float().pow(2).mean().sqrt().clamp_min(1e-6)
    pose_tokens = (delta / rms.to(delta.dtype)) * target_rms
    # If pose ~= neutral, delta~0 — keep zeros
    if float(rms) < 1e-4:
        pose_tokens = torch.zeros_like(pose_tokens)
    enc = torch.cat([text_emb, pose_tokens], dim=1)

    uncond = tok([NEG], padding="max_length", max_length=tok.model_max_length,
                 truncation=True, return_tensors="pt")
    uemb = pipe.text_encoder(uncond.input_ids.to(device))[0].to(dtype=dtype)
    prompt_embeds = torch.cat([torch.cat([uemb, torch.zeros_like(pose_tokens)], 1), enc], 0)

    ip = pipe.prepare_ip_adapter_image_embeds(
        ip_adapter_image=[ref], ip_adapter_image_embeds=None, device=device,
        num_images_per_prompt=1, do_classifier_free_guidance=True,
    )
    ip = [e.to(device=device, dtype=dtype) for e in ip]
    g = torch.Generator(device="cpu").manual_seed(SEED)
    latents = pipe.prepare_latents(1, pipe.unet.config.in_channels, RES, RES, dtype, device, g)
    pipe.scheduler.set_timesteps(STEPS, device=device)
    for t in pipe.scheduler.timesteps:
        lin = pipe.scheduler.scale_model_input(torch.cat([latents, latents], 0), t)
        noise = pipe.unet(lin, t, encoder_hidden_states=prompt_embeds,
                          added_cond_kwargs={"image_embeds": ip}, return_dict=False)[0]
        nu, nc = noise.chunk(2)
        latents = pipe.scheduler.step(nu + CFG * (nc - nu), t, latents, return_dict=False)[0]
    dec = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    dec = (dec.float().clamp(-1, 1) + 1) / 2
    arr = (dec * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
    return Image.fromarray(arr)


def mae(a, b) -> float:
    return float(np.mean(np.abs(
        np.asarray(a.convert("RGB"), np.float32) - np.asarray(b.convert("RGB"), np.float32)
    )))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    ref = Image.open(REF).convert("RGB")

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        str(IP), subfolder="models/image_encoder", torch_dtype=dtype
    )
    pipe = StableDiffusionPipeline.from_single_file(
        str(MODEL), torch_dtype=dtype, safety_checker=None, image_encoder=image_encoder,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.load_ip_adapter(str(IP), subfolder="models", weight_name="ip-adapter_sd15_light.bin")
    pipe.set_ip_adapter_scale(IP_SCALE)

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config") or {}
    adapter = PoseAdapter(
        num_params=NUM_PARAMS,
        cross_attention_dim=pipe.unet.config.cross_attention_dim,
        mlp_hidden=int(cfg.get("mlp_hidden", 512)),
        num_tokens_per_param=int(cfg.get("num_tokens_per_param", 1)),
        drop_prob=0.0,
    )
    adapter.load_state_dict(ckpt["pose_adapter"])
    adapter.to(device=device, dtype=dtype).eval()

    poses = make_poses()
    neutral = rest_pose()

    print("=== IP + pose-prompt tags ===")
    prompt_imgs = {}
    for name, pose in poses.items():
        prompt = pose_to_prompt(pose)
        print(f"{name}: {prompt}")
        img = gen_ip_prompt(pipe, ref, prompt, device, dtype)
        img.save(OUT / f"prompt__{name}.png")
        prompt_imgs[name] = img
        print(f"  saved mean={float(np.asarray(img).mean()):.1f}")

    print("\nMAE vs rest (prompt-only):")
    for name in poses:
        if name == "rest":
            continue
        print(f"  {name}: {mae(prompt_imgs['rest'], prompt_imgs[name]):.2f}")

    print("\n=== IP + pose-prompt + tiny delta_rms0.12 ===")
    combo = {}
    for name, pose in poses.items():
        prompt = pose_to_prompt(pose)
        img = gen_prompt_plus_delta(
            pipe, adapter, ref, prompt, pose, neutral, 0.12, device, dtype
        )
        img.save(OUT / f"combo__{name}.png")
        combo[name] = img
        print(f"  {name} mean={float(np.asarray(img).mean()):.1f}")
    print("MAE vs rest (combo):")
    for name in poses:
        if name == "rest":
            continue
        print(f"  {name}: {mae(combo['rest'], combo[name]):.2f}")


if __name__ == "__main__":
    main()
