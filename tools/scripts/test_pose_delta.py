"""A/B pose injection modes: confirm visible movement with a reference image."""

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
REF_CANDIDATES = [
    ROOT / "refs" / "train_char_1.png",
    ROOT / "refs" / "train_char_2.png",
]
OUT = ROOT / "outputs" / "pose_delta_ab"
PROMPT = "masterpiece, best quality, 1girl, solo, anime style, looking at viewer"
NEG = "lowres, bad anatomy, worst quality, low quality, blurry"
SEED = 42
STEPS = 20
CFG = 5.0
IP_SCALE = 1.0
RES = 512


def rest_pose() -> list[float]:
    v = {n: 0.0 for n in PARAM_NAMES}
    v["ParamEyeLOpen"] = 1.0
    v["ParamEyeROpen"] = 1.0
    v["EyeOpenLeft"] = 1.0
    v["EyeOpenRight"] = 1.0
    return [v[n] for n in PARAM_NAMES]


def mouth_open() -> list[float]:
    v = rest_pose()
    v[PARAM_NAMES.index("ParamMouthOpenY")] = 1.0
    v[PARAM_NAMES.index("MouthOpen")] = 1.0
    return v


def angle_right() -> list[float]:
    v = rest_pose()
    # unit space ≈ slider / 30
    for name in ("ParamAngleY", "FaceAngleY"):
        v[PARAM_NAMES.index(name)] = 1.0
    return v


def eyes_closed() -> list[float]:
    v = rest_pose()
    for name in ("ParamEyeLOpen", "ParamEyeROpen", "EyeOpenLeft", "EyeOpenRight"):
        v[PARAM_NAMES.index(name)] = 0.0
    return v


def build_pose_tokens(
    adapter: PoseAdapter,
    pose: torch.Tensor,
    neutral: torch.Tensor,
    mode: str,
    gain: float,
    target_rms: float,
) -> torch.Tensor:
    """Return pose-side tokens to concat onto text (B, 16, 768)."""
    with torch.inference_mode():
        tok = adapter(pose, train=False)
        neu = adapter(neutral, train=False)
        if mode == "blend_abs":
            return tok * gain
        if mode == "delta":
            return (tok - neu) * gain
        if mode == "delta_rms":
            delta = tok - neu
            rms = delta.float().pow(2).mean().sqrt().clamp_min(1e-6)
            return (delta / rms.to(delta.dtype)) * target_rms
        if mode == "delta_rms_gain":
            delta = tok - neu
            rms = delta.float().pow(2).mean().sqrt().clamp_min(1e-6)
            return (delta / rms.to(delta.dtype)) * target_rms * gain
        raise ValueError(mode)


@torch.inference_mode()
def generate(
    pipe,
    adapter,
    ref: Image.Image,
    pose: list[float],
    neutral: list[float],
    mode: str,
    gain: float,
    target_rms: float,
    device: str,
    dtype: torch.dtype,
) -> Image.Image:
    tok = pipe.tokenizer
    text_inputs = tok(
        [PROMPT],
        padding="max_length",
        max_length=tok.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_emb = pipe.text_encoder(text_inputs.input_ids.to(device))[0].to(dtype=dtype)
    pose_t = torch.tensor([pose], dtype=dtype, device=device)
    neu_t = torch.tensor([neutral], dtype=dtype, device=device)
    pose_tokens = build_pose_tokens(adapter, pose_t, neu_t, mode, gain, target_rms)
    encoder_hidden = torch.cat([text_emb, pose_tokens], dim=1)

    uncond_inputs = tok(
        [NEG],
        padding="max_length",
        max_length=tok.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_emb = pipe.text_encoder(uncond_inputs.input_ids.to(device))[0].to(dtype=dtype)
    uncond_pose = torch.zeros_like(pose_tokens)
    prompt_embeds = torch.cat(
        [torch.cat([uncond_emb, uncond_pose], dim=1), encoder_hidden], dim=0
    )

    ip_embeds = pipe.prepare_ip_adapter_image_embeds(
        ip_adapter_image=[ref],
        ip_adapter_image_embeds=None,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )
    ip_embeds = [e.to(device=device, dtype=dtype) for e in ip_embeds]

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    latents = pipe.prepare_latents(
        1, pipe.unet.config.in_channels, RES, RES, dtype, device, generator
    )
    pipe.scheduler.set_timesteps(STEPS, device=device)
    for t in pipe.scheduler.timesteps:
        latent_in = torch.cat([latents, latents], dim=0)
        latent_in = pipe.scheduler.scale_model_input(latent_in, t)
        noise_pred = pipe.unet(
            latent_in,
            t,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={"image_embeds": ip_embeds},
            return_dict=False,
        )[0]
        n_u, n_c = noise_pred.chunk(2)
        noise_pred = n_u + CFG * (n_c - n_u)
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    decoded = (decoded.float().clamp(-1, 1) + 1) / 2
    arr = (decoded * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
    return Image.fromarray(arr)


def mae(a: Image.Image, b: Image.Image) -> float:
    A = np.asarray(a.convert("RGB"), dtype=np.float32)
    B = np.asarray(b.convert("RGB"), dtype=np.float32)
    return float(np.mean(np.abs(A - B)))


def main() -> None:
    ref_path = next((p for p in REF_CANDIDATES if p.is_file()), None)
    if ref_path is None:
        raise SystemExit("No reference image found")
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"device={device} ref={ref_path}")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        str(IP), subfolder="models/image_encoder", torch_dtype=dtype
    )
    pipe = StableDiffusionPipeline.from_single_file(
        str(MODEL),
        torch_dtype=dtype,
        safety_checker=None,
        image_encoder=image_encoder,
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

    neutral = rest_pose()
    poses = {
        "rest": rest_pose(),
        "mouth": mouth_open(),
        "angleY": angle_right(),
        "eyes_closed": eyes_closed(),
    }

    # modes: name -> (mode, gain, target_rms)
    modes = {
        "old_blend08": ("blend_abs", 0.08, 0.0),
        "delta_g4": ("delta", 4.0, 0.0),
        "delta_g8": ("delta", 8.0, 0.0),
        "delta_rms015": ("delta_rms", 1.0, 0.15),
        "delta_rms030": ("delta_rms", 1.0, 0.30),
        "delta_rms050": ("delta_rms", 1.0, 0.50),
        "delta_rms030_g2": ("delta_rms_gain", 2.0, 0.30),
    }

    results: dict[str, dict[str, Image.Image]] = {}
    for mname, (mode, gain, trms) in modes.items():
        results[mname] = {}
        for pname, pose in poses.items():
            img = generate(
                pipe, adapter, Image.open(ref_path).convert("RGB"),
                pose, neutral, mode, gain, trms, device, dtype,
            )
            path = OUT / f"{mname}__{pname}.png"
            img.save(path)
            mean = float(np.asarray(img).mean())
            print(f"saved {path.name} mean={mean:.1f}")
            results[mname][pname] = img

    print("\n=== MAE vs rest (higher = more pose movement) ===")
    for mname in modes:
        row = results[mname]
        parts = []
        for pname in ("mouth", "angleY", "eyes_closed"):
            parts.append(f"{pname}={mae(row['rest'], row[pname]):.2f}")
        # collapse heuristic: very low mean or NaN already caught; check std
        std = float(np.asarray(row["rest"]).std())
        print(f"{mname:18s} std_rest={std:5.1f}  " + "  ".join(parts))


if __name__ == "__main__":
    main()
