"""Stage-level timing for PoseEngine generate (prompt-tags path).

Breaks the default `pipe()` call into text encode, IP encode, UNet denoise,
and VAE decode so we can see the real bottleneck.
"""

from __future__ import annotations

import statistics as st
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.engine import (  # noqa: E402
    DEFAULT_CFG,
    DEFAULT_NEGATIVE,
    DEFAULT_PROMPT,
    DEFAULT_REF,
    DEFAULT_STEPS,
    PoseEngine,
    best_pose_checkpoint,
    pose_params_to_tags,
    rest_pose_vector,
)
from backend.utils.params import PARAM_NAMES  # noqa: E402


def make_pose(**overrides: float) -> list[float]:
    values = {name: 0.0 for name in PARAM_NAMES}
    for name in ("ParamEyeLOpen", "ParamEyeROpen", "EyeOpenLeft", "EyeOpenRight"):
        values[name] = 1.0
    values.update(overrides)
    values["FaceAngleX"] = values["ParamAngleX"]
    values["FaceAngleY"] = values["ParamAngleY"]
    values["FaceAngleZ"] = values["ParamAngleZ"]
    values["MouthOpen"] = values["ParamMouthOpenY"]
    values["EyeOpenLeft"] = values["ParamEyeLOpen"]
    values["EyeOpenRight"] = values["ParamEyeROpen"]
    return [values[n] for n in PARAM_NAMES]


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    sync()
    t0 = time.perf_counter()
    out = fn()
    sync()
    return out, time.perf_counter() - t0


def profile_once(eng: PoseEngine, ref: Image.Image, pose: list[float], seed: int = 42):
    assert eng.pipe is not None
    pipe = eng.pipe
    device = eng.device
    dtype = pipe.unet.dtype
    steps = eng.steps
    cfg = eng.cfg
    resolution = eng.resolution
    do_cfg = cfg > 1.01

    normed = eng._normalize(pose, mode="slider_to_unit")
    pose_tags = pose_params_to_tags(normed)
    effective = f"{DEFAULT_PROMPT}, {pose_tags}"

    timings: dict[str, float] = {}

    # --- text encode ---
    def text_encode():
        tok = pipe.tokenizer
        pos = tok(
            [effective],
            padding="max_length",
            max_length=tok.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        neg = tok(
            [DEFAULT_NEGATIVE],
            padding="max_length",
            max_length=tok.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            pe = pipe.text_encoder(pos.input_ids.to(device))[0]
            ne = pipe.text_encoder(neg.input_ids.to(device))[0]
        return pe, ne

    (prompt_embeds, neg_embeds), timings["text_encode"] = timed(text_encode)
    if do_cfg:
        prompt_embeds = torch.cat([neg_embeds, prompt_embeds], dim=0)

    # --- IP-Adapter image encode (same path pipe uses) ---
    def ip_encode():
        # Prefer public helper if present; else fall back to encode_image via image_encoder.
        with torch.inference_mode():
            if hasattr(pipe, "prepare_ip_adapter_image_embeds"):
                embeds = pipe.prepare_ip_adapter_image_embeds(
                    ip_adapter_image=ref,
                    ip_adapter_image_embeds=None,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=do_cfg,
                )
                return embeds
            # Older / alternate API
            image_encoder = pipe.image_encoder
            from diffusers.utils import load_image  # noqa: F401

            clip = pipe.feature_extractor(images=ref, return_tensors="pt").pixel_values
            clip = clip.to(device=device, dtype=dtype)
            image_embeds = image_encoder(clip).image_embeds
            if do_cfg:
                image_embeds = torch.cat([torch.zeros_like(image_embeds), image_embeds])
            return [image_embeds]

    ip_embeds, timings["ip_encode"] = timed(ip_encode)

    # --- prepare latents ---
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def prep_latents():
        with torch.inference_mode():
            return pipe.prepare_latents(
                1,
                pipe.unet.config.in_channels,
                resolution,
                resolution,
                dtype,
                device,
                generator,
            )

    latents, timings["prep_latents"] = timed(prep_latents)

    # --- UNet denoise loop ---
    pipe.scheduler.set_timesteps(steps, device=device)
    step_times: list[float] = []

    def one_step(t, latents):
        with torch.inference_mode():
            latent_in = torch.cat([latents, latents], dim=0) if do_cfg else latents
            latent_in = pipe.scheduler.scale_model_input(latent_in, t)
            noise_pred = pipe.unet(
                latent_in,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={"image_embeds": ip_embeds},
                return_dict=False,
            )[0]
            if do_cfg:
                n_uncond, n_cond = noise_pred.chunk(2)
                noise_pred = n_uncond + cfg * (n_cond - n_uncond)
            return pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    sync()
    t_denoise0 = time.perf_counter()
    for t in pipe.scheduler.timesteps:
        latents, dt = timed(lambda t=t, latents=latents: one_step(t, latents))
        step_times.append(dt)
    sync()
    timings["unet_denoise"] = time.perf_counter() - t_denoise0
    timings["unet_step_mean"] = st.mean(step_times)
    timings["unet_step_max"] = max(step_times)

    # --- VAE decode ---
    def vae_decode():
        with torch.inference_mode():
            decoded = pipe.vae.decode(
                latents / pipe.vae.config.scaling_factor, return_dict=False
            )[0]
            decoded = (decoded.float().clamp(-1, 1) + 1) / 2
            arr = (
                (decoded * 255)
                .round()
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()[0]
            )
            return Image.fromarray(arr)

    image, timings["vae_decode"] = timed(vae_decode)

    # --- full pipe() for comparison ---
    def full_pipe():
        with torch.inference_mode():
            return pipe(
                prompt=effective,
                negative_prompt=DEFAULT_NEGATIVE,
                ip_adapter_image=ref,
                num_inference_steps=steps,
                guidance_scale=cfg,
                width=resolution,
                height=resolution,
                generator=torch.Generator(device="cpu").manual_seed(seed),
            ).images[0]

    _, timings["full_pipe"] = timed(full_pipe)
    timings["staged_sum"] = (
        timings["text_encode"]
        + timings["ip_encode"]
        + timings["prep_latents"]
        + timings["unet_denoise"]
        + timings["vae_decode"]
    )
    return image, timings


def fmt(seconds: float) -> str:
    return f"{seconds * 1000:7.1f} ms"


def main() -> None:
    ckpt = best_pose_checkpoint()
    if ckpt is None:
        raise SystemExit("No PoseAdapter checkpoint found")
    if not DEFAULT_REF.is_file():
        raise SystemExit(f"Missing reference image: {DEFAULT_REF}")

    ref = Image.open(DEFAULT_REF).convert("RGB")
    print(f"device check…")
    print(f"  torch={torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  gpu={torch.cuda.get_device_name(0)}")
    print(f"checkpoint={ckpt.name}")
    print(f"ref={DEFAULT_REF.name} size={ref.size}")
    print(f"steps={DEFAULT_STEPS} cfg={DEFAULT_CFG} resolution=512")

    eng = PoseEngine(steps=DEFAULT_STEPS, cfg=DEFAULT_CFG, log=print)
    t_load = time.perf_counter()
    step = eng.load(ckpt)
    load_s = time.perf_counter() - t_load
    print(f"load_time={load_s:.2f}s loaded_step={step}")

    print("\n=== warmup (full generate) ===")
    eng.warmup(ref, rest_pose_vector())

    pose = make_pose(ParamAngleY=-15.0, ParamMouthOpenY=0.4)
    runs = 3
    print(f"\n=== staged profile ({runs} runs) ===")
    all_t: dict[str, list[float]] = {}
    for i in range(runs):
        _, timings = profile_once(eng, ref, pose, seed=42 + i)
        print(f"\nrun {i + 1}/{runs}:")
        for k in (
            "text_encode",
            "ip_encode",
            "prep_latents",
            "unet_denoise",
            "unet_step_mean",
            "vae_decode",
            "staged_sum",
            "full_pipe",
        ):
            all_t.setdefault(k, []).append(timings[k])
            print(f"  {k:16s} {fmt(timings[k])}")

    print("\n=== mean over runs ===")
    means = {k: st.mean(v) for k, v in all_t.items()}
    ranked = sorted(
        ((k, v) for k, v in means.items() if k not in ("staged_sum", "full_pipe", "unet_step_mean")),
        key=lambda kv: kv[1],
        reverse=True,
    )
    total = means["staged_sum"]
    for k, v in ranked:
        pct = 100.0 * v / total if total > 0 else 0.0
        print(f"  {k:16s} {fmt(v)}  ({pct:5.1f}%)")
    print(f"  {'staged_sum':16s} {fmt(means['staged_sum'])}")
    print(f"  {'full_pipe':16s} {fmt(means['full_pipe'])}")
    print(f"  {'unet_step_mean':16s} {fmt(means['unet_step_mean'])}  (x{DEFAULT_STEPS} steps, CFG={DEFAULT_CFG})")
    print(f"\nBOTTLENECK: {ranked[0][0]} ({100.0 * ranked[0][1] / total:.1f}% of staged generate)")


if __name__ == "__main__":
    main()
