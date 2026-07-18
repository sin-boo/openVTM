"""Anything V5 + IP-Adapter + PoseAdapter inference engine (fast path)."""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from PIL import Image

from backend.paths import (
    ckpt_dir,
    default_ref_path,
    ip_adapter_dir,
    model_path,
    outputs_dir,
    param_stats_path,
)
from backend.training.pose_adapter import PoseAdapter
from backend.utils.params import NUM_PARAMS, PARAM_NAMES, normalize_params

# Dev/script convenience (resolved at import). Runtime code should call path helpers.
MODEL_PATH = model_path()
IP_ADAPTER_DIR = ip_adapter_dir()
CKPT_DIR = ckpt_dir()
PARAM_STATS = param_stats_path()
DEFAULT_REF = default_ref_path()
OUTPUT_DIR = outputs_dir()

DEFAULT_PROMPT = "masterpiece, best quality, 1girl, solo, anime style"
DEFAULT_NEGATIVE = (
    "lowres, bad anatomy, bad hands, worst quality, low quality, blurry, "
    "jpeg artifacts, watermark, different character"
)

# Tuned for latency on consumer GPUs (3060 Ti / 40xx / 50xx).
DEFAULT_STEPS = 8
DEFAULT_CFG = 4.5
DEFAULT_IP_SCALE = 0.85
DEFAULT_RESOLUTION = 512

IP_ADAPTER_WEIGHTS = {
    "plus (closer to image)": "ip-adapter-plus_sd15.bin",
    "standard (balanced)": "ip-adapter_sd15.bin",
    "light (follows text more)": "ip-adapter_sd15_light.bin",
}

POSE_DRIVE_PROMPT = "prompt tags (visible motion)"
POSE_DRIVE_ADAPTER = "adapter delta (experimental)"
POSE_DRIVE_OFF = "off (IP only)"
POSE_DRIVE_MODES = (POSE_DRIVE_PROMPT, POSE_DRIVE_ADAPTER, POSE_DRIVE_OFF)

_STEP_RE = re.compile(r"pose_adapter_step_(\d+)\.pt$", re.I)

LogFn = Callable[[str], None]


def rest_pose_vector() -> list[float]:
    values = {name: 0.0 for name in PARAM_NAMES}
    for name in ("ParamEyeLOpen", "ParamEyeROpen", "EyeOpenLeft", "EyeOpenRight"):
        values[name] = 1.0
    return [values[n] for n in PARAM_NAMES]


def pose_params_to_tags(pose: Sequence[float]) -> str:
    """Map normalized Live2D params → SD prompt tags.

    Convention matches tracking (`face_to_param_values`) and VTS:
      ParamAngleX = yaw   (negative → screen-left, positive → screen-right)
      ParamAngleY = pitch (positive → up, negative → down)
      ParamAngleZ = roll
    Values are expected in roughly [-1, 1] after `slider_to_unit`.
    """
    d = dict(zip(PARAM_NAMES, pose))
    tags: list[str] = []
    ax = float(d["ParamAngleX"])  # yaw L/R
    ay = float(d["ParamAngleY"])  # pitch U/D
    az = float(d["ParamAngleZ"])
    # Graduated yaw tags — left/right was previously wired to ParamAngleY (bug).
    if ax > 0.55:
        tags.append(
            "face turned to the right, looking right, three-quarter view from the right"
        )
    elif ax > 0.2:
        tags.append("head turned slightly to the right, face angled right")
    elif ax < -0.55:
        tags.append(
            "face turned to the left, looking left, three-quarter view from the left"
        )
    elif ax < -0.2:
        tags.append("head turned slightly to the left, face angled left")
    else:
        tags.append("looking at viewer, facing viewer")
    if ay > 0.35:
        tags.append("looking up, chin up")
    elif ay < -0.35:
        tags.append("looking down, head tilted down")
    if abs(az) > 0.3:
        tags.append("head tilted")

    mouth = max(float(d["ParamMouthOpenY"]), float(d["MouthOpen"]))
    if mouth > 0.65:
        tags.append("open mouth, talking, speaking")
    elif mouth > 0.25:
        tags.append("slightly open mouth")
    else:
        tags.append("closed mouth")

    smile = float(d.get("MouthSmile", 0.0)) + float(d.get("ParamMouthForm", 0.0))
    if smile > 0.35:
        tags.append("smile")

    eye = min(
        float(d["ParamEyeLOpen"]),
        float(d["ParamEyeROpen"]),
        float(d["EyeOpenLeft"]),
        float(d["EyeOpenRight"]),
    )
    if eye < 0.2:
        tags.append("closed eyes")
    elif eye < 0.55:
        tags.append("half-closed eyes")
    else:
        tags.append("eyes open")

    ebx, eby = float(d["ParamEyeBallX"]), float(d["ParamEyeBallY"])
    if ebx > 0.4:
        tags.append("looking to the left")
    elif ebx < -0.4:
        tags.append("looking to the right")
    if eby > 0.4:
        tags.append("looking up")
    elif eby < -0.4:
        tags.append("looking down")
    return ", ".join(tags)


def slider_to_unit(raw: Sequence[float]) -> list[float]:
    out: list[float] = []
    for name, v in zip(PARAM_NAMES, raw):
        if "Angle" in name:
            out.append(max(-1.0, min(1.0, float(v) / 30.0)))
        else:
            out.append(max(-1.0, min(1.0, float(v))))
    return out


def list_pose_checkpoints() -> list[tuple[str, Path]]:
    directory = ckpt_dir()
    if not directory.is_dir():
        return []
    items: list[tuple[str, Path]] = []
    for path in directory.glob("pose_adapter_*.pt"):
        name = path.name
        if name == "pose_adapter_latest.pt":
            label = "latest"
        elif name == "pose_adapter_final.pt":
            label = "final"
        else:
            m = _STEP_RE.match(name)
            label = f"step {int(m.group(1)):,}" if m else name
        items.append((label, path))

    def sort_key(item: tuple[str, Path]) -> tuple[int, int, str]:
        label, path = item
        m = _STEP_RE.match(path.name)
        if m:
            # Highest step first in spirit; UI still sorts ascending for dropdown —
            # default picker uses best_pose_checkpoint().
            return (0, int(m.group(1)), label)
        if path.name == "pose_adapter_latest.pt":
            return (1, 0, label)
        if path.name == "pose_adapter_final.pt":
            return (2, 0, label)
        return (3, 0, label)

    items.sort(key=sort_key)
    return items


def best_pose_checkpoint() -> Path | None:
    """Prefer highest step_XXXXX.pt (e.g. 20k). Ignore stale latest/final aliases."""
    step_ckpts: list[tuple[int, Path]] = []
    for label, path in list_pose_checkpoints():
        m = _STEP_RE.match(path.name)
        if m:
            step_ckpts.append((int(m.group(1)), path))
    if not step_ckpts:
        # Fall back to any listed checkpoint.
        items = list_pose_checkpoints()
        return items[-1][1] if items else None
    return max(step_ckpts, key=lambda x: x[0])[1]


def best_pose_checkpoint_label() -> str:
    path = best_pose_checkpoint()
    if path is None:
        return ""
    m = _STEP_RE.match(path.name)
    if m:
        return f"step {int(m.group(1)):,}"
    return path.name


@dataclass
class GenerateResult:
    image: Image.Image
    elapsed: float
    seed: int
    prompt: str
    pose_tags: str
    timings: dict[str, float]


class PoseEngine:
    """Load once, generate many frames. Caches IP embeds across consecutive poses."""

    def __init__(
        self,
        *,
        device: str | None = None,
        steps: int = DEFAULT_STEPS,
        cfg: float = DEFAULT_CFG,
        ip_scale: float = DEFAULT_IP_SCALE,
        resolution: int = DEFAULT_RESOLUTION,
        ip_weight: str = "ip-adapter-plus_sd15.bin",
        pose_drive: str = POSE_DRIVE_PROMPT,
        pose_target_rms: float = 0.12,
        compile_unet: bool = True,
        # Fraction of steps (0..1) that use CFG. 1.0 = every step (default).
        # e.g. 0.5 with 8 steps → CFG on first 4, single-pass on last 4.
        cfg_until: float = 1.0,
        log: LogFn | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.steps = int(steps)
        self.cfg = float(cfg)
        self.ip_scale = float(ip_scale)
        self.resolution = int(resolution)
        self.ip_weight = ip_weight
        self.pose_drive = pose_drive
        self.pose_target_rms = float(pose_target_rms)
        self.compile_unet = bool(compile_unet)
        self.cfg_until = float(cfg_until)
        self._log = log or (lambda _m: None)

        self.pipe = None
        self.pose_adapter: PoseAdapter | None = None
        self.loaded_ckpt: Path | None = None
        self._loaded_ip_weight: str | None = None
        self._param_stats: dict | None = None
        self._ip_cache_key: tuple | None = None
        self._ip_cache_embeds = None
        self._unet_compiled = False
        self.last_timings: dict[str, float] = {}

    def log(self, msg: str) -> None:
        self._log(msg)

    def _enable_fast_backends(self) -> None:
        if self.device != "cuda":
            return
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    def _maybe_compile_unet(self) -> None:
        """Compile UNet for faster denoise. Same weights → near-identical images."""
        assert self.pipe is not None
        if not self.compile_unet or self.device != "cuda" or self._unet_compiled:
            return
        if not hasattr(torch, "compile"):
            self.log("torch.compile unavailable — leaving UNet eager")
            return
        try:
            # dynamic=False: fixed shapes (CFG batch=2). Truncated CFG runs
            # no-CFG steps under torch._dynamo.disable() so compile stays stable.
            self.pipe.unet = torch.compile(
                self.pipe.unet,
                mode="default",
                fullgraph=False,
                dynamic=False,
            )
            self._unet_compiled = True
            self.log("UNet torch.compile enabled (mode=default)")
        except Exception as exc:
            self._unet_compiled = False
            self.log(f"torch.compile skipped: {exc}")

    def _encode_prompt_embeds(
        self,
        prompt: str,
        negative: str,
        *,
        dtype: torch.dtype,
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.pipe is not None
        tok = self.pipe.tokenizer
        pos = tok(
            [prompt],
            padding="max_length",
            max_length=tok.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        neg = tok(
            [negative or ""],
            padding="max_length",
            max_length=tok.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        pe = self.pipe.text_encoder(pos.input_ids.to(device))[0].to(dtype=dtype)
        ne = self.pipe.text_encoder(neg.input_ids.to(device))[0].to(dtype=dtype)
        return pe, ne

    def _vae_decode_image(self, latents: torch.Tensor) -> Image.Image:
        assert self.pipe is not None
        decoded = self.pipe.vae.decode(
            latents / self.pipe.vae.config.scaling_factor, return_dict=False
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

    def _denoise_truncated_cfg(
        self,
        *,
        prompt_embeds: torch.Tensor,
        negative_embeds: torch.Tensor,
        ip_embeds_cfg: list[torch.Tensor],
        seed: int,
        steps: int,
        cfg: float,
        cfg_until: float,
        resolution: int,
        dtype: torch.dtype,
        device: str,
    ) -> tuple[Image.Image, float, float]:
        """8 (or N) steps; CFG only on the first floor(steps * cfg_until) steps."""
        assert self.pipe is not None
        pipe = self.pipe
        cfg_until = max(0.0, min(1.0, float(cfg_until)))
        cfg_steps = int(steps * cfg_until + 1e-6)
        # Always at least 0; if cfg disabled entirely, all steps are single-pass.
        do_any_cfg = cfg > 1.01 and cfg_steps > 0

        ip_cond = [e.chunk(2)[1] if do_any_cfg and e.shape[0] >= 2 else e for e in ip_embeds_cfg]

        generator = torch.Generator(device="cpu").manual_seed(seed)
        latents = pipe.prepare_latents(
            1,
            pipe.unet.config.in_channels,
            resolution,
            resolution,
            dtype,
            device,
            generator,
        )
        pipe.scheduler.set_timesteps(steps, device=device)

        t_b = time.perf_counter()
        for i, t in enumerate(pipe.scheduler.timesteps):
            use_cfg = do_any_cfg and i < cfg_steps
            if use_cfg:
                latent_in = torch.cat([latents, latents], dim=0)
                text = torch.cat([negative_embeds, prompt_embeds], dim=0)
                ip = ip_embeds_cfg
            else:
                latent_in = latents
                text = prompt_embeds
                ip = ip_cond
            latent_in = pipe.scheduler.scale_model_input(latent_in, t)
            if use_cfg:
                noise_pred = pipe.unet(
                    latent_in,
                    t,
                    encoder_hidden_states=text,
                    added_cond_kwargs={"image_embeds": ip},
                    return_dict=False,
                )[0]
                n_uncond, n_cond = noise_pred.chunk(2)
                noise_pred = n_uncond + cfg * (n_cond - n_uncond)
            else:
                # Eager original UNet for batch=1 — avoids compile recompiles.
                unet = getattr(pipe.unet, "_orig_mod", pipe.unet)
                noise_pred = unet(
                    latent_in,
                    t,
                    encoder_hidden_states=text,
                    added_cond_kwargs={"image_embeds": ip},
                    return_dict=False,
                )[0]
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        t_denoise = time.perf_counter() - t_b

        t_c = time.perf_counter()
        image = self._vae_decode_image(latents)
        t_decode = time.perf_counter() - t_c
        return image, t_denoise, t_decode

    def load(self, ckpt_path: Path | None = None) -> int | str:
        self._enable_fast_backends()
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
        from transformers import CLIPVisionModelWithProjection

        t_all = time.perf_counter()
        stats_path = param_stats_path()
        if stats_path.is_file():
            import json

            self.log(f"[1/7] Loading param stats from {stats_path.name}…")
            self._param_stats = json.loads(stats_path.read_text(encoding="utf-8"))
        else:
            self.log("[1/7] No param_stats.json — using slider_to_unit normalization")

        ckpt_path = ckpt_path or best_pose_checkpoint()
        base_model = model_path()
        ip_dir = ip_adapter_dir()
        if ckpt_path is None or not ckpt_path.is_file():
            raise FileNotFoundError(f"No PoseAdapter checkpoint in {ckpt_dir()}")
        if not base_model.is_file():
            raise FileNotFoundError(f"Missing {base_model}")

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.log(f"[2/7] Loading CLIP image encoder ({dtype})…")
        t0 = time.perf_counter()
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            str(ip_dir),
            subfolder="models/image_encoder",
            torch_dtype=dtype,
        )
        self.log(f"[2/7] Image encoder ready ({time.perf_counter() - t0:.1f}s)")

        self.log(f"[3/7] Loading Anything V5 weights from {base_model.name}…")
        t0 = time.perf_counter()
        pipe = StableDiffusionPipeline.from_single_file(
            str(base_model),
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
            image_encoder=image_encoder,
        )
        self.log(f"[3/7] Base pipeline loaded ({time.perf_counter() - t0:.1f}s)")

        self.log(f"[4/7] Loading IP-Adapter ({self.ip_weight})…")
        t0 = time.perf_counter()
        pipe.load_ip_adapter(
            str(ip_dir),
            subfolder="models",
            weight_name=self.ip_weight,
            image_encoder_folder=None,
        )
        self._loaded_ip_weight = self.ip_weight
        self.log(f"[4/7] IP-Adapter attached ({time.perf_counter() - t0:.1f}s)")

        # DPM++ Multistep: good quality at 6–10 steps.
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver++",
        )
        self.log(f"[5/7] Moving pipeline to {self.device}…")
        t0 = time.perf_counter()
        pipe = pipe.to(self.device)
        if self.device == "cuda":
            try:
                pipe.unet.to(memory_format=torch.channels_last)
            except Exception:
                pass
            # Prefer fused SDPA kernels when available.
            try:
                pipe.unet.set_attn_processor(pipe.unet.attn_processors)
            except Exception:
                pass
            try:
                free, total = torch.cuda.mem_get_info()
                used = (total - free) / (1024**3)
                self.log(
                    f"[5/7] On GPU ({time.perf_counter() - t0:.1f}s) — "
                    f"VRAM ~{used:.1f} / {total / (1024**3):.1f} GB"
                )
            except Exception:
                self.log(f"[5/7] On GPU ({time.perf_counter() - t0:.1f}s)")
        else:
            self.log(f"[5/7] On CPU ({time.perf_counter() - t0:.1f}s)")
        pipe.set_ip_adapter_scale(self.ip_scale)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

        self.log("[6/7] Compiling UNet (optional)…")
        t0 = time.perf_counter()
        self._maybe_compile_unet()
        self.log(f"[6/7] UNet compile step done ({time.perf_counter() - t0:.1f}s)")

        self.log(f"[7/7] Loading PoseAdapter checkpoint {ckpt_path.name}…")
        t0 = time.perf_counter()
        step = self.load_pose_adapter(ckpt_path)
        self.log(f"[7/7] PoseAdapter ready ({time.perf_counter() - t0:.1f}s, step {step})")

        compiled = "compiled" if self._unet_compiled else "eager"
        self.log(
            f"Ready — {ckpt_path.name} (step {step}) steps={self.steps} cfg={self.cfg} "
            f"unet={compiled} total={time.perf_counter() - t_all:.1f}s"
        )
        return step

    def unload(self) -> None:
        """Drop SD / IP-Adapter / pose adapter from GPU (and CPU) memory."""
        import gc

        self._ip_cache_key = None
        self._ip_cache_embeds = None
        self._param_stats = None
        self.loaded_ckpt = None
        self._loaded_ip_weight = None
        self._unet_compiled = False

        pipe = self.pipe
        adapter = self.pose_adapter
        self.pipe = None
        self.pose_adapter = None

        for obj in (adapter, pipe):
            if obj is None:
                continue
            try:
                if hasattr(obj, "to"):
                    obj.to("cpu")
            except Exception:
                pass
            try:
                del obj
            except Exception:
                pass

        gc.collect()
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        self.log("Models unloaded — GPU cache cleared")

    def load_pose_adapter(self, ckpt_path: Path) -> int | str:
        assert self.pipe is not None
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["pose_adapter"]
        cfg = ckpt.get("config") or {}
        pose = PoseAdapter(
            num_params=NUM_PARAMS,
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            mlp_hidden=int(cfg.get("mlp_hidden", 512)),
            num_tokens_per_param=int(cfg.get("num_tokens_per_param", 1)),
            drop_prob=0.0,
        )
        pose.load_state_dict(state, strict=False)
        pose.to(device=self.device, dtype=dtype)
        pose.eval()
        self.pose_adapter = pose
        self.loaded_ckpt = ckpt_path
        return ckpt.get("step", "?")

    def set_ip_adapter(self, weight_name: str, scale: float | None = None) -> None:
        assert self.pipe is not None
        if weight_name != self._loaded_ip_weight:
            self.pipe.load_ip_adapter(
                str(ip_adapter_dir()),
                subfolder="models",
                weight_name=weight_name,
                image_encoder_folder=None,
            )
            self._loaded_ip_weight = weight_name
            self._ip_cache_key = None
            self._ip_cache_embeds = None
        if scale is not None:
            self.ip_scale = float(scale)
        self.pipe.set_ip_adapter_scale(self.ip_scale)

    def _normalize(self, pose: Sequence[float], *, mode: str = "slider_to_unit") -> list[float]:
        if mode == "stats" and self._param_stats and self._param_stats.get("normalized"):
            return normalize_params(
                pose,
                self._param_stats["param_mins"],
                self._param_stats["param_maxs"],
            )
        return slider_to_unit(pose)

    def _ref_cache_token(self, ref: Image.Image) -> tuple:
        """Stable key for a PIL image (API passes ref.copy() each generate)."""
        return (ref.size, ref.mode, hashlib.blake2b(ref.tobytes(), digest_size=16).hexdigest())

    def _cached_ip_embeds(self, ref: Image.Image, *, do_cfg: bool):
        assert self.pipe is not None
        key = (self._ref_cache_token(ref), self._loaded_ip_weight, do_cfg, self.ip_scale)
        if self._ip_cache_key == key and self._ip_cache_embeds is not None:
            return self._ip_cache_embeds
        dtype = next(self.pipe.unet.parameters()).dtype
        embeds = self.pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image=[ref],
            ip_adapter_image_embeds=None,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
        )
        embeds = [e.to(device=self.device, dtype=dtype) for e in embeds]
        self._ip_cache_key = key
        self._ip_cache_embeds = embeds
        return embeds

    def generate(
        self,
        *,
        pose: Sequence[float],
        ref: Image.Image,
        prompt: str = DEFAULT_PROMPT,
        negative: str = DEFAULT_NEGATIVE,
        seed: int = 42,
        steps: int | None = None,
        cfg: float | None = None,
        pose_drive: str | None = None,
        pose_target_rms: float | None = None,
        norm_mode: str = "slider_to_unit",
        already_normalized: bool = False,
        cfg_until: float | None = None,
    ) -> GenerateResult:
        assert self.pipe is not None and self.pose_adapter is not None
        pipe = self.pipe
        device = self.device
        dtype = next(self.pose_adapter.parameters()).dtype
        steps = int(steps if steps is not None else self.steps)
        cfg = float(cfg if cfg is not None else self.cfg)
        drive = pose_drive or self.pose_drive
        target_rms = float(
            pose_target_rms if pose_target_rms is not None else self.pose_target_rms
        )
        until = float(cfg_until if cfg_until is not None else self.cfg_until)
        until = max(0.0, min(1.0, until))
        resolution = self.resolution
        if seed < 0:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

        normed = list(pose) if already_normalized else self._normalize(pose, mode=norm_mode)
        pose_tags = pose_params_to_tags(normed)
        if drive.startswith("prompt"):
            effective = f"{prompt}, {pose_tags}" if prompt else pose_tags
        else:
            effective = prompt

        t0 = time.perf_counter()
        t_ip = t_denoise = t_decode = 0.0

        if not drive.startswith("adapter"):
            do_cfg = cfg > 1.01
            # Truncated CFG needs a manual loop; full CFG keeps the fast official pipe path.
            use_trunc = until < 0.999 and do_cfg
            with torch.inference_mode():
                if use_trunc:
                    t_a = time.perf_counter()
                    prompt_embeds, negative_embeds = self._encode_prompt_embeds(
                        effective, negative, dtype=dtype, device=device
                    )
                    ip_embeds = self._cached_ip_embeds(ref, do_cfg=True)
                    t_ip = time.perf_counter() - t_a
                    image, t_denoise, t_decode = self._denoise_truncated_cfg(
                        prompt_embeds=prompt_embeds,
                        negative_embeds=negative_embeds,
                        ip_embeds_cfg=ip_embeds,
                        seed=seed,
                        steps=steps,
                        cfg=cfg,
                        cfg_until=until,
                        resolution=resolution,
                        dtype=dtype,
                        device=device,
                    )
                else:
                    generator = torch.Generator(device="cpu").manual_seed(seed)
                    t_a = time.perf_counter()
                    ip_embeds = self._cached_ip_embeds(ref, do_cfg=do_cfg)
                    t_ip = time.perf_counter() - t_a
                    t_b = time.perf_counter()
                    result = pipe(
                        prompt=effective,
                        negative_prompt=negative or None,
                        ip_adapter_image_embeds=ip_embeds,
                        num_inference_steps=steps,
                        guidance_scale=cfg,
                        width=resolution,
                        height=resolution,
                        generator=generator,
                    )
                    t_denoise = time.perf_counter() - t_b
                    image = result.images[0]
            elapsed = time.perf_counter() - t0
            self.last_timings = {
                "total": elapsed,
                "ip": t_ip,
                "denoise": t_denoise,
                "decode": t_decode,
                "cfg_until": until,
                "fps": 1.0 / elapsed if elapsed > 0 else 0.0,
            }
            return GenerateResult(
                image=image,
                elapsed=elapsed,
                seed=seed,
                prompt=effective,
                pose_tags=pose_tags,
                timings=dict(self.last_timings),
            )

        # Adapter-delta experimental path (manual UNet loop).
        do_cfg = cfg > 1.01
        tok = pipe.tokenizer
        with torch.inference_mode():
            t_a = time.perf_counter()
            text_inputs = tok(
                [effective],
                padding="max_length",
                max_length=tok.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_emb = pipe.text_encoder(text_inputs.input_ids.to(device))[0].to(dtype=dtype)
            pose_t = torch.tensor([normed], dtype=dtype, device=device)
            neu_t = torch.tensor([rest_pose_vector()], dtype=dtype, device=device)
            pose_tokens = self.pose_adapter(pose_t, train=False)
            neu_tokens = self.pose_adapter(neu_t, train=False)
            delta = pose_tokens - neu_tokens
            rms = delta.float().pow(2).mean().sqrt().clamp_min(1e-6)
            if float(rms) < 1e-4 or target_rms <= 0.0:
                pose_tokens = torch.zeros_like(delta)
            else:
                pose_tokens = (delta / rms.to(delta.dtype)) * target_rms
            encoder_hidden = torch.cat([text_emb, pose_tokens], dim=1)

            if do_cfg:
                uncond_inputs = tok(
                    [negative or ""],
                    padding="max_length",
                    max_length=tok.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                uncond_emb = pipe.text_encoder(uncond_inputs.input_ids.to(device))[0].to(
                    dtype=dtype
                )
                uncond_hidden = torch.cat(
                    [uncond_emb, torch.zeros_like(pose_tokens)], dim=1
                )
                prompt_embeds = torch.cat([uncond_hidden, encoder_hidden], dim=0)
            else:
                prompt_embeds = encoder_hidden

            ip_embeds = self._cached_ip_embeds(ref, do_cfg=do_cfg)
            t_ip = time.perf_counter() - t_a

            generator = torch.Generator(device="cpu").manual_seed(seed)
            latents = pipe.prepare_latents(
                1,
                pipe.unet.config.in_channels,
                resolution,
                resolution,
                dtype,
                device,
                generator,
            )
            pipe.scheduler.set_timesteps(steps, device=device)
            t_b = time.perf_counter()
            for t in pipe.scheduler.timesteps:
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
                latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            t_denoise = time.perf_counter() - t_b

            t_c = time.perf_counter()
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
            image = Image.fromarray(arr)
            t_decode = time.perf_counter() - t_c

        elapsed = time.perf_counter() - t0
        self.last_timings = {
            "total": elapsed,
            "ip": t_ip,
            "denoise": t_denoise,
            "decode": t_decode,
            "fps": 1.0 / elapsed if elapsed > 0 else 0.0,
        }
        return GenerateResult(
            image=image,
            elapsed=elapsed,
            seed=seed,
            prompt=effective,
            pose_tags=pose_tags,
            timings=dict(self.last_timings),
        )

    def warmup(self, ref: Image.Image, pose: Sequence[float] | None = None) -> None:
        """Run a throwaway generate so CUDA/cudnn (and torch.compile) settle."""
        pose = pose or rest_pose_vector()
        # Use full step count so compiled graphs match real generates.
        steps = self.steps if self._unet_compiled else min(4, self.steps)
        if self._unet_compiled:
            self.log("Warmup (compiling UNet — first call can take a while)…")
        self.generate(pose=pose, ref=ref, seed=0, steps=steps, cfg_until=1.0)
