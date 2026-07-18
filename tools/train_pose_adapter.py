"""Train PoseAdapter on frozen Anything V5 + frozen IP-Adapter.

Only PoseAdapter weights are updated. Identity comes from IP-Adapter (reference
image); pose comes from the 16-param vector via concatenated cross-attn tokens.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.pose_adapter import PoseAdapter  # noqa: E402
from utils.params import NUM_PARAMS  # noqa: E402


@dataclass
class TrainConfig:
    checkpoint: str
    ip_adapter_dir: str
    ip_adapter_weight: str
    workdir: str
    train_index: str
    val_index: str
    resolution: int = 512
    batch_size: int = 4
    grad_accum: int = 2
    lr: float = 1.0e-4
    max_steps: int = 40000
    warmup_steps: int = 500
    save_every: int = 2000
    sample_every: int = 1000
    log_every: int = 50
    num_workers: int = 4
    seed: int = 42
    drop_pose_prob: float = 0.1
    ip_scale: float = 0.7
    prompt: str = "masterpiece, best quality, 1girl, solo, anime style"
    negative_prompt: str = (
        "lowres, bad anatomy, bad hands, worst quality, low quality, blurry"
    )
    mixed_precision: str = "fp16"
    gradient_checkpointing: bool = True
    mlp_hidden: int = 512
    num_tokens_per_param: int = 1


def load_config(path: Path) -> TrainConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    # Resolve relative paths against pack root.
    for key in (
        "checkpoint",
        "ip_adapter_dir",
        "workdir",
        "train_index",
        "val_index",
    ):
        if key in raw and raw[key] and not Path(raw[key]).is_absolute():
            raw[key] = str((ROOT / raw[key]).resolve())
    fields = {f.name for f in TrainConfig.__dataclass_fields__.values()}
    return TrainConfig(**{k: v for k, v in raw.items() if k in fields})


class PoseIndexDataset(Dataset):
    def __init__(self, index_path: Path, resolution: int = 512) -> None:
        self.rows: list[dict] = []
        with index_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        if not self.rows:
            raise RuntimeError(f"Empty index: {index_path}")
        self.resolution = int(resolution)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_rgb(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        size = self.resolution
        scale = size / min(w, h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        left = (nw - size) // 2
        top = (nh - size) // 2
        return img.crop((left, top, left + size, top + size))

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image = self._load_rgb(row["image"])
        ref = self._load_rgb(row["ref_image"])
        params = torch.tensor(row["params"], dtype=torch.float32)
        if params.numel() != NUM_PARAMS:
            raise ValueError(f"Bad params len for {row['sample_id']}")
        # To tensor in [-1, 1]
        arr = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        pixel = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return {
            "pixel_values": pixel,
            "ref_image": ref,  # PIL — collate keeps list
            "params": params,
            "sample_id": row["sample_id"],
        }


def _collate(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch], dim=0),
        "params": torch.stack([b["params"] for b in batch], dim=0),
        "ref_images": [b["ref_image"] for b in batch],
        "sample_ids": [b["sample_id"] for b in batch],
    }


def _encode_prompt(pipe, prompt: str, negative: str, bsz: int, device, dtype):
    tok = pipe.tokenizer
    text_enc = pipe.text_encoder
    text_inputs = tok(
        [prompt] * bsz,
        padding="max_length",
        max_length=tok.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond = tok(
        [negative] * bsz,
        padding="max_length",
        max_length=tok.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_emb = text_enc(text_inputs.input_ids.to(device))[0].to(dtype=dtype)
        # Training uses conditional branch only (CFG handled at inference).
    return text_emb


def _prepare_ip_embeds(pipe, ref_images: list[Image.Image], device, dtype):
    """Encode reference images to IP-Adapter image embeds (list for SD1.5)."""
    # diffusers prepare_ip_adapter_image_embeds handles projection layers.
    embeds = pipe.prepare_ip_adapter_image_embeds(
        ip_adapter_image=ref_images,
        ip_adapter_image_embeds=None,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    # Returns list of tensors; cast dtype.
    out = []
    for e in embeds:
        out.append(e.to(device=device, dtype=dtype))
    return out


@torch.no_grad()
def _save_sample_grid(
    pipe,
    pose_adapter: PoseAdapter,
    val_loader: DataLoader,
    step: int,
    out_dir: Path,
    cfg: TrainConfig,
    device,
    dtype,
) -> None:
    pose_adapter.eval()
    batch = next(iter(val_loader))
    n = min(4, batch["pixel_values"].shape[0])
    params = batch["params"][:n].to(device=device, dtype=dtype)
    refs = batch["ref_images"][:n]
    text_emb = _encode_prompt(pipe, cfg.prompt, cfg.negative_prompt, n, device, dtype)
    pose_tokens = pose_adapter(params, train=False)
    encoder_hidden = torch.cat([text_emb, pose_tokens], dim=1)
    ip_embeds = _prepare_ip_embeds(pipe, refs, device, dtype)

    latents = torch.randn(
        n,
        pipe.unet.config.in_channels,
        cfg.resolution // 8,
        cfg.resolution // 8,
        device=device,
        dtype=dtype,
    )
    pipe.scheduler.set_timesteps(20, device=device)
    for t in pipe.scheduler.timesteps:
        noise_pred = pipe.unet(
            latents,
            t,
            encoder_hidden_states=encoder_hidden,
            added_cond_kwargs={"image_embeds": ip_embeds},
            return_dict=False,
        )[0]
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    latents = latents / pipe.vae.config.scaling_factor
    images = pipe.vae.decode(latents.to(dtype=pipe.vae.dtype), return_dict=False)[0]
    images = (images.float().clamp(-1, 1) + 1) / 2
    images = (images * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

    out_dir.mkdir(parents=True, exist_ok=True)
    # Side-by-side: ref | generated | target
    targets = batch["pixel_values"][:n]
    targets = ((targets + 1) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
    strips = []
    for i in range(n):
        ref = refs[i].resize((cfg.resolution, cfg.resolution))
        gen = Image.fromarray(images[i])
        tgt = Image.fromarray(targets[i])
        strip = Image.new("RGB", (cfg.resolution * 3, cfg.resolution))
        strip.paste(ref, (0, 0))
        strip.paste(gen, (cfg.resolution, 0))
        strip.paste(tgt, (cfg.resolution * 2, 0))
        strips.append(strip)
    grid = Image.new("RGB", (cfg.resolution * 3, cfg.resolution * n))
    for i, s in enumerate(strips):
        grid.paste(s, (0, i * cfg.resolution))
    path = out_dir / f"sample_step_{step:06d}.png"
    grid.save(path)
    print(f"Saved sample grid: {path}")
    pose_adapter.train()


def train(cfg: TrainConfig) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.grad_accum,
        mixed_precision=cfg.mixed_precision,
    )
    set_seed(cfg.seed)
    workdir = Path(cfg.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = workdir / "checkpoints"
    sample_dir = workdir / "samples"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        print(f"Loading SD from {cfg.checkpoint}")

    from diffusers import DDPMScheduler, StableDiffusionPipeline
    from transformers import CLIPVisionModelWithProjection

    weight_dtype = torch.float16 if cfg.mixed_precision == "fp16" else torch.float32
    if cfg.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        cfg.ip_adapter_dir,
        subfolder="models/image_encoder",
        torch_dtype=weight_dtype,
    )
    pipe = StableDiffusionPipeline.from_single_file(
        cfg.checkpoint,
        torch_dtype=weight_dtype,
        safety_checker=None,
        requires_safety_checker=False,
        image_encoder=image_encoder,
    )
    pipe.load_ip_adapter(
        cfg.ip_adapter_dir,
        subfolder="models",
        weight_name=cfg.ip_adapter_weight,
        image_encoder_folder=None,
    )
    pipe.set_ip_adapter_scale(cfg.ip_scale)
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Freeze everything except PoseAdapter.
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.image_encoder.requires_grad_(False)
    for name, param in pipe.unet.named_parameters():
        if "attn2.processor" in name or "to_k_ip" in name or "to_v_ip" in name:
            param.requires_grad_(False)

    if cfg.gradient_checkpointing:
        pipe.unet.enable_gradient_checkpointing()

    cross_dim = pipe.unet.config.cross_attention_dim
    pose_adapter = PoseAdapter(
        num_params=NUM_PARAMS,
        cross_attention_dim=cross_dim,
        mlp_hidden=cfg.mlp_hidden,
        num_tokens_per_param=cfg.num_tokens_per_param,
        drop_prob=cfg.drop_pose_prob,
    )

    trainable = sum(p.numel() for p in pose_adapter.parameters() if p.requires_grad)
    if accelerator.is_main_process:
        print(f"PoseAdapter trainable params: {trainable / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        pose_adapter.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return float(step) / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_ds = PoseIndexDataset(Path(cfg.train_index), resolution=cfg.resolution)
    val_ds = PoseIndexDataset(Path(cfg.val_index), resolution=cfg.resolution)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=_collate,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(4, cfg.batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=_collate,
    )

    pipe.vae.to(accelerator.device, dtype=weight_dtype)
    pipe.text_encoder.to(accelerator.device, dtype=weight_dtype)
    pipe.unet.to(accelerator.device, dtype=weight_dtype)
    pipe.image_encoder.to(accelerator.device, dtype=weight_dtype)
    # Move IP projection layers if present.
    if hasattr(pipe.unet, "encoder_hid_proj") and pipe.unet.encoder_hid_proj is not None:
        pipe.unet.encoder_hid_proj.to(accelerator.device, dtype=weight_dtype)

    pose_adapter, optimizer, train_loader, scheduler = accelerator.prepare(
        pose_adapter, optimizer, train_loader, scheduler
    )

    noise_scheduler = pipe.scheduler
    global_step = 0
    pose_adapter.train()
    t0 = time.perf_counter()

    if accelerator.is_main_process:
        (workdir / "train_config.json").write_text(
            json.dumps(cfg.__dict__, indent=2), encoding="utf-8"
        )

    progress = tqdm(
        total=cfg.max_steps,
        disable=not accelerator.is_main_process,
        desc="train",
    )
    while global_step < cfg.max_steps:
        for batch in train_loader:
            with accelerator.accumulate(pose_adapter):
                pixel = batch["pixel_values"].to(
                    device=accelerator.device, dtype=weight_dtype
                )
                params = batch["params"].to(
                    device=accelerator.device, dtype=weight_dtype
                )
                refs = batch["ref_images"]
                bsz = pixel.shape[0]

                with torch.no_grad():
                    latents = pipe.vae.encode(pixel).latent_dist.sample()
                    latents = latents * pipe.vae.config.scaling_factor
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                        dtype=torch.long,
                    )
                    noisy = noise_scheduler.add_noise(latents, noise, timesteps)
                    text_emb = _encode_prompt(
                        pipe,
                        cfg.prompt,
                        cfg.negative_prompt,
                        bsz,
                        accelerator.device,
                        weight_dtype,
                    )
                    ip_embeds = _prepare_ip_embeds(
                        pipe, refs, accelerator.device, weight_dtype
                    )

                pose_tokens = pose_adapter(params, train=True)
                encoder_hidden = torch.cat([text_emb, pose_tokens], dim=1)

                model_pred = pipe.unet(
                    noisy,
                    timesteps,
                    encoder_hidden_states=encoder_hidden,
                    added_cond_kwargs={"image_embeds": ip_embeds},
                    return_dict=False,
                )[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(pose_adapter.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                if accelerator.is_main_process and global_step % cfg.log_every == 0:
                    elapsed = time.perf_counter() - t0
                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"step={global_step} loss={loss.item():.4f} "
                        f"lr={lr:.2e} elapsed={elapsed:.0f}s"
                    )
                if (
                    accelerator.is_main_process
                    and global_step % cfg.sample_every == 0
                ):
                    unwrapped = accelerator.unwrap_model(pose_adapter)
                    _save_sample_grid(
                        pipe,
                        unwrapped,
                        val_loader,
                        global_step,
                        sample_dir,
                        cfg,
                        accelerator.device,
                        weight_dtype,
                    )
                if (
                    accelerator.is_main_process
                    and global_step % cfg.save_every == 0
                ):
                    unwrapped = accelerator.unwrap_model(pose_adapter)
                    path = ckpt_dir / f"pose_adapter_step_{global_step:06d}.pt"
                    torch.save(
                        {
                            "step": global_step,
                            "pose_adapter": unwrapped.state_dict(),
                            "config": cfg.__dict__,
                        },
                        path,
                    )
                    # Also write "latest"
                    torch.save(
                        {
                            "step": global_step,
                            "pose_adapter": unwrapped.state_dict(),
                            "config": cfg.__dict__,
                        },
                        ckpt_dir / "pose_adapter_latest.pt",
                    )
                    print(f"Saved {path}")
                if global_step >= cfg.max_steps:
                    break
        if global_step >= cfg.max_steps:
            break

    progress.close()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(pose_adapter)
        final = ckpt_dir / "pose_adapter_final.pt"
        torch.save(
            {
                "step": global_step,
                "pose_adapter": unwrapped.state_dict(),
                "config": cfg.__dict__,
            },
            final,
        )
        print(f"Training done. Final checkpoint: {final}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "pose_adapter.yaml",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr

    for required in (cfg.checkpoint, cfg.train_index, cfg.val_index):
        if not Path(required).exists():
            raise FileNotFoundError(
                f"Missing {required}. Run scripts/build_dataset.py first "
                f"and ensure models are present."
            )
    train(cfg)


if __name__ == "__main__":
    main()
