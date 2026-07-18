"""Benchmark PoseEngine: 1 frame, then 5 consecutive frames with moving poses."""

from __future__ import annotations

import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image

from backend.engine import (
    DEFAULT_CFG,
    DEFAULT_REF,
    DEFAULT_STEPS,
    OUTPUT_DIR,
    PoseEngine,
    best_pose_checkpoint,
    rest_pose_vector,
)
from backend.utils.params import PARAM_NAMES


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


MOTION_POSES = [
    make_pose(ParamAngleY=-20.0, ParamMouthOpenY=0.0),
    make_pose(ParamAngleY=-10.0, ParamMouthOpenY=0.35, MouthSmile=0.4),
    make_pose(ParamAngleY=0.0, ParamMouthOpenY=0.8, ParamEyeBallX=0.3),
    make_pose(ParamAngleY=12.0, ParamMouthOpenY=0.2, ParamAngleX=8.0),
    make_pose(ParamAngleY=22.0, ParamMouthOpenY=0.0, ParamEyeLOpen=0.15, ParamEyeROpen=0.15),
]


def fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.0f}ms"


def main() -> None:
    ckpt = best_pose_checkpoint()
    if ckpt is None:
        raise SystemExit("No PoseAdapter checkpoint found")
    if not DEFAULT_REF.is_file():
        raise SystemExit(f"Missing reference image: {DEFAULT_REF}")

    ref = Image.open(DEFAULT_REF).convert("RGB")
    print(f"checkpoint={ckpt.name}")
    print(f"ref={DEFAULT_REF.name} size={ref.size}")
    print(f"steps={DEFAULT_STEPS} cfg={DEFAULT_CFG}")

    eng = PoseEngine(steps=DEFAULT_STEPS, cfg=DEFAULT_CFG, log=print)
    t_load = time.perf_counter()
    step = eng.load(ckpt)
    print(f"load_time={time.perf_counter() - t_load:.2f}s loaded_step={step}")

    print("\n=== warmup ===")
    eng.warmup(ref, rest_pose_vector())

    print("\n=== 1 frame ===")
    r1 = eng.generate(pose=MOTION_POSES[0], ref=ref, seed=42)
    print(f"  total={fmt_ms(r1.elapsed)} fps={r1.timings['fps']:.2f}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out1 = OUTPUT_DIR / "bench_1frame.png"
    r1.image.save(out1)
    print(f"  saved {out1}")

    print("\n=== 5 consecutive frames ===")
    times: list[float] = []
    for i, pose in enumerate(MOTION_POSES):
        ri = eng.generate(pose=pose, ref=ref, seed=42 + i)
        times.append(ri.elapsed)
        out = OUTPUT_DIR / f"bench_motion_{i + 1}.png"
        ri.image.save(out)
        print(f"  frame {i + 1}/5: {fmt_ms(ri.elapsed)} ({ri.timings['fps']:.2f} FPS) -> {out.name}")

    total = sum(times)
    print("\n=== summary ===")
    print(f"  1-frame:       {fmt_ms(r1.elapsed)} ({r1.timings['fps']:.2f} FPS)")
    print(f"  5-frame total: {fmt_ms(total)}")
    print(f"  5-frame mean:  {fmt_ms(st.mean(times))} ({1.0 / st.mean(times):.2f} FPS)")


if __name__ == "__main__":
    main()
