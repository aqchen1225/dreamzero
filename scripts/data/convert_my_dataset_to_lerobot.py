#!/usr/bin/env python3
"""Convert custom skill-decomposition dataset (my_dataset format) to LeRobot v2 format.

The generated dataset can then be converted to DreamZero/GEAR metadata by running:

python scripts/data/convert_lerobot_to_gear.py \
  --dataset-path <output_dir> \
  --embodiment-tag dream_skill \
  --state-keys '{"prev_skill_type_id": [0, 1], "prev_obj_id": [1, 2], "prev_dst_id": [2, 3]}' \
  --action-keys '{"skill_type_id": [0, 1], "obj_id": [1, 2], "dst_id": [2, 3]}' \
  --task-key annotation.task
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

SCRIPT_TYPES = [
    "insert_multi_scripts",
    "insert_scripts",
    "navigate_scripts",
    "pick_and_place_scripts",
    "pour_scripts",
]


@dataclass
class Episode:
    scene_id: int
    script_type: str
    script_path: Path
    image_dir: Path


def parse_skill_line(line: str) -> tuple[str, list[str]]:
    m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*$", line)
    if m is None:
        line = line.strip()
        if not line:
            return "NOP", []
        return line, []
    skill_type = m.group(1)
    args_str = m.group(2).strip()
    if not args_str:
        return skill_type, []
    args = [x.strip() for x in args_str.split(",")]
    return skill_type, args


def sample_indices(num_frames: int, n_per_step: int) -> list[int]:
    if num_frames <= 0:
        return [0] * n_per_step
    if n_per_step <= 1:
        return [0]
    if num_frames >= n_per_step:
        return np.linspace(0, num_frames - 1, n_per_step).round().astype(int).tolist()
    out = list(range(num_frames))
    out.extend([num_frames - 1] * (n_per_step - num_frames))
    return out


def gather_episodes(texts_root: Path, imgs_root: Path, max_scenes: int | None) -> list[Episode]:
    episodes: list[Episode] = []
    scene_dirs = sorted([p for p in texts_root.glob("scene_*") if p.is_dir()])
    if max_scenes is not None:
        scene_dirs = scene_dirs[:max_scenes]

    for scene_dir in scene_dirs:
        scene_match = re.match(r"scene_(\d+)", scene_dir.name)
        if scene_match is None:
            continue
        scene_id = int(scene_match.group(1))
        for script_type in SCRIPT_TYPES:
            script_type_dir = scene_dir / script_type
            if not script_type_dir.exists():
                continue
            for ep_dir in sorted([d for d in script_type_dir.glob("ep_*") if d.is_dir()]):
                for script_file in sorted(ep_dir.glob("*.txt")):
                    image_dir = imgs_root / f"scene_{scene_id}" / script_type / ep_dir.name / script_file.stem
                    if not image_dir.exists():
                        continue
                    episodes.append(Episode(scene_id, script_type, script_file, image_dir))
    return episodes


def build_vocab(episodes: list[Episode]) -> tuple[dict[str, int], dict[str, int]]:
    skill_types = {"<PAD>": 0, "<UNK>": 1}
    entities = {"<PAD>": 0, "<UNK>": 1}
    for ep in episodes:
        lines = [ln.strip() for ln in ep.script_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for ln in lines:
            skill, args = parse_skill_line(ln)
            skill_types.setdefault(skill, len(skill_types))
            for arg in args:
                entities.setdefault(arg, len(entities))
    return skill_types, entities


def get_scene_text(scene_text_dir: Path, scene_id: int) -> str:
    scene_dir = scene_text_dir / f"scene_{scene_id}"
    scene_files = sorted(scene_dir.glob("task_*.txt"))
    if not scene_files:
        return ""
    return scene_files[0].read_text(encoding="utf-8").strip()


def get_instruction_from_filename(path: Path) -> str:
    # e.g. 1766471704_0_机器人，帮我把水果盘拿到床上.txt
    parts = path.stem.split("_")
    if len(parts) >= 3:
        return parts[-1]
    return path.stem


def load_step_frames(step_dir: Path, camera: str) -> list[np.ndarray]:
    cam_dir = step_dir / camera
    if not cam_dir.exists():
        return []
    pngs = sorted(cam_dir.glob("*.png"))
    frames: list[np.ndarray] = []
    for p in pngs:
        with Image.open(p) as img:
            frames.append(np.asarray(img.convert("RGB")))
    return frames


def write_episode(
    episode_index: int,
    episode: Episode,
    output_dir: Path,
    fps: int,
    n_per_step: int,
    camera: str,
    skill_vocab: dict[str, int],
    entity_vocab: dict[str, int],
    include_scene_text: bool,
    texts_root: Path,
) -> tuple[int, tuple[int, int]]:
    steps = [ln.strip() for ln in episode.script_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not steps:
        return 0, (0, 0)

    scene_text = get_scene_text(texts_root, episode.scene_id) if include_scene_text else ""
    instruction = get_instruction_from_filename(episode.script_path)
    if scene_text:
        task_text = f"任务: {instruction}\n场景: {scene_text}"
    else:
        task_text = instruction

    video_frames: list[np.ndarray] = []
    rows: list[dict] = []

    prev_triplet = (0, 0, 0)
    for step_id, skill_line in enumerate(steps):
        step_name = f"step_{step_id:02d}_"
        candidates = sorted([d for d in episode.image_dir.glob(f"{step_name}*") if d.is_dir()])
        if not candidates:
            continue
        frames = load_step_frames(candidates[0], camera)
        if not frames:
            continue

        sel = sample_indices(len(frames), n_per_step)
        skill_type, args = parse_skill_line(skill_line)
        obj = args[0] if len(args) >= 1 else "<PAD>"
        dst = args[1] if len(args) >= 2 else "<PAD>"
        curr_triplet = (
            skill_vocab.get(skill_type, skill_vocab["<UNK>"]),
            entity_vocab.get(obj, entity_vocab["<UNK>"]),
            entity_vocab.get(dst, entity_vocab["<UNK>"]),
        )

        for frame_i in sel:
            frame = frames[min(max(frame_i, 0), len(frames) - 1)]
            video_frames.append(frame)
            t = (len(video_frames) - 1) / fps
            rows.append(
                {
                    "timestamp": float(t),
                    "frame_index": int(len(video_frames) - 1),
                    "episode_index": int(episode_index),
                    "observation.state": np.array(prev_triplet, dtype=np.float32),
                    "action": np.array(curr_triplet, dtype=np.float32),
                    "annotation.task": task_text,
                }
            )
        prev_triplet = curr_triplet

    if not rows:
        return 0, (0, 0)

    chunk_index = episode_index // 1000
    parquet_path = output_dir / f"data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet"
    video_path = output_dir / (
        f"videos/chunk-{chunk_index:03d}/observation.images.{camera}/episode_{episode_index:06d}.mp4"
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    imageio.mimsave(video_path.as_posix(), video_frames, fps=fps)
    pd.DataFrame(rows).to_parquet(parquet_path)

    h, w = video_frames[0].shape[:2]
    return len(rows), (h, w)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert my_dataset skill data to LeRobot v2 format")
    parser.add_argument("--input-root", type=Path, default=Path("my_dataset"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--camera", type=str, default="camA", choices=["camA", "camB"])
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--samples-per-step", type=int, default=1)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--include-scene-text", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.force:
        raise ValueError(f"Output directory {args.output_root} is not empty. Use --force to overwrite.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "meta").mkdir(exist_ok=True)

    texts_root = args.input_root / "texts"
    imgs_root = args.input_root / "imgs"
    episodes = gather_episodes(texts_root, imgs_root, args.max_scenes)
    if not episodes:
        raise ValueError("No episodes found. Check input directory layout.")

    skill_vocab, entity_vocab = build_vocab(episodes)
    (args.output_root / "meta" / "skill_vocab.json").write_text(
        json.dumps(skill_vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_root / "meta" / "entity_vocab.json").write_text(
        json.dumps(entity_vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total_written = 0
    image_hw = (0, 0)
    kept_episodes = 0
    for ep_idx, ep in enumerate(tqdm(episodes, desc="Converting episodes")):
        count, hw = write_episode(
            episode_index=ep_idx,
            episode=ep,
            output_dir=args.output_root,
            fps=args.fps,
            n_per_step=args.samples_per_step,
            camera=args.camera,
            skill_vocab=skill_vocab,
            entity_vocab=entity_vocab,
            include_scene_text=args.include_scene_text,
            texts_root=texts_root,
        )
        if count > 0:
            kept_episodes += 1
            total_written += count
            image_hw = hw

    if kept_episodes == 0:
        raise ValueError("No valid episodes were converted. Ensure camera folders contain PNG frames.")

    h, w = image_hw
    info = {
        "codebase_version": "v2.0",
        "robot_type": "dream_skill",
        "total_episodes": kept_episodes,
        "total_frames": total_written,
        "total_tasks": 0,
        "total_videos": kept_episodes,
        "total_chunks": max(1, (kept_episodes + 999) // 1000),
        "chunks_size": 1000,
        "fps": args.fps,
        "splits": {"train": f"0:{kept_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/observation.images.{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "timestamp": {"dtype": "float32", "shape": [1], "names": ["timestamp"]},
            "frame_index": {"dtype": "int64", "shape": [1], "names": ["frame_index"]},
            "episode_index": {"dtype": "int64", "shape": [1], "names": ["episode_index"]},
            "observation.state": {"dtype": "float32", "shape": [3], "names": ["state"]},
            "action": {"dtype": "float32", "shape": [3], "names": ["action"]},
            "annotation.task": {"dtype": "string", "shape": [1], "names": None},
            f"observation.images.{args.camera}": {
                "dtype": "video",
                "shape": [h, w, 3],
                "names": ["height", "width", "channel"],
                "video_info": {"video.fps": args.fps, "video.channels": 3},
            },
        },
    }
    (args.output_root / "meta" / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Done.")
    print(f"episodes={kept_episodes}, frames={total_written}, camera={args.camera}, fps={args.fps}")
    print("Next: run scripts/data/convert_lerobot_to_gear.py on this output directory.")


if __name__ == "__main__":
    main()
