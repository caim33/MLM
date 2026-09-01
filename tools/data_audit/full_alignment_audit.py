#!/usr/bin/env python3
"""Exhaustive filename/key alignment checks for the canonical datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root

    # HumanML3D has original and mirrored M-prefixed rows.
    human_motion = {p.stem for p in (root / "humanml3d/motion").glob("*.npy")}
    human_caption = {p.stem for p in (root / "humanml3d/captions").glob("*.txt")}
    normalize_human = lambda value: value[1:] if value.startswith("M") else value
    human_motion_base = {normalize_human(value) for value in human_motion}
    human_caption_base = {normalize_human(value) for value in human_caption}

    # MotionX frame captions are split into nested directories.
    motionx_motion = {p.stem for p in (root / "motionx/motion").glob("*.npy")}
    motionx_paths: defaultdict[str, list[str]] = defaultdict(list)
    motionx_frame_root = root / "motionx/captions/frame"
    for path in motionx_frame_root.rglob("*.txt"):
        motionx_paths[path.stem].append(str(path.relative_to(motionx_frame_root)))
    motionx_caption = set(motionx_paths)

    # Main MotionX media uses underscores in video names and compact IDs in NPYs.
    normalize_media = lambda value: value.replace("_", "")
    motionx_video_paths: defaultdict[str, list[str]] = defaultdict(list)
    motionx_video_root = root / "motionx/videos"
    for path in motionx_video_root.rglob("*.mp4"):
        motionx_video_paths[normalize_media(path.stem)].append(
            str(path.relative_to(motionx_video_root))
        )
    motionx_video = set(motionx_video_paths)

    # SONIC caption keys live in the `filename` field.
    sonic_motion = {p.stem for p in (root / "sonic/motion").glob("*.npy")}
    sonic_caption: list[str] = []
    parse_failures: list[str] = []
    for path in sorted((root / "sonic/captions").glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    sonic_caption.append(str(row["filename"]))
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    parse_failures.append(f"{path.name}:{line_number}:{type(exc).__name__}")
    sonic_caption_set = set(sonic_caption)
    sonic_orphans = sorted(sonic_caption_set - sonic_motion)
    sonic_twins = {}
    for value in sonic_orphans:
        twin = value[:-2] if value.endswith("_M") else value + "_M"
        sonic_twins[value] = {"twin": twin, "twin_motion_exists": twin in sonic_motion}

    result = {
        "schema_version": 1,
        "root": str(root),
        "humanml3d": {
            "motion_files": len(human_motion),
            "caption_files": len(human_caption),
            "caption_without_exact_motion": sorted(human_caption - human_motion),
            "motion_without_exact_caption": sorted(human_motion - human_caption),
            "base_ids": {
                "motion": len(human_motion_base),
                "caption": len(human_caption_base),
                "caption_without_motion": sorted(human_caption_base - human_motion_base),
                "motion_without_caption": sorted(human_motion_base - human_caption_base),
            },
            "mirrored_motion_base_count": sum(
                count > 1 for count in Counter(map(normalize_human, human_motion)).values()
            ),
            "mirrored_caption_base_count": sum(
                count > 1 for count in Counter(map(normalize_human, human_caption)).values()
            ),
        },
        "motionx_frame": {
            "motion_files": len(motionx_motion),
            "caption_files": sum(map(len, motionx_paths.values())),
            "unique_caption_stems": len(motionx_caption),
            "duplicate_caption_stems": {
                key: values for key, values in sorted(motionx_paths.items()) if len(values) > 1
            },
            "caption_without_motion": sorted(motionx_caption - motionx_motion),
            "motion_without_caption": sorted(motionx_motion - motionx_caption),
            "orphan_caption_paths": {
                key: motionx_paths[key] for key in sorted(motionx_caption - motionx_motion)
            },
        },
        "motionx_video": {
            "motion_files": len(motionx_motion),
            "video_files": sum(map(len, motionx_video_paths.values())),
            "unique_normalized_video_ids": len(motionx_video),
            "duplicate_normalized_video_ids": {
                key: values
                for key, values in sorted(motionx_video_paths.items())
                if len(values) > 1
            },
            "video_without_motion": sorted(motionx_video - motionx_motion),
            "motion_without_video": sorted(motionx_motion - motionx_video),
        },
        "sonic": {
            "motion_files": len(sonic_motion),
            "caption_rows": len(sonic_caption),
            "unique_caption_filenames": len(sonic_caption_set),
            "duplicate_caption_filenames": len(sonic_caption) - len(sonic_caption_set),
            "parse_failures": parse_failures,
            "caption_without_motion": sonic_orphans,
            "motion_without_caption": sorted(sonic_motion - sonic_caption_set),
            "orphan_twin_analysis": sonic_twins,
        },
        "qwen_media": qwen_media_alignment(root, normalize_media),
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


def qwen_media_alignment(root: Path, normalize_media) -> dict[str, object]:
    media_root = root / "qwen_qa/media"
    motionx = media_root / "motionx_374"
    motionx_sets = {
        "motion": {normalize_media(p.stem) for p in (motionx / "motion").glob("*.npy")},
        "video": {normalize_media(p.stem) for p in (motionx / "videos").glob("*.mp4")},
        "qa": {normalize_media(p.stem) for p in (motionx / "qa").rglob("*.json")},
    }
    generated = media_root / "generated_success_assets"
    generated_sets = {
        "motion": {
            normalize_media(p.stem) for p in (generated / "motions").glob("*.npy")
        },
        "video": {
            normalize_media(p.stem) for p in (generated / "videos").glob("*.mp4")
        },
    }
    return {
        "motionx_374": {
            "motion_files": len(motionx_sets["motion"]),
            "video_files": len(motionx_sets["video"]),
            "qa_files": len(motionx_sets["qa"]),
            "video_without_motion": sorted(motionx_sets["video"] - motionx_sets["motion"]),
            "motion_without_video": sorted(motionx_sets["motion"] - motionx_sets["video"]),
            "qa_without_video": sorted(motionx_sets["qa"] - motionx_sets["video"]),
            "video_without_qa": sorted(motionx_sets["video"] - motionx_sets["qa"]),
        },
        "generated_success_assets": {
            "motion_files": len(generated_sets["motion"]),
            "video_files": len(generated_sets["video"]),
            "video_without_motion": sorted(
                generated_sets["video"] - generated_sets["motion"]
            ),
            "motion_without_video": sorted(
                generated_sets["motion"] - generated_sets["video"]
            ),
        },
    }


if __name__ == "__main__":
    main()
