from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path

DATASET_CONFIG = {
    "humanml3d": {"fps": 20, "motion": "humanml3d/motion"},
    "sonic": {"fps": 30, "motion": "sonic/motion"},
    "motionx": {"fps": 30, "motion": "motionx/motion"},
}

SUMMARY = {
    "generated_from": "dataset directory audit snapshot 2026-09-02",
    "datasets": {
        "humanml3d": {
            "label": "HumanML3D",
            "motion_files": 29228,
            "video_files": 0,
            "caption_files": 29232,
            "size_gb": 4.339,
            "fps": 20,
            "length": "median 149 · p95 199 · range 1–468 frames",
            "total_frames": 4115010,
            "duration_hours": 57.15,
            "frame_stats": {"min": 1, "median": 149, "mean": 140.79, "p95": 199, "max": 468},
            "coverage": "14,614 original + 14,614 mirrored motions",
            "quality": "4 captions do not have a matching motion file.",
        },
        "sonic": {
            "label": "SONIC",
            "motion_files": 142216,
            "video_files": 0,
            "caption_files": 142220,
            "size_gb": 32.891,
            "fps": 30,
            "length": "median 170 · p95 472 · range 17–5,404 frames",
            "total_frames": 31048704,
            "duration_hours": 287.49,
            "frame_stats": {"min": 17, "median": 170, "mean": 218.32, "p95": 472, "max": 5404},
            "coverage": "Temporal event metadata in JSONL/CSV/Parquet",
            "quality": "4 caption keys map only to the opposite _M twin.",
        },
        "motionx": {
            "label": "MotionX",
            "motion_files": 64246,
            "video_files": 115990,
            "caption_files": 64219,
            "size_gb": 150.497,
            "fps": 30,
            "length": "median 299 · p95 299 · range 1–299 frames",
            "total_frames": 19130131,
            "duration_hours": 177.13,
            "frame_stats": {"min": 1, "median": 299, "mean": 297.76, "p95": 299, "max": 299},
            "coverage": "Every motion has a video; motion covers 55.39% of videos",
            "quality": "3 frame captions lack motion; 30 motions lack frame captions.",
        },
    },
    "totals": {
        "motion_files": 235690,
        "video_files": 115990,
        "size_gb": 187.727,
        "feature_dims": 263,
        "joints": 22,
    },
}


def frame_count(path: Path, size: int) -> int:
    # Files in these three canonical motion roots were written with a 128-byte
    # NumPy v1 header and float32 (T, 263) payload. Fall back to parsing only
    # unusual files instead of opening every one of the 235k arrays.
    payload = size - 128
    stride = 263 * 4
    if payload >= 0 and payload % stride == 0:
        return payload // stride
    import numpy as np

    array = np.load(path, mmap_mode="r")
    if array.ndim != 2 or array.shape[1] != 263:
        raise ValueError(f"unexpected shape {array.shape} for {path}")
    return int(array.shape[0])


def normalized_video_id(stem: str) -> str:
    return stem.replace("_", "")


def motionx_caption_path(data_root: Path, sample_id: str) -> Path | None:
    candidates = (
        data_root / "motionx/captions/frame/motion_script_only" / f"{sample_id}.txt",
        data_root / "motionx/captions/frame" / f"{sample_id}.txt",
    )
    return next((path for path in candidates if path.is_file()), None)


def build(data_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "index.sqlite"
    tmp_path = output_dir / "index.sqlite.tmp"
    if tmp_path.exists():
        tmp_path.unlink()

    started = time.time()
    connection = sqlite3.connect(tmp_path)
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        CREATE TABLE samples (
            dataset TEXT NOT NULL,
            id TEXT NOT NULL,
            motion_path TEXT NOT NULL,
            video_path TEXT,
            caption_path TEXT,
            caption_offset INTEGER,
            caption_length INTEGER,
            clip_caption_path TEXT,
            clip_caption_offset INTEGER,
            clip_caption_length INTEGER,
            cot_path TEXT,
            frames INTEGER NOT NULL,
            fps INTEGER NOT NULL,
            bytes INTEGER NOT NULL,
            PRIMARY KEY (dataset, id)
        );
        CREATE INDEX samples_dataset_id ON samples(dataset, id);
        CREATE INDEX samples_has_video ON samples(dataset, video_path);
        """
    )

    video_map: dict[str, str] = {}
    video_root = data_root / "motionx/videos"
    for current_root, _, files in os.walk(video_root):
        current = Path(current_root)
        for filename in files:
            if not filename.lower().endswith(".mp4"):
                continue
            absolute = current / filename
            relative = absolute.relative_to(data_root).as_posix()
            video_map.setdefault(normalized_video_id(absolute.stem), relative)

    insert_sql = """
        INSERT INTO samples(
            dataset, id, motion_path, video_path, caption_path,
            caption_offset, caption_length, frames, fps, bytes
        ) VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?)
    """

    for dataset, config in DATASET_CONFIG.items():
        motion_root = data_root / config["motion"]
        batch = []
        for entry in os.scandir(motion_root):
            if not entry.is_file() or not entry.name.endswith(".npy"):
                continue
            motion_path = Path(entry.path)
            sample_id = motion_path.stem
            relative_motion = motion_path.relative_to(data_root).as_posix()
            caption_path: Path | None = None
            video_path: str | None = None
            if dataset == "humanml3d":
                candidate = data_root / "humanml3d/captions" / f"{sample_id}.txt"
                caption_path = candidate if candidate.is_file() else None
            elif dataset == "motionx":
                caption_path = motionx_caption_path(data_root, sample_id)
                video_path = video_map.get(sample_id)

            file_size = entry.stat().st_size
            batch.append(
                (
                    dataset,
                    sample_id,
                    relative_motion,
                    video_path,
                    caption_path.relative_to(data_root).as_posix()
                    if caption_path
                    else None,
                    frame_count(motion_path, file_size),
                    config["fps"],
                    file_size,
                )
            )
            if len(batch) >= 1000:
                connection.executemany(insert_sql, batch)
                connection.commit()
                batch.clear()
        if batch:
            connection.executemany(insert_sql, batch)
            connection.commit()

    sonic_metadata = data_root / "sonic/captions/seed_metadata_v002_temporal_labels.jsonl"
    with sonic_metadata.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            try:
                sample_id = json.loads(line)["filename"]
            except (json.JSONDecodeError, KeyError):
                continue
            connection.execute(
                """
                UPDATE samples
                SET caption_path=?, caption_offset=?, caption_length=?
                WHERE dataset='sonic' AND id=?
                """,
                (
                    sonic_metadata.relative_to(data_root).as_posix(),
                    offset,
                    len(line),
                    sample_id,
                ),
            )
    connection.commit()

    motionx_descriptions = data_root / "motionx/captions/original/descriptions.jsonl"
    if motionx_descriptions.is_file():
        with motionx_descriptions.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                try:
                    video_name = json.loads(line)["video"]
                    sample_id = normalized_video_id(Path(video_name).stem)
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
                connection.execute(
                    """
                    UPDATE samples
                    SET clip_caption_path=?, clip_caption_offset=?, clip_caption_length=?
                    WHERE dataset='motionx' AND id=?
                    """,
                    (
                        motionx_descriptions.relative_to(data_root).as_posix(),
                        offset,
                        len(line),
                        sample_id,
                    ),
                )

    cot_run = data_root / "motionx/captions/complex/PRODUCTION/runs/motionx_deepseek_prod200_20260414"
    curated_cot_list = data_root / "motionx/captions/complex/PRODUCTION/sample_lists/motionx_triplets_49405_curated.txt"
    raw_cot_list = data_root / "motionx/captions/complex/PRODUCTION/sample_lists/motionx_parseable_49408.txt"
    cot_list = curated_cot_list if curated_cot_list.is_file() else raw_cot_list
    if cot_list.is_file():
        cot_updates = []
        for sample_id in cot_list.read_text(encoding="utf-8").splitlines():
            sample_id = sample_id.strip()
            if not sample_id:
                continue
            relative = (cot_run / sample_id / "step2_generation.json").relative_to(data_root).as_posix()
            cot_updates.append((relative, sample_id))
        connection.executemany(
            "UPDATE samples SET cot_path=? WHERE dataset='motionx' AND id=?",
            cot_updates,
        )
    connection.commit()

    counts = {
        row[0]: row[1]
        for row in connection.execute(
            "SELECT dataset, COUNT(*) FROM samples GROUP BY dataset"
        )
    }
    connection.close()

    if db_path.exists():
        db_path.unlink()
    tmp_path.replace(db_path)
    summary = dict(SUMMARY)
    summary["indexed_samples"] = counts
    summary["index_build_seconds"] = round(time.time() - started, 2)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"index": str(db_path), "counts": counts, "seconds": summary["index_build_seconds"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("MOTION_DATA_ROOT",
        Path(__file__).resolve().parent.parents[2] / "dataset")),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    build(args.data_root.resolve(),
          (args.output_dir or args.data_root / "data_page" / "data").resolve())


if __name__ == "__main__":
    main()
