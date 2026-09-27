from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

APP_BOOT_ROOT = Path(__file__).resolve().parent
LOCAL_DEPS = Path(os.environ.get("MOTION_DATA_PAGE_DEPS", APP_BOOT_ROOT / ".deps"))
if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))

from aiohttp import web

from motion import KINEMATIC_CHAINS, pack_motion


APP_ROOT = APP_BOOT_ROOT
DATA_ROOT = Path(
    os.environ.get("MOTION_DATA_ROOT", APP_ROOT.parents[2] / "dataset")
).resolve()
INDEX_DIR = Path(
    os.environ.get("MOTION_DATA_INDEX_DIR", DATA_ROOT / "data_page" / "data")
).resolve()
STATIC_ROOT = APP_ROOT / "static"
INDEX_PATH = INDEX_DIR / "index.sqlite"
SUMMARY_PATH = INDEX_DIR / "summary.json"
TASK_STATS_PATH = INDEX_DIR / "task_stats.json"
SMPLH_MODEL_PATH = (
    Path(os.environ["MOTION_SMPLH_MODEL"])
    if os.environ.get("MOTION_SMPLH_MODEL")
    else None
)
DATASETS = {"humanml3d", "sonic", "motionx"}
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def require_dataset(value: str) -> str:
    if value not in DATASETS:
        raise web.HTTPBadRequest(text="unknown dataset")
    return value


def require_id(value: str) -> str:
    if not SAFE_ID.fullmatch(value):
        raise web.HTTPBadRequest(text="invalid sample id")
    return value


def resolve_data_path(relative: str) -> Path:
    candidate = (DATA_ROOT / relative).resolve()
    try:
        candidate.relative_to(DATA_ROOT)
    except ValueError as exc:
        raise web.HTTPForbidden(text="path outside dataset root") from exc
    if not candidate.is_file():
        raise web.HTTPNotFound(text="data file is missing")
    return candidate


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(INDEX_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def row_for(dataset: str, sample_id: str) -> sqlite3.Row:
    with connect() as connection:
        row = connection.execute(
            "SELECT * FROM samples WHERE dataset=? AND id=?", (dataset, sample_id)
        ).fetchone()
    if row is None:
        raise web.HTTPNotFound(text="sample not found")
    return row


def caption_for(row: sqlite3.Row):
    relative = row["caption_path"]
    if not relative:
        return None
    path = resolve_data_path(relative)
    if row["caption_offset"] is not None:
        with path.open("rb") as handle:
            handle.seek(row["caption_offset"])
            payload = json.loads(handle.read(row["caption_length"]))
        return {
            "kind": "events",
            "events": payload.get("events", []),
            "count": payload.get("num_events", 0),
        }
    return line_caption_for(row, offset=0, limit=40)


def line_caption_for(
    row: sqlite3.Row, offset: int = 0, limit: int = 40, query: str = ""
):
    relative = row["caption_path"]
    if not relative:
        return None
    path = resolve_data_path(relative)
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if row["dataset"] == "humanml3d":
        # Canonical HumanML3D rows may append token and segment fields after '#'.
        lines = [line.split("#", 1)[0].strip() for line in lines]
        lines = [line for line in lines if line]
    if query:
        normalized_query = query.casefold()
        lines = [line for line in lines if normalized_query in line.casefold()]
    total = len(lines)
    offset = max(0, min(offset, total))
    limit = max(1, min(limit, 200))
    end = min(total, offset + limit)
    return {
        "kind": "lines",
        "lines": lines[offset:end],
        "count": total,
        "offset": offset,
        "next_offset": end if end < total else None,
        "query": query,
    }


def clip_caption_for(row: sqlite3.Row):
    relative = row["clip_caption_path"]
    if not relative:
        return None
    path = resolve_data_path(relative)
    with path.open("rb") as handle:
        handle.seek(row["clip_caption_offset"])
        payload = json.loads(handle.read(row["clip_caption_length"]))
    description = str(payload.get("description", "")).strip()
    return {
        "kind": "clip",
        "description": description,
        "count": 1 if description else 0,
    }


def cot_for(row: sqlite3.Row):
    relative = row["cot_path"]
    if not relative:
        return None
    path = resolve_data_path(relative)
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = payload.get("per_segment", [])
    return {
        "kind": "cot",
        "sample_summary": payload.get("sample_summary", ""),
        "final_answer": payload.get("final_answer", ""),
        "segments": [
            {
                "time_range": item.get("time_range", ""),
                "cot_type": item.get("cot_type", ""),
                "think": item.get("think", ""),
                "answer": item.get("answer", ""),
            }
            for item in segments
        ],
        "count": len(segments),
    }


async def index_page(request: web.Request) -> web.StreamResponse:
    if request.host.partition(":")[0].lower() == "viewer.caimeng.online":
        raise web.HTTPFound("/viewer")
    return web.FileResponse(STATIC_ROOT / "index.html")


async def viewer_page(_: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_ROOT / "viewer.html")


async def api_summary(_: web.Request) -> web.Response:
    if not SUMMARY_PATH.is_file():
        raise web.HTTPServiceUnavailable(text="index has not been built")
    payload = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if TASK_STATS_PATH.is_file():
        payload["task_taxonomy"] = json.loads(
            TASK_STATS_PATH.read_text(encoding="utf-8")
        )
    payload["smplh"] = {
        "available": bool(SMPLH_MODEL_PATH and SMPLH_MODEL_PATH.is_file())
        and importlib.util.find_spec("smplx") is not None,
        "vertices": 6890,
        "faces": 13776,
        "joints": 52,
        "betas": 16,
        "pose_source": "21 body rotations from the 263D representation",
        "hands": "neutral",
        "shape": "neutral",
    }
    return web.json_response(payload)


async def api_samples(request: web.Request) -> web.Response:
    dataset = require_dataset(request.query.get("dataset", "humanml3d"))
    query = request.query.get("q", "").strip()[:120]
    try:
        limit = max(1, min(100, int(request.query.get("limit", "40"))))
        offset = max(0, int(request.query.get("offset", "0")))
    except ValueError as exc:
        raise web.HTTPBadRequest(text="invalid paging value") from exc
    has_video = request.query.get("has_video") == "1"
    has_text = request.query.get("has_text") == "1"
    has_cot = request.query.get("has_cot") == "1"

    clauses = ["dataset=?"]
    params: list[object] = [dataset]
    if query:
        clauses.append("id LIKE ?")
        params.append(f"%{query}%")
    if has_video:
        clauses.append("video_path IS NOT NULL")
    if has_text:
        clauses.append(
            "(caption_path IS NOT NULL OR clip_caption_path IS NOT NULL OR cot_path IS NOT NULL)"
        )
    if has_cot:
        clauses.append("cot_path IS NOT NULL")
    where = " AND ".join(clauses)
    with connect() as connection:
        total = connection.execute(
            f"SELECT COUNT(*) FROM samples WHERE {where}", params
        ).fetchone()[0]
        rows = connection.execute(
            f"""
            SELECT id, frames, fps, bytes,
                   video_path IS NOT NULL AS has_video,
                   caption_path IS NOT NULL AS has_caption,
                   clip_caption_path IS NOT NULL AS has_clip_caption,
                   cot_path IS NOT NULL AS has_cot
            FROM samples
            WHERE {where}
            ORDER BY id
            LIMIT ? OFFSET ?
            """,
            [*params, limit, offset],
        ).fetchall()
    return web.json_response(
        {
            "dataset": dataset,
            "total": total,
            "offset": offset,
            "limit": limit,
            "items": [dict(row) for row in rows],
        }
    )


async def api_sample(request: web.Request) -> web.Response:
    dataset = require_dataset(request.match_info["dataset"])
    sample_id = require_id(request.match_info["sample_id"])
    row = row_for(dataset, sample_id)
    payload = {
        "dataset": dataset,
        "id": sample_id,
        "frames": row["frames"],
        "fps": row["fps"],
        "duration": round(row["frames"] / row["fps"], 3),
        "bytes": row["bytes"],
        "has_video": row["video_path"] is not None,
        "sync_verified": False,
        "video_pairing": "sample-id match; temporal zero-point not independently verified",
        "video_url": f"/api/video/motionx/{sample_id}"
        if row["video_path"]
        else None,
        "caption": caption_for(row),
        "clip_caption": clip_caption_for(row) if dataset == "motionx" else None,
        "cot": cot_for(row) if dataset == "motionx" else None,
        "representation": {
            "features": 263,
            "joints": 22,
            "format": "HumanML3D RIC",
        },
    }
    return web.json_response(payload)


async def api_text(request: web.Request) -> web.Response:
    dataset = require_dataset(request.match_info["dataset"])
    sample_id = require_id(request.match_info["sample_id"])
    row = row_for(dataset, sample_id)
    try:
        offset = max(0, int(request.query.get("offset", "0")))
        limit = max(1, min(200, int(request.query.get("limit", "40"))))
    except ValueError as exc:
        raise web.HTTPBadRequest(text="invalid text paging value") from exc
    query = request.query.get("q", "").strip()[:160]
    if row["caption_offset"] is not None:
        return web.json_response(caption_for(row))
    payload = line_caption_for(row, offset=offset, limit=limit, query=query)
    if payload is None:
        raise web.HTTPNotFound(text="text annotation not found")
    return web.json_response(payload)


async def api_motion(request: web.Request) -> web.Response:
    dataset = require_dataset(request.match_info["dataset"])
    sample_id = require_id(request.match_info["sample_id"])
    row = row_for(dataset, sample_id)
    motion_path = resolve_data_path(row["motion_path"])
    joints, frames = pack_motion(str(motion_path))
    headers = {
        "X-Motion-Frames": str(frames),
        "X-Motion-Joints": "22",
        "X-Motion-Fps": str(row["fps"]),
        "X-Motion-Dtype": "float32-le",
        "Cache-Control": "private, max-age=300",
    }
    return web.Response(
        body=joints.tobytes(order="C"),
        content_type="application/octet-stream",
        headers=headers,
    )


async def api_video(request: web.Request) -> web.FileResponse:
    dataset = require_dataset(request.match_info["dataset"])
    if dataset != "motionx":
        raise web.HTTPNotFound(text="this dataset has no video pool")
    sample_id = require_id(request.match_info["sample_id"])
    row = row_for(dataset, sample_id)
    if not row["video_path"]:
        raise web.HTTPNotFound(text="video not found")
    response = web.FileResponse(resolve_data_path(row["video_path"]))
    response.headers["Cache-Control"] = "private, max-age=3600"
    return response


async def api_smplh_topology(_: web.Request) -> web.Response:
    try:
        from smplh_mesh import get_renderer

        renderer = await asyncio.get_event_loop().run_in_executor(None, get_renderer)
    except Exception as exc:
        raise web.HTTPServiceUnavailable(text=f"SMPL-H renderer unavailable: {exc}")
    return web.Response(
        body=renderer.faces.tobytes(order="C"),
        content_type="application/octet-stream",
        headers={
            "X-SMPLH-Faces": str(renderer.faces.shape[0]),
            "X-SMPLH-Index-Dtype": "uint32-le",
            "Cache-Control": "private, max-age=86400",
        },
    )


async def api_smplh_mesh(request: web.Request) -> web.Response:
    dataset = require_dataset(request.match_info["dataset"])
    sample_id = require_id(request.match_info["sample_id"])
    try:
        frame = int(request.match_info["frame"])
    except ValueError as exc:
        raise web.HTTPBadRequest(text="invalid frame") from exc
    row = row_for(dataset, sample_id)
    motion_path = resolve_data_path(row["motion_path"])
    try:
        from smplh_mesh import get_renderer

        loop = asyncio.get_event_loop()
        renderer = await loop.run_in_executor(None, get_renderer)
        vertices = await loop.run_in_executor(
            None, renderer.vertices_for_frame, motion_path, frame
        )
    except Exception as exc:
        raise web.HTTPServiceUnavailable(text=f"SMPL-H conversion failed: {exc}")
    return web.Response(
        body=vertices.tobytes(order="C"),
        content_type="application/octet-stream",
        headers={
            "X-SMPLH-Vertices": str(vertices.shape[0]),
            "X-SMPLH-Dtype": "float32-le",
            "X-Motion-Frame": str(max(0, min(frame, row["frames"] - 1))),
            "Cache-Control": "private, max-age=3600",
        },
    )


async def health(_: web.Request) -> web.Response:
    return web.json_response(
        {
            "ok": INDEX_PATH.is_file() and SUMMARY_PATH.is_file(),
            "datasets": sorted(DATASETS),
            "kinematic_chains": KINEMATIC_CHAINS,
        }
    )


def create_app() -> web.Application:
    app = web.Application(client_max_size=4 * 1024 * 1024)
    app.router.add_get("/", index_page)
    app.router.add_get("/viewer", viewer_page)
    app.router.add_get("/health", health)
    app.router.add_get("/api/summary", api_summary)
    app.router.add_get("/api/samples", api_samples)
    app.router.add_get("/api/sample/{dataset}/{sample_id}", api_sample)
    app.router.add_get("/api/text/{dataset}/{sample_id}", api_text)
    app.router.add_get("/api/motion/{dataset}/{sample_id}", api_motion)
    app.router.add_get("/api/video/{dataset}/{sample_id}", api_video)
    app.router.add_get("/api/smplh/topology", api_smplh_topology)
    app.router.add_get(
        "/api/smplh/mesh/{dataset}/{sample_id}/{frame}", api_smplh_mesh
    )
    app.router.add_static("/static", STATIC_ROOT, show_index=False)
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not INDEX_PATH.is_file():
        raise SystemExit("Run `python3 build_index.py` before starting the server.")
    web.run_app(create_app(), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
