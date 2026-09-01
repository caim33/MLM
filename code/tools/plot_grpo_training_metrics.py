#!/usr/bin/env python3
"""Plot and summarize GRPO training metrics from an ms-swift run directory."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class BasicStats:
    mean: float
    min: float
    max: float
    min_step: Optional[int] = None
    max_step: Optional[int] = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze and plot GRPO training metrics.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to one run directory, e.g. .../v0-YYYYMMDD-HHMMSS",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for figures/report. Default: <run-dir>/analysis",
    )
    parser.add_argument(
        "--ema",
        type=int,
        default=20,
        help="EMA span for smoothed curves. Default: 20",
    )
    parser.add_argument(
        "--tail-window",
        type=int,
        default=20,
        help="Window size for tail focus plot/report. Default: 20",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI. Default: 150",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or not raw.startswith("{"):
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                yield data


def _parse_step(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if "/" in text:
            text = text.split("/", 1)[0]
        try:
            return int(text)
        except ValueError:
            return None
    return None


def _to_float(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _extract_metric_rows(logging_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for data in _read_jsonl(logging_path):
        step = _parse_step(data.get("global_step/max_steps"))
        reward = _to_float(data.get("reward"))
        if step is None or reward is None:
            continue
        rows.append(
            {
                "step": step,
                "reward": reward,
                "reward_std": _to_float(data.get("reward_std")),
                "semantic": _to_float(data.get("rewards/MotionSemanticORM/mean")),
                "format": _to_float(data.get("rewards/MotionFormatORM/mean")),
                "vmv_bonus": _to_float(data.get("rewards/MotionVMVGroupBonusORM/mean")),
                "mean_length": _to_float(data.get("completions/mean_length")),
                "min_length": _to_float(data.get("completions/min_length")),
                "max_length": _to_float(data.get("completions/max_length")),
                "clipped_ratio": _to_float(data.get("completions/clipped_ratio")),
                "kl": _to_float(data.get("kl")),
            }
        )
    if not rows:
        raise ValueError(f"No metric rows with reward found in {logging_path}")

    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    return df


def _ema(series: pd.Series, span: int) -> pd.Series:
    if span <= 1:
        return series.copy()
    return series.ewm(span=span, adjust=False).mean()


def _corr(df: pd.DataFrame, x: str, y: str) -> float:
    sub = df[[x, y]].dropna()
    if len(sub) < 2:
        return float("nan")
    return float(sub[x].corr(sub[y]))


def _basic_stats(df: pd.DataFrame, metric: str) -> BasicStats:
    sub = df[["step", metric]].dropna()
    if sub.empty:
        return BasicStats(mean=float("nan"), min=float("nan"), max=float("nan"))
    idx_min = sub[metric].idxmin()
    idx_max = sub[metric].idxmax()
    return BasicStats(
        mean=float(sub[metric].mean()),
        min=float(sub.loc[idx_min, metric]),
        max=float(sub.loc[idx_max, metric]),
        min_step=int(sub.loc[idx_min, "step"]),
        max_step=int(sub.loc[idx_max, "step"]),
    )


def _bucket_ranges(max_step: int) -> List[Tuple[int, int]]:
    # ms-swift reward rows may end at 179 even when max_steps is 180.
    if max_step >= 179:
        return [(1, 60), (61, 120), (121, 180)]
    b = max(1, max_step // 3)
    return [(1, b), (b + 1, 2 * b), (2 * b + 1, max_step)]


def _bucket_stats(df: pd.DataFrame, max_step: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for start, end in _bucket_ranges(max_step):
        seg = df[(df["step"] >= start) & (df["step"] <= end)]
        if seg.empty:
            continue
        out.append(
            {
                "range": f"{start}-{end}",
                "count": int(len(seg)),
                "reward_mean": float(seg["reward"].mean()),
                "semantic_mean": float(seg["semantic"].mean()) if seg["semantic"].notna().any() else float("nan"),
                "format_mean": float(seg["format"].mean()) if seg["format"].notna().any() else float("nan"),
                "vmv_mean": float(seg["vmv_bonus"].mean()) if seg["vmv_bonus"].notna().any() else float("nan"),
                "clipped_mean": float(seg["clipped_ratio"].mean()) if seg["clipped_ratio"].notna().any() else float("nan"),
                "mean_length_mean": float(seg["mean_length"].mean()) if seg["mean_length"].notna().any() else float("nan"),
            }
        )
    return out


def _clip_group_stats(df: pd.DataFrame) -> Dict[str, Any]:
    clip_col = df["clipped_ratio"].fillna(0.0)
    clip0 = df[clip_col == 0.0]
    clip_pos = df[clip_col > 0.0]
    res: Dict[str, Any] = {}
    for name, sub in [("clip_eq_0", clip0), ("clip_gt_0", clip_pos)]:
        if sub.empty:
            res[name] = {"count": 0}
            continue
        res[name] = {
            "count": int(len(sub)),
            "reward_mean": float(sub["reward"].mean()),
            "semantic_mean": float(sub["semantic"].mean()) if sub["semantic"].notna().any() else float("nan"),
            "format_mean": float(sub["format"].mean()) if sub["format"].notna().any() else float("nan"),
        }
    return res


def _top_bottom_steps(df: pd.DataFrame, k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    cols = ["step", "reward", "clipped_ratio", "mean_length", "format", "semantic", "vmv_bonus"]
    valid = df[cols].dropna(subset=["reward"])
    best = valid.nlargest(k, "reward")
    worst = valid.nsmallest(k, "reward")
    return {
        "top_reward_steps": best.to_dict(orient="records"),
        "bottom_reward_steps": worst.to_dict(orient="records"),
    }


def _safe_fmt(value: Any, ndigits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.{ndigits}f}"
    return str(value)


def _count_strict_answer_warnings(train_log: Path) -> int:
    if not train_log.exists():
        return 0
    text = train_log.read_text(encoding="utf-8", errors="ignore")
    return len(re.findall(r"strict <answer> extraction missing tags", text))


def _save_total_reward(df: pd.DataFrame, out: Path, ema_span: int, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["step"], df["reward"], color="#5a5a5a", alpha=0.45, linewidth=1.2, label="reward (raw)")
    ax.plot(df["step"], _ema(df["reward"], ema_span), color="#0072B2", linewidth=2.0, label=f"reward (EMA{ema_span})")
    ax.set_title("Total Reward Trend")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def _save_reward_components(df: pd.DataFrame, out: Path, ema_span: int, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for col, color, name in [
        ("semantic", "#009E73", "semantic"),
        ("format", "#CC79A7", "format"),
        ("vmv_bonus", "#D55E00", "vmv_bonus"),
    ]:
        if not df[col].notna().any():
            continue
        ax.plot(df["step"], df[col], color=color, alpha=0.2, linewidth=1.0, label=f"{name} (raw)")
        ax.plot(df["step"], _ema(df[col], ema_span), color=color, linewidth=2.0, label=f"{name} (EMA{ema_span})")
    ax.set_title("Reward Components Trend")
    ax.set_xlabel("Step")
    ax.set_ylabel("Component Score")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def _save_length_clip(df: pd.DataFrame, out: Path, ema_span: int, dpi: int) -> None:
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    if df["mean_length"].notna().any():
        ax1.plot(df["step"], df["mean_length"], color="#5a5a5a", alpha=0.3, linewidth=1.0, label="mean_length (raw)")
        ax1.plot(
            df["step"],
            _ema(df["mean_length"], ema_span),
            color="#1F77B4",
            linewidth=2.0,
            label=f"mean_length (EMA{ema_span})",
        )
    if df["clipped_ratio"].notna().any():
        ax2.plot(
            df["step"],
            df["clipped_ratio"],
            color="#B22222",
            alpha=0.25,
            linewidth=1.0,
            label="clipped_ratio (raw)",
        )
        ax2.plot(
            df["step"],
            _ema(df["clipped_ratio"], ema_span),
            color="#E69F00",
            linewidth=2.0,
            label=f"clipped_ratio (EMA{ema_span})",
        )

    ax1.set_title("Completion Length & Clipped Ratio")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Mean Completion Length")
    ax2.set_ylabel("Clipped Ratio")
    ax2.set_ylim(0.0, 1.0)
    ax1.grid(alpha=0.25)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def _save_scatter_with_fit(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out: Path,
    dpi: int,
) -> None:
    sub = df[[x, y]].dropna()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(sub[x], sub[y], alpha=0.65, s=24, color="#1f77b4")
    if len(sub) >= 2:
        coeff = np.polyfit(sub[x], sub[y], deg=1)
        x_line = np.linspace(float(sub[x].min()), float(sub[x].max()), 100)
        y_line = coeff[0] * x_line + coeff[1]
        ax.plot(x_line, y_line, color="#d62728", linewidth=2.0, label="linear fit")
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def _save_format_semantic_total(df: pd.DataFrame, out: Path, ema_span: int, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for col, color in [("reward", "#0072B2"), ("semantic", "#009E73"), ("format", "#CC79A7")]:
        if not df[col].notna().any():
            continue
        ax.plot(df["step"], df[col], color=color, alpha=0.2, linewidth=1.0, label=f"{col} (raw)")
        ax.plot(df["step"], _ema(df[col], ema_span), color=color, linewidth=2.0, label=f"{col} (EMA{ema_span})")
    ax.set_title("Total vs Semantic/Format")
    ax.set_xlabel("Step")
    ax.set_ylabel("Score")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def _save_tail_focus(df: pd.DataFrame, out: Path, ema_span: int, tail_window: int, dpi: int) -> None:
    tail = df.tail(tail_window).copy()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(tail["step"], tail["reward"], color="#5a5a5a", alpha=0.35, linewidth=1.2, label="reward (raw)")
    axes[0].plot(
        tail["step"],
        _ema(tail["reward"], min(ema_span, max(2, tail_window // 2))),
        color="#0072B2",
        linewidth=2.0,
        label="reward (EMA)",
    )
    axes[0].set_ylabel("Reward")
    axes[0].set_title(f"Tail Window Focus (last {tail_window} metric steps)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    ax2 = axes[1]
    ax2b = ax2.twinx()
    if tail["mean_length"].notna().any():
        ax2.plot(tail["step"], tail["mean_length"], color="#1F77B4", linewidth=1.8, label="mean_length")
    if tail["clipped_ratio"].notna().any():
        ax2b.plot(tail["step"], tail["clipped_ratio"], color="#D55E00", linewidth=1.8, label="clipped_ratio")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Mean Length")
    ax2b.set_ylabel("Clipped Ratio")
    ax2b.set_ylim(0.0, 1.0)
    ax2.grid(alpha=0.25)

    lines_a, labels_a = ax2.get_legend_handles_labels()
    lines_b, labels_b = ax2b.get_legend_handles_labels()
    ax2.legend(lines_a + lines_b, labels_a + labels_b, loc="upper left")

    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def _dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_summary(df: pd.DataFrame, run_dir: Path, train_log: Path, tail_window: int) -> Dict[str, Any]:
    max_step = int(df["step"].max())
    reward_stats = _basic_stats(df, "reward")
    sem_stats = _basic_stats(df, "semantic")
    fmt_stats = _basic_stats(df, "format")
    vmv_stats = _basic_stats(df, "vmv_bonus")
    clip_stats = _basic_stats(df, "clipped_ratio")
    len_stats = _basic_stats(df, "mean_length")

    clip_col = df["clipped_ratio"].fillna(0.0)
    clip_gt0 = int((clip_col > 0.0).sum())
    tail = df.tail(tail_window)

    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "metric_steps": int(len(df)),
        "max_step": max_step,
        "reward": reward_stats.__dict__,
        "semantic": sem_stats.__dict__,
        "format": fmt_stats.__dict__,
        "vmv_bonus": vmv_stats.__dict__,
        "clipped_ratio": {
            **clip_stats.__dict__,
            "gt0_count": clip_gt0,
            "gt0_ratio": float(clip_gt0 / len(df)),
        },
        "mean_length": len_stats.__dict__,
        "correlations": {
            "clip_vs_reward": _corr(df, "clipped_ratio", "reward"),
            "length_vs_reward": _corr(df, "mean_length", "reward"),
            "format_vs_reward": _corr(df, "format", "reward"),
            "semantic_vs_reward": _corr(df, "semantic", "reward"),
        },
        "bucket_stats": _bucket_stats(df, max_step=max_step),
        "clip_group_stats": _clip_group_stats(df),
        "tail_window": {
            "window": int(tail_window),
            "start_step": int(tail["step"].min()),
            "end_step": int(tail["step"].max()),
            "reward_mean": float(tail["reward"].mean()),
            "clipped_mean": float(tail["clipped_ratio"].mean()) if tail["clipped_ratio"].notna().any() else float("nan"),
        },
        "top_bottom": _top_bottom_steps(df, k=5),
        "strict_answer_missing_tag_warnings": _count_strict_answer_warnings(train_log),
    }
    return summary


def _write_report(path: Path, summary: Dict[str, Any], out_dir: Path, ema_span: int) -> None:
    corr = summary["correlations"]
    clip_stats = summary["clipped_ratio"]
    lines: List[str] = []
    lines.append("# GRPO Training Metrics Report")
    lines.append("")
    lines.append(f"- Run directory: `{summary['run_dir']}`")
    lines.append(f"- Metric steps: `{summary['metric_steps']}`")
    lines.append(f"- Max step: `{summary['max_step']}`")
    lines.append(f"- EMA span: `{ema_span}`")
    lines.append("")
    lines.append("## Global Summary")
    lines.append("")
    lines.append(
        f"- Reward mean/min/max: `{_safe_fmt(summary['reward']['mean'])}` / "
        f"`{_safe_fmt(summary['reward']['min'])}` (step {summary['reward']['min_step']}) / "
        f"`{_safe_fmt(summary['reward']['max'])}` (step {summary['reward']['max_step']})"
    )
    lines.append(
        f"- Semantic mean: `{_safe_fmt(summary['semantic']['mean'])}`, "
        f"Format mean: `{_safe_fmt(summary['format']['mean'])}`, "
        f"VMV mean: `{_safe_fmt(summary['vmv_bonus']['mean'])}`"
    )
    lines.append(
        f"- Clipped ratio mean/max: `{_safe_fmt(clip_stats['mean'])}` / `{_safe_fmt(clip_stats['max'])}`; "
        f"`clip>0` count: `{clip_stats['gt0_count']}/{summary['metric_steps']}`"
    )
    lines.append(
        f"- Mean completion length mean/max: `{_safe_fmt(summary['mean_length']['mean'])}` / "
        f"`{_safe_fmt(summary['mean_length']['max'])}`"
    )
    lines.append("")
    lines.append("## Correlation Diagnostics")
    lines.append("")
    lines.append(f"- clip vs reward: `{_safe_fmt(corr['clip_vs_reward'])}`")
    lines.append(f"- mean_length vs reward: `{_safe_fmt(corr['length_vs_reward'])}`")
    lines.append(f"- format vs reward: `{_safe_fmt(corr['format_vs_reward'])}`")
    lines.append(f"- semantic vs reward: `{_safe_fmt(corr['semantic_vs_reward'])}`")
    lines.append("")
    lines.append("## Buckets")
    lines.append("")
    for item in summary["bucket_stats"]:
        lines.append(
            f"- {item['range']}: reward `{_safe_fmt(item['reward_mean'])}`, "
            f"semantic `{_safe_fmt(item['semantic_mean'])}`, "
            f"format `{_safe_fmt(item['format_mean'])}`, "
            f"vmv `{_safe_fmt(item['vmv_mean'])}`, "
            f"clip `{_safe_fmt(item['clipped_mean'])}`, "
            f"length `{_safe_fmt(item['mean_length_mean'])}`"
        )
    lines.append("")
    lines.append("## Warning Counter")
    lines.append("")
    lines.append(
        f"- strict `<answer>` extraction missing tags warnings: "
        f"`{summary['strict_answer_missing_tag_warnings']}`"
    )
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for name in [
        "total_reward_trend.png",
        "reward_components_trend.png",
        "length_clip_trend.png",
        "reward_vs_clip_scatter.png",
        "reward_vs_length_scatter.png",
        "format_semantic_vs_total.png",
        "tail_window_focus.png",
    ]:
        lines.append(f"- `{(out_dir / name).as_posix()}`")
    lines.append("")
    lines.append("## Top/Bottom Reward Steps")
    lines.append("")
    lines.append("- Top 5 reward steps:")
    for row in summary["top_bottom"]["top_reward_steps"]:
        lines.append(
            f"  - step `{row['step']}` reward `{_safe_fmt(row['reward'])}` "
            f"clip `{_safe_fmt(row.get('clipped_ratio'))}` len `{_safe_fmt(row.get('mean_length'))}` "
            f"format `{_safe_fmt(row.get('format'))}` semantic `{_safe_fmt(row.get('semantic'))}`"
        )
    lines.append("- Bottom 5 reward steps:")
    for row in summary["top_bottom"]["bottom_reward_steps"]:
        lines.append(
            f"  - step `{row['step']}` reward `{_safe_fmt(row['reward'])}` "
            f"clip `{_safe_fmt(row.get('clipped_ratio'))}` len `{_safe_fmt(row.get('mean_length'))}` "
            f"format `{_safe_fmt(row.get('format'))}` semantic `{_safe_fmt(row.get('semantic'))}`"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    out_dir = args.out_dir.resolve() if args.out_dir else (run_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    logging_path = run_dir / "logging.jsonl"
    train_log = run_dir.parent / "train.log"
    if not logging_path.exists():
        raise FileNotFoundError(f"logging.jsonl not found: {logging_path}")

    df = _extract_metric_rows(logging_path)
    ema_span = max(1, int(args.ema))
    tail_window = max(5, int(args.tail_window))

    _save_total_reward(df, out_dir / "total_reward_trend.png", ema_span=ema_span, dpi=args.dpi)
    _save_reward_components(df, out_dir / "reward_components_trend.png", ema_span=ema_span, dpi=args.dpi)
    _save_length_clip(df, out_dir / "length_clip_trend.png", ema_span=ema_span, dpi=args.dpi)
    _save_scatter_with_fit(
        df,
        x="clipped_ratio",
        y="reward",
        title="Reward vs Clipped Ratio",
        xlabel="Clipped Ratio",
        ylabel="Reward",
        out=out_dir / "reward_vs_clip_scatter.png",
        dpi=args.dpi,
    )
    _save_scatter_with_fit(
        df,
        x="mean_length",
        y="reward",
        title="Reward vs Mean Completion Length",
        xlabel="Mean Completion Length",
        ylabel="Reward",
        out=out_dir / "reward_vs_length_scatter.png",
        dpi=args.dpi,
    )
    _save_format_semantic_total(df, out_dir / "format_semantic_vs_total.png", ema_span=ema_span, dpi=args.dpi)
    _save_tail_focus(
        df,
        out=out_dir / "tail_window_focus.png",
        ema_span=ema_span,
        tail_window=tail_window,
        dpi=args.dpi,
    )

    summary = _build_summary(df, run_dir=run_dir, train_log=train_log, tail_window=tail_window)
    _dump_json(out_dir / "metrics_summary.json", summary)
    _write_report(out_dir / "report.md", summary=summary, out_dir=out_dir, ema_span=ema_span)

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "out_dir": str(out_dir),
                "metric_steps": summary["metric_steps"],
                "max_step": summary["max_step"],
                "reward_mean": summary["reward"]["mean"],
                "clipped_ratio_mean": summary["clipped_ratio"]["mean"],
                "strict_answer_missing_tag_warnings": summary["strict_answer_missing_tag_warnings"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
