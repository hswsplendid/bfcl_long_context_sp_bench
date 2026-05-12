#!/usr/bin/env python3
"""Build cw8000-reference-style Semi-Prefill overhead reports.

This creates a second set of analysis folders without touching existing results.
The report and chart names intentionally mirror
/root/semi_prefill_bench/results/cw8000_cw8000_rs500/analysis.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/root")
TODAY = "20260507"
R_PF = 0.15
R_DEC = 12.0
C_FIXED = 20.0

COLORS = {
    "full": "#1E88E5",
    "incr": "#43A047",
    "semi": "#FF5722",
    "decode": "#9C27B0",
    "sum_pf": "#8D6E63",
    "sum_dec": "#D32F2F",
    "b2": "#FF9800",
    "c1": "#FF5722",
    "a": "#E0E0E0",
}


@dataclass(frozen=True)
class Dataset:
    slug: str
    title: str
    data_root: Path
    output_dir: Path
    kind: str
    report_name: str
    caveat: str
    fallback_config: dict[str, Any]


DATASETS = [
    Dataset(
        slug="tau2_telecom",
        title="Tau2 Telecom MMS Semi-Prefill Overhead",
        data_root=ROOT
        / "tau2_telecom_sp_bench/results/real_mms_easy_user3_agent1_n20_compressed_only_cw32000_thr30000_c12800_20260507_001",
        output_dir=ROOT
        / "tau2_telecom_sp_bench/results/real_mms_easy_user3_agent1_n20_compressed_only_cw32000_thr30000_c12800_20260507_001"
        / f"analysis_{TODAY}_cw8000_reference",
        kind="flat",
        report_name="Semi-Prefill_Overhead分析报告_tau2_telecom_cw8000_reference.md",
        caveat="目录名声明 n20，但当前目录中只有 4 个 timing/ABC 文件，其中 3 个含有效 timing。",
        fallback_config={},
    ),
    Dataset(
        slug="agentbench_ltp",
        title="AgentBench LTP Semi-Prefill Overhead",
        data_root=ROOT / "agentbench_semi_prefill_bench/results/compressed",
        output_dir=ROOT / "agentbench_semi_prefill_bench" / f"analysis_{TODAY}_cw8000_reference",
        kind="flat",
        report_name="Semi-Prefill_Overhead分析报告_agentbench_ltp_cw8000_reference.md",
        caveat="compressed 目录缺少 run_config.json；context/threshold 参数来自 /root/agentbench_semi_prefill_bench/config.py 的 32k 配置。",
        fallback_config={
            "model": "Llama-3.3-70B-Instruct",
            "context_window": 32000,
            "threshold": 30000,
            "reserve_tokens": 2000,
            "keep_recent_tokens": 2800,
            "summary_max_tokens": 1024,
            "config_source": "/root/agentbench_semi_prefill_bench/config.py",
        },
    ),
    Dataset(
        slug="bfcl_long_context",
        title="BFCL Long-Context Semi-Prefill Overhead",
        data_root=ROOT / "bfcl_long_context_sp_bench/results",
        output_dir=ROOT / "bfcl_long_context_sp_bench/results" / f"analysis_{TODAY}_cw8000_reference",
        kind="multi_run",
        report_name="Semi-Prefill_Overhead分析报告_bfcl_long_context_cw8000_reference.md",
        caveat="validate 样本在 score summary 中为 invalid；本报告只分析压缩/prefill 工作负载，不报告最终准确率。",
        fallback_config={},
    ),
]


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def num(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def pick(source: dict[str, Any] | None, keys: list[str]) -> float | None:
    if not isinstance(source, dict):
        return None
    for key in keys:
        value = num(source.get(key))
        if value is not None:
            return value
    return None


def turn_of(row: dict[str, Any], fallback: int = 0) -> int:
    value = row.get("turn", row.get("turn_idx", fallback))
    parsed = num(value)
    return int(parsed) if parsed is not None else fallback


def step_of(row: dict[str, Any]) -> int | None:
    parsed = num(row.get("step", row.get("step_idx")))
    return int(parsed) if parsed is not None else None


def label_for(sample_id: str, run_name: str) -> str:
    if sample_id.startswith("puzzle_"):
        return "P" + sample_id.split("_", 1)[1]
    if sample_id.startswith("multi_turn_long_context_"):
        suffix = sample_id.rsplit("_", 1)[-1]
        return ("V" if run_name.startswith("validate_") else "C") + suffix
    match = re.search(r"([0-9a-f]{8,12})$", sample_id)
    if match:
        return match.group(1)[:6]
    return sample_id if len(sample_id) <= 18 else sample_id[:8] + "..." + sample_id[-6:]


def discover(dataset: Dataset) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    if dataset.kind == "flat":
        timing_dir = dataset.data_root / "timing"
        for timing_path in sorted(timing_dir.glob("*.json")):
            sources.append({"run_root": dataset.data_root, "run_name": dataset.data_root.name, "timing_path": timing_path})
    else:
        for run_root in sorted(dataset.data_root.iterdir()):
            timing_dir = run_root / "timing"
            if not timing_dir.is_dir():
                continue
            for timing_path in sorted(timing_dir.glob("*.json")):
                sources.append({"run_root": run_root, "run_name": run_root.name, "timing_path": timing_path})
    return sources


def find_abc(run_root: Path, sample_id: str) -> Path | None:
    abc_dir = run_root / "abc_segments"
    if not abc_dir.exists():
        return None
    exact = abc_dir / f"{sample_id}.json"
    if exact.exists():
        return exact
    matches = sorted(abc_dir.glob(f"{sample_id}*.json"))
    return matches[0] if matches else None


def summary_rows(data_root: Path, run_root: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in [run_root / "summary.jsonl", run_root / "agentbench_results/results.jsonl", run_root / "tau2_results/results.json"]:
        if path.suffix == ".jsonl":
            loaded = load_jsonl(path)
        else:
            obj = load_json(path, [])
            loaded = obj if isinstance(obj, list) else [obj] if isinstance(obj, dict) else []
        for item in loaded:
            if not isinstance(item, dict):
                continue
            key = item.get("id") or item.get("sample_id") or item.get("task_id") or item.get("puzzle_id")
            if key is not None:
                rows[str(key)] = item
    for path in sorted(data_root.glob("*score_summary*.json")):
        obj = load_json(path, [])
        if not isinstance(obj, list):
            continue
        for item in obj:
            if isinstance(item, dict) and item.get("id") is not None:
                rows[str(item["id"])] = {**rows.get(str(item["id"]), {}), **item}
    return rows


def extract_events(raw: Any, sample_id: str, label: str, run_name: str) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    events: list[dict[str, Any]] = []
    for idx, event in enumerate(raw, start=1):
        if not isinstance(event, dict):
            continue
        segments = event.get("abc_segments") if isinstance(event.get("abc_segments"), dict) else {}
        before = segments.get("before", {}) if isinstance(segments, dict) else {}
        after = segments.get("after", {}) if isinstance(segments, dict) else {}
        pre = pick(event, ["pre_prompt_tokens", "before_prompt_tokens", "original_prompt_tokens"])
        post = pick(event, ["post_prompt_tokens", "after_prompt_tokens", "compressed_prompt_tokens"])
        a = pick(event, ["A_tokens", "a_tokens"]) or pick(after, ["A_tokens", "a_tokens"]) or pick(before, ["A_tokens", "a_tokens"]) or 0.0
        b1 = pick(event, ["B1_tokens", "b1_tokens"]) or pick(before, ["B1_tokens", "b1_tokens"]) or 0.0
        b2 = pick(event, ["B2_tokens", "B2_tokens_after", "b2_tokens"]) or pick(after, ["B2_tokens", "B2_tokens_after", "b2_tokens"])
        c1 = pick(event, ["C1_tokens_after", "C1_after_tokens", "C1_tokens", "c1_tokens"]) or pick(
            after, ["C1_tokens_after", "C1_tokens", "c1_tokens"]
        ) or pick(before, ["C1_tokens", "c1_tokens"])
        if b2 is None and post is not None and c1 is not None:
            b2 = max(post - a - c1, 0.0)
        if c1 is None and post is not None and b2 is not None:
            c1 = max(post - a - b2, 0.0)
        b2 = b2 or 0.0
        c1 = c1 or 0.0
        b2_c1 = pick(event, ["semi_prefill_tokens", "B2_plus_C1_tokens", "compression_prefill_tokens"])
        if b2_c1 is None:
            b2_c1 = b2 + c1
        saving = pick(event, ["token_saving_pct", "saving_pct"])
        if saving is None and pre:
            saving = ((pre - (post or 0.0)) / pre) * 100.0
        events.append(
            {
                "sample_id": sample_id,
                "sample_label": label,
                "run_name": run_name,
                "event": idx,
                "turn": turn_of(event, -1),
                "step": step_of(event),
                "A": a,
                "B1": b1,
                "B2": b2,
                "C1": c1,
                "B2_C1": b2_c1 or 0.0,
                "pre_prompt_tokens": pre or 0.0,
                "post_prompt_tokens": post or 0.0,
                "semi_prefill_ms": (b2_c1 or 0.0) * R_PF + C_FIXED,
                "summary_prefill_ms": b1 * R_PF,
                "summary_decode_ms": b2 * R_DEC,
                "summary_generation_time_s": pick(event, ["summary_generation_time_s", "summary_time_s"]) or 0.0,
                "token_saving_pct": saving or 0.0,
            }
        )
    return events


def config_from_run(run_config: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    context = run_config.get("context", {}) if isinstance(run_config, dict) else {}
    preset = run_config.get("preset", {}) if isinstance(run_config, dict) else {}
    return {
        "model": run_config.get("model") or fallback.get("model") or "Llama-3.3-70B-Instruct",
        "context_window": context.get("context_window") or preset.get("context_window") or fallback.get("context_window"),
        "threshold": context.get("threshold_tokens") or preset.get("threshold") or fallback.get("threshold"),
        "reserve_tokens": context.get("reserve_tokens") or preset.get("reserve_tokens") or fallback.get("reserve_tokens"),
        "keep_recent_tokens": context.get("keep_recent_tokens") or preset.get("keep_recent_tokens") or fallback.get("keep_recent_tokens"),
        "summary_max_tokens": context.get("summary_max_tokens") or preset.get("summary_max_tokens") or fallback.get("summary_max_tokens"),
        "source": "run_config.json" if run_config else fallback.get("config_source", "fallback"),
    }


def build_sample(source: dict[str, Any], dataset: Dataset, summaries_cache: dict[Path, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    run_root = source["run_root"]
    run_name = source["run_name"]
    timing_path = source["timing_path"]
    sample_id = timing_path.stem
    label = label_for(sample_id, run_name)
    timing_raw = load_json(timing_path, [])
    timing = [row for row in timing_raw if isinstance(row, dict)] if isinstance(timing_raw, list) else []
    abc_path = find_abc(run_root, sample_id)
    events = extract_events(load_json(abc_path, []) if abc_path else [], sample_id, label, run_name)
    run_config = load_json(run_root / "run_config.json", {}) or {}
    config = config_from_run(run_config, dataset.fallback_config)
    if run_root not in summaries_cache:
        summaries_cache[run_root] = summary_rows(dataset.data_root, run_root)
    summary = summaries_cache[run_root].get(sample_id, {})

    turns: dict[int, dict[str, float]] = defaultdict(lambda: {"prompt": 0.0, "output": 0.0, "sp_tokens": 0.0})
    classifications = Counter()
    prompts: list[float] = []
    outputs: list[float] = []
    timing_total_ms = 0.0
    for index, row in enumerate(timing):
        turn = turn_of(row, index)
        prompt = num(row.get("prompt_tokens"), 0.0) or 0.0
        output = num(row.get("output_tokens"), 0.0) or 0.0
        prompts.append(prompt)
        outputs.append(output)
        turns[turn]["prompt"] = max(turns[turn]["prompt"], prompt)
        turns[turn]["output"] += output
        sp_tokens = num(row.get("semi_prefill_tokens"), 0.0) or 0.0
        if row.get("classification") == "semi_prefill" and sp_tokens == 0.0:
            sp_tokens = prompt
        turns[turn]["sp_tokens"] += sp_tokens
        classifications[str(row.get("classification") or "unknown")] += 1
        timing_total_ms += num(row.get("total_ms"), 0.0) or 0.0

    context_series = [{"turn": turn, "prompt_tokens": values["prompt"]} for turn, values in sorted(turns.items())]
    if summary.get("turns") or summary.get("decoded_turns"):
        rounds = int(num(summary.get("turns") or summary.get("decoded_turns"), 0) or 0)
    else:
        rounds = len(context_series)
    event_by_turn = defaultdict(list)
    for event in events:
        event_by_turn[int(event["turn"])].append(event)

    full_prefill_ms = (prompts[0] * R_PF + C_FIXED) if prompts else 0.0
    incremental_prefill_ms = 0.0
    churn_rows: list[dict[str, float]] = []
    previous = prompts[0] if prompts else 0.0
    for item in context_series:
        turn = int(item["turn"])
        prompt = item["prompt_tokens"]
        if turn == context_series[0]["turn"]:
            churn = 1.0 if prompt else 0.0
        elif event_by_turn.get(turn):
            pref = sum(event["B2_C1"] for event in event_by_turn[turn])
            churn = pref / prompt if prompt else 0.0
        else:
            delta = max(prompt - previous, 0.0)
            churn = delta / prompt if prompt else 0.0
            incremental_prefill_ms += delta * R_PF + C_FIXED
        churn_rows.append({"turn": turn, "churn": churn})
        previous = prompt if prompt > 0 else previous

    semi_prefill_ms = sum(event["semi_prefill_ms"] for event in events)
    summary_prefill_ms = sum(event["summary_prefill_ms"] for event in events)
    summary_decode_ms = sum(event["summary_decode_ms"] for event in events)
    decode_ms = sum(outputs) * R_DEC
    baseline_ms = full_prefill_ms + incremental_prefill_ms + decode_ms
    async_total_ms = baseline_ms + semi_prefill_ms
    sync_total_ms = async_total_ms + summary_prefill_ms + summary_decode_ms

    delta_values = []
    previous_prompt = None
    compression_turns = {int(event["turn"]) for event in events}
    for item in context_series:
        prompt = item["prompt_tokens"]
        if previous_prompt is not None and int(item["turn"]) not in compression_turns:
            delta_values.append(max(prompt - previous_prompt, 0.0))
        if prompt > 0:
            previous_prompt = prompt

    return {
        "sample_id": sample_id,
        "sample_label": label,
        "run_name": run_name,
        "active": bool(prompts and prompts[0] > 0 and len(timing) > 0),
        "rounds": rounds,
        "steps": len(timing),
        "n_compressions": len(events),
        "compression_rate": len(events) / rounds if rounds else 0.0,
        "first_context": prompts[0] if prompts else 0.0,
        "max_context": max(prompts) if prompts else 0.0,
        "accumulated_context": sum(prompts),
        "context_multiplier": (sum(prompts) / prompts[0]) if prompts and prompts[0] else 0.0,
        "output_tokens": sum(outputs),
        "timing_total_ms": timing_total_ms,
        "classification_counts": dict(classifications),
        "delta_median": median(delta_values) if delta_values else 0.0,
        "delta_mean": mean(delta_values) if delta_values else 0.0,
        "delta_min": min(delta_values) if delta_values else 0.0,
        "delta_max": max(delta_values) if delta_values else 0.0,
        "context_series": context_series,
        "churn_series": churn_rows,
        "events": events,
        "config": config,
        "score_valid": summary.get("valid"),
        "score": summary.get("score"),
        "error_type": summary.get("error_type"),
        "phase_async": {
            "full_prefill_ms": full_prefill_ms,
            "incremental_prefill_ms": incremental_prefill_ms,
            "semi_prefill_ms": semi_prefill_ms,
            "decode_ms": decode_ms,
            "total_ms": async_total_ms,
        },
        "phase_sync": {
            "full_prefill_ms": full_prefill_ms,
            "incremental_prefill_ms": incremental_prefill_ms,
            "semi_prefill_ms": semi_prefill_ms,
            "summary_prefill_ms": summary_prefill_ms,
            "summary_decode_ms": summary_decode_ms,
            "decode_ms": decode_ms,
            "total_ms": sync_total_ms,
        },
        "baseline_ms": baseline_ms,
        "async_overhead_pct": semi_prefill_ms / baseline_ms * 100.0 if baseline_ms else 0.0,
        "sync_overhead_pct": (semi_prefill_ms + summary_prefill_ms + summary_decode_ms) / baseline_ms * 100.0 if baseline_ms else 0.0,
    }


def aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    active = [sample for sample in samples if sample["active"]]
    events = [event for sample in samples for event in sample["events"]]

    def sphase(name: str, key: str) -> float:
        return sum(sample[name].get(key, 0.0) for sample in active)

    phase_async = {
        "full_prefill_ms": sphase("phase_async", "full_prefill_ms"),
        "incremental_prefill_ms": sphase("phase_async", "incremental_prefill_ms"),
        "semi_prefill_ms": sphase("phase_async", "semi_prefill_ms"),
        "decode_ms": sphase("phase_async", "decode_ms"),
    }
    phase_async["total_ms"] = sum(phase_async.values())
    phase_sync = {
        "full_prefill_ms": sphase("phase_sync", "full_prefill_ms"),
        "incremental_prefill_ms": sphase("phase_sync", "incremental_prefill_ms"),
        "semi_prefill_ms": sphase("phase_sync", "semi_prefill_ms"),
        "summary_prefill_ms": sphase("phase_sync", "summary_prefill_ms"),
        "summary_decode_ms": sphase("phase_sync", "summary_decode_ms"),
        "decode_ms": sphase("phase_sync", "decode_ms"),
    }
    phase_sync["total_ms"] = sum(phase_sync.values())
    baseline_ms = sum(sample["baseline_ms"] for sample in active)
    b2c1 = [event["B2_C1"] for event in events]
    c1 = [event["C1"] for event in events]
    b2 = [event["B2"] for event in events]
    multipliers = [sample["context_multiplier"] for sample in active if sample["context_multiplier"]]
    rounds = [sample["rounds"] for sample in active]
    compressions = [sample["n_compressions"] for sample in active]
    return {
        "sample_count": len(samples),
        "active_sample_count": len(active),
        "compression_event_count": len(events),
        "total_rounds": sum(rounds),
        "total_steps": sum(sample["steps"] for sample in active),
        "avg_rounds": mean(rounds) if rounds else 0.0,
        "avg_compressions": mean(compressions) if compressions else 0.0,
        "compression_rate": sum(compressions) / sum(rounds) if sum(rounds) else 0.0,
        "avg_accumulated_context": mean([sample["accumulated_context"] for sample in active]) if active else 0.0,
        "avg_context_multiplier": mean(multipliers) if multipliers else 0.0,
        "min_context_multiplier": min(multipliers) if multipliers else 0.0,
        "max_context_multiplier": max(multipliers) if multipliers else 0.0,
        "avg_delta_median": median([sample["delta_median"] for sample in active if sample["delta_median"]]) if active else 0.0,
        "phase_async": phase_async,
        "phase_sync": phase_sync,
        "baseline_ms": baseline_ms,
        "async_overhead_pct": phase_async["semi_prefill_ms"] / baseline_ms * 100.0 if baseline_ms else 0.0,
        "sync_overhead_pct": (phase_sync["semi_prefill_ms"] + phase_sync["summary_prefill_ms"] + phase_sync["summary_decode_ms"])
        / baseline_ms
        * 100.0
        if baseline_ms
        else 0.0,
        "avg_b2_c1": mean(b2c1) if b2c1 else 0.0,
        "min_b2_c1": min(b2c1) if b2c1 else 0.0,
        "max_b2_c1": max(b2c1) if b2c1 else 0.0,
        "avg_c1": mean(c1) if c1 else 0.0,
        "avg_b2": mean(b2) if b2 else 0.0,
        "avg_semi_prefill_ms": mean([event["semi_prefill_ms"] for event in events]) if events else 0.0,
        "avg_sync_event_ms": mean([event["semi_prefill_ms"] + event["summary_prefill_ms"] + event["summary_decode_ms"] for event in events])
        if events
        else 0.0,
        "summary_decode_share_in_sync_prefill": phase_sync["summary_decode_ms"]
        / max(
            phase_sync["full_prefill_ms"]
            + phase_sync["incremental_prefill_ms"]
            + phase_sync["semi_prefill_ms"]
            + phase_sync["summary_prefill_ms"]
            + phase_sync["summary_decode_ms"],
            1.0,
        )
        * 100.0,
    }


def ensure_dirs(out: Path) -> tuple[Path, Path, Path]:
    charts = out / "charts"
    tables = out / "tables"
    code = out / "code"
    for path in [out, charts, tables, code]:
        path.mkdir(parents=True, exist_ok=True)
    if Path(__file__).exists():
        shutil.copyfile(Path(__file__), code / "generate_cw8000_reference_analysis.py")
    return charts, tables, code


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def active_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [sample for sample in samples if sample["active"]]


def representative_samples(samples: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    active = active_samples(samples)
    by_label = {sample["sample_label"]: sample for sample in active}
    preferred = [by_label[label] for label in ["P0", "P3", "P5"] if label in by_label]
    if preferred:
        for sample in active:
            if sample not in preferred:
                preferred.append(sample)
            if len(preferred) >= limit:
                break
        return preferred[:limit]
    return active[:limit]


def plot_context_length(samples: list[dict[str, Any]], agg: dict[str, Any], charts: Path, title: str) -> None:
    active = active_samples(samples)
    n = len(active)
    cols = 3 if n > 1 else 1
    rows = math.ceil(n / cols) if n else 1
    fig, axes = plt.subplots(rows, cols, figsize=(6.6 * cols, 3.8 * rows), sharey=False)
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, sample in zip(axes_list, active):
        events_by_turn = {int(event["turn"]) for event in sample["events"]}
        turns = [int(item["turn"]) for item in sample["context_series"]]
        prompts = [item["prompt_tokens"] for item in sample["context_series"]]
        colors = [COLORS["full"] if i == 0 else COLORS["semi"] if turn in events_by_turn else COLORS["incr"] for i, turn in enumerate(turns)]
        ax.bar(turns, prompts, color=colors, alpha=0.9)
        threshold = num(sample["config"].get("threshold"))
        context_window = num(sample["config"].get("context_window"))
        if threshold:
            ax.axhline(threshold, color="#FF5252", linestyle="--", linewidth=1.2, label=f"Threshold ({threshold:.0f})")
        if context_window:
            ax.axhline(context_window, color="#B71C1C", linestyle="--", linewidth=1.0, alpha=0.55, label=f"CW ({context_window:.0f})")
        ax.set_title(f"{sample['sample_label']} ({sample['rounds']} turns, {sample['n_compressions']} comp.)")
        ax.set_xlabel("Turn")
        ax.set_ylabel("Input Tokens")
        ax.grid(True, alpha=0.25)
        if threshold or context_window:
            ax.legend(fontsize=8, loc="upper left")
    for ax in axes_list[len(active) :]:
        ax.axis("off")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS["full"], label="Full Prefill (T0)"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["incr"], label="Incremental (cached)"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["semi"], label="Semi-Prefill (compression)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Context Length Over Turns ({title})", fontsize=15, y=1.05)
    savefig(charts / "context_length_over_turns.png")


def plot_context_churn(samples: list[dict[str, Any]], charts: Path) -> None:
    active = active_samples(samples)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for sample in active:
        xs = [item["turn"] for item in sample["churn_series"]]
        ys = [item["churn"] * 100.0 for item in sample["churn_series"]]
        ax.plot(xs, ys, marker="o", linewidth=1.4, markersize=3, label=sample["sample_label"])
    ax.set_title("Context Churn Ratio")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Churn (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    savefig(charts / "context_churn_ratio.png")


def plot_semi_prefill_composition(samples: list[dict[str, Any]], charts: Path) -> None:
    active = [sample for sample in active_samples(samples) if sample["events"]]
    n = len(active)
    cols = 3 if n > 1 else 1
    rows = math.ceil(n / cols) if n else 1
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 3.8 * rows), sharey=False)
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, sample in zip(axes_list, active):
        events = sample["events"]
        labels = [f"T{int(event['turn'])}" for event in events]
        a = [event["A"] for event in events]
        b2 = [event["B2"] for event in events]
        c1 = [event["C1"] for event in events]
        x = list(range(len(labels)))
        ax.bar(x, a, color=COLORS["a"], edgecolor="#888", label="A (system, cached)")
        ax.bar(x, b2, bottom=a, color=COLORS["b2"], label="B2 (summary, re-prefill)")
        ax.bar(x, c1, bottom=[aa + bb for aa, bb in zip(a, b2)], color=COLORS["c1"], label="C1 (recent, re-prefill)")
        ax.set_title(f"{sample['sample_label']} ({len(events)} compressions)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Tokens")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes_list[len(active) :]:
        ax.axis("off")
    fig.suptitle("Semi-Prefill Token Composition: A (cached) vs B2+C1 (re-prefill)", fontsize=14, y=1.02)
    savefig(charts / "semi_prefill_composition.png")


def plot_semi_prefill_spikes(samples: list[dict[str, Any]], agg: dict[str, Any], charts: Path) -> None:
    events = [event for sample in active_samples(samples) for event in sample["events"]]
    labels = [f"{event['sample_label']}-T{int(event['turn'])}" for event in events]
    values = [event["semi_prefill_ms"] for event in events]
    baseline = (agg["avg_delta_median"] * R_PF + C_FIXED) if agg["avg_delta_median"] else 0.0
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.46), 5.2))
    ax.bar(range(len(labels)), values, color=COLORS["semi"], label="Semi-Prefill event")
    if baseline:
        ax.axhline(baseline, color=COLORS["incr"], linestyle="--", linewidth=1.5, label=f"Incremental baseline ({baseline:.0f} ms)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=55, ha="right")
    ax.set_ylabel("Theoretical Prefill Cost (ms)")
    ax.set_title("Semi-Prefill Spikes vs Incremental Baseline")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    savefig(charts / "semi_prefill_spikes.png")


def stacked_bars(ax: Any, labels: list[str], rows: list[dict[str, float]], keys: list[tuple[str, str, str]]) -> None:
    bottom = [0.0] * len(labels)
    for key, name, color in keys:
        values = [row.get(key, 0.0) for row in rows]
        ax.bar(labels, values, bottom=bottom, label=name, color=color)
        bottom = [base + value for base, value in zip(bottom, values)]


def phase_rows(samples: list[dict[str, Any]], mode: str) -> tuple[list[str], list[dict[str, float]]]:
    active = active_samples(samples)
    labels = [sample["sample_label"] for sample in active]
    rows = [{key: value / 1000.0 for key, value in sample[f"phase_{mode}"].items() if key != "total_ms"} for sample in active]
    return labels, rows


def plot_phase_breakdowns(samples: list[dict[str, Any]], agg: dict[str, Any], charts: Path) -> None:
    labels, async_rows = phase_rows(samples, "async")
    keys_async = [
        ("full_prefill_ms", "Full Prefill (T0)", COLORS["full"]),
        ("incremental_prefill_ms", "Incremental Prefill", COLORS["incr"]),
        ("semi_prefill_ms", "Semi-Prefill", COLORS["semi"]),
        ("decode_ms", "Decode", COLORS["decode"]),
    ]
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.7), 5.5))
    stacked_bars(ax, labels, async_rows, keys_async)
    ax.set_title("Phase Breakdown: Prefill / Semi-Prefill / Decode (Prefix Caching)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Theoretical Latency (s)")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.25)
    savefig(charts / "phase_breakdown_stacked.png")

    plot_phase_pies(samples, agg, charts)

    labels, sync_rows = phase_rows(samples, "sync")
    xlabels = []
    rows = []
    for sample, async_row, sync_row in zip(active_samples(samples), async_rows, sync_rows):
        xlabels.extend([sample["sample_label"] + " A", sample["sample_label"] + " S"])
        rows.extend([async_row, sync_row])
    keys_sync = keys_async + [
        ("summary_prefill_ms", "Summary Prefill", COLORS["sum_pf"]),
        ("summary_decode_ms", "Summary Decode", COLORS["sum_dec"]),
    ]
    fig, ax = plt.subplots(figsize=(max(11, len(xlabels) * 0.48), 5.8))
    stacked_bars(ax, xlabels, rows, keys_sync)
    ax.set_title("Phase Breakdown Sync vs Async")
    ax.set_ylabel("Theoretical Latency (s)")
    ax.tick_params(axis="x", rotation=55)
    ax.legend(loc="upper left", ncol=2)
    ax.grid(True, axis="y", alpha=0.25)
    savefig(charts / "phase_breakdown_sync_vs_async.png")


def plot_phase_pies(samples: list[dict[str, Any]], agg: dict[str, Any], charts: Path) -> None:
    selected = representative_samples(samples)
    pies = selected + [{"sample_label": "Aggregate", "phase_async": agg["phase_async"]}]
    fig, axes = plt.subplots(1, len(pies), figsize=(5.2 * len(pies), 4.6))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
    labels = ["Full", "Incr.", "Semi", "Decode"]
    colors = [COLORS["full"], COLORS["incr"], COLORS["semi"], COLORS["decode"]]
    for ax, sample in zip(axes_list, pies):
        phase = sample["phase_async"]
        values = [phase.get("full_prefill_ms", 0), phase.get("incremental_prefill_ms", 0), phase.get("semi_prefill_ms", 0), phase.get("decode_ms", 0)]
        total = sum(values) / 1000.0
        ax.pie(values, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax.set_title(f"{sample['sample_label']}\n({total:.1f}s)")
    fig.suptitle("Phase Pie Charts (Async)", fontsize=14)
    savefig(charts / "phase_pie_charts.png")


def plot_prefill_only(samples: list[dict[str, Any]], agg: dict[str, Any], charts: Path) -> None:
    active = active_samples(samples)
    labels = [sample["sample_label"] for sample in active]
    async_rows = [
        {
            "Full Prefill": sample["phase_async"]["full_prefill_ms"] / 1000.0,
            "Incr. Prefill": sample["phase_async"]["incremental_prefill_ms"] / 1000.0,
            "Semi-Prefill": sample["phase_async"]["semi_prefill_ms"] / 1000.0,
        }
        for sample in active
    ]
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.7), 5.2))
    stacked_bars(
        ax,
        labels,
        async_rows,
        [("Full Prefill", "Full Prefill", COLORS["full"]), ("Incr. Prefill", "Incr. Prefill", COLORS["incr"]), ("Semi-Prefill", "Semi-Prefill", COLORS["semi"])],
    )
    ax.set_title("Async Prefill-Only Breakdown (Decode Excluded)")
    ax.set_ylabel("Latency (s)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    savefig(charts / "async_prefill_only_bar.png")

    pies = representative_samples(samples) + [{"sample_label": "Aggregate", "phase_async": agg["phase_async"]}]
    fig, axes = plt.subplots(1, len(pies), figsize=(5.2 * len(pies), 4.6))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, sample in zip(axes_list, pies):
        phase = sample["phase_async"]
        values = [phase["full_prefill_ms"], phase["incremental_prefill_ms"], phase["semi_prefill_ms"]]
        labels2 = ["Full Prefill", "Incr. Prefill", "Semi-Prefill"]
        ax.pie(values, labels=labels2, colors=[COLORS["full"], COLORS["incr"], COLORS["semi"]], autopct="%1.1f%%", startangle=90)
        ax.set_title(f"{sample['sample_label']}\n({sum(values) / 1000.0:.1f}s)")
    fig.suptitle("Async Prefill-Only Distribution (Decode Excluded)", fontsize=14)
    savefig(charts / "async_prefill_only_pie.png")

    async_prefill = {
        "Full Prefill": agg["phase_async"]["full_prefill_ms"] / 1000.0,
        "Incr. Prefill": agg["phase_async"]["incremental_prefill_ms"] / 1000.0,
        "Semi-Prefill": agg["phase_async"]["semi_prefill_ms"] / 1000.0,
        "Summary Prefill": 0.0,
        "Summary Decode": 0.0,
    }
    sync_prefill = {
        "Full Prefill": agg["phase_sync"]["full_prefill_ms"] / 1000.0,
        "Incr. Prefill": agg["phase_sync"]["incremental_prefill_ms"] / 1000.0,
        "Semi-Prefill": agg["phase_sync"]["semi_prefill_ms"] / 1000.0,
        "Summary Prefill": agg["phase_sync"]["summary_prefill_ms"] / 1000.0,
        "Summary Decode": agg["phase_sync"]["summary_decode_ms"] / 1000.0,
    }
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    stacked_bars(
        ax,
        ["Async", "Sync"],
        [async_prefill, sync_prefill],
        [
            ("Full Prefill", "Full Prefill", COLORS["full"]),
            ("Incr. Prefill", "Incr. Prefill", COLORS["incr"]),
            ("Semi-Prefill", "Semi-Prefill", COLORS["semi"]),
            ("Summary Prefill", "Summary Prefill", COLORS["sum_pf"]),
            ("Summary Decode", "Summary Decode", COLORS["sum_dec"]),
        ],
    )
    ax.set_title("Prefill-Only Sync vs Async")
    ax.set_ylabel("Latency (s)")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.25)
    savefig(charts / "prefill_only_sync_vs_async_bar.png")

    plot_sync_async_pies(async_prefill, sync_prefill, charts)


def plot_sync_async_pies(async_prefill: dict[str, float], sync_prefill: dict[str, float], charts: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    labels = list(async_prefill)
    colors = [COLORS["full"], COLORS["incr"], COLORS["semi"], COLORS["sum_pf"], COLORS["sum_dec"]]
    axes[0].pie(list(async_prefill.values()), labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    axes[0].set_title(f"Async ({sum(async_prefill.values()):.1f}s)")
    axes[1].pie(list(sync_prefill.values()), labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    axes[1].set_title(f"Sync ({sum(sync_prefill.values()):.1f}s)")
    fig.suptitle("Prefill-Only Breakdown")
    savefig(charts / "sync_vs_async_prefill_only.png")


def plot_sync_async_summary(samples: list[dict[str, Any]], agg: dict[str, Any], charts: Path) -> None:
    active = active_samples(samples)
    labels = [sample["sample_label"] for sample in active] + ["Aggregate"]
    async_totals = [sample["phase_async"]["total_ms"] / 1000.0 for sample in active] + [agg["phase_async"]["total_ms"] / 1000.0]
    sync_totals = [sample["phase_sync"]["total_ms"] / 1000.0 for sample in active] + [agg["phase_sync"]["total_ms"] / 1000.0]
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.7), 5.2))
    width = 0.38
    x = list(range(len(labels)))
    ax.bar([i - width / 2 for i in x], async_totals, width=width, color="#64B5F6", label="Async")
    ax.bar([i + width / 2 for i in x], sync_totals, width=width, color="#EF5350", label="Sync")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Total theoretical latency (s)")
    ax.set_title("Sync vs Async Stacked")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    savefig(charts / "sync_vs_async_stacked.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    phase_labels = ["Full", "Incr.", "Semi", "Decode", "SumPF", "SumDec"]
    async_values = [
        agg["phase_async"]["full_prefill_ms"],
        agg["phase_async"]["incremental_prefill_ms"],
        agg["phase_async"]["semi_prefill_ms"],
        agg["phase_async"]["decode_ms"],
        0,
        0,
    ]
    sync_values = [
        agg["phase_sync"]["full_prefill_ms"],
        agg["phase_sync"]["incremental_prefill_ms"],
        agg["phase_sync"]["semi_prefill_ms"],
        agg["phase_sync"]["decode_ms"],
        agg["phase_sync"]["summary_prefill_ms"],
        agg["phase_sync"]["summary_decode_ms"],
    ]
    colors = [COLORS["full"], COLORS["incr"], COLORS["semi"], COLORS["decode"], COLORS["sum_pf"], COLORS["sum_dec"]]
    axes[0].pie(async_values, labels=phase_labels, colors=colors, autopct="%1.1f%%", startangle=90)
    axes[0].set_title("Async")
    axes[1].pie(sync_values, labels=phase_labels, colors=colors, autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Sync")
    fig.suptitle("Sync vs Async Pie")
    savefig(charts / "sync_vs_async_pie.png")

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.bar(["Async", "Sync"], [agg["async_overhead_pct"], agg["sync_overhead_pct"]], color=["#64B5F6", "#EF5350"])
    ax.set_ylabel("Compression overhead vs no-compression baseline (%)")
    ax.set_title("Overhead Async vs Sync")
    ax.grid(True, axis="y", alpha=0.25)
    savefig(charts / "overhead_async_vs_sync.png")


def plot_event_costs(samples: list[dict[str, Any]], charts: Path) -> None:
    events = [event for sample in active_samples(samples) for event in sample["events"]]
    labels = [f"{event['sample_label']}-T{int(event['turn'])}" for event in events]
    if not events:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No compression events", ha="center", va="center")
        ax.axis("off")
        savefig(charts / "per_event_sync_breakdown.png")
        shutil.copyfile(charts / "per_event_sync_breakdown.png", charts / "per_event_sync_vs_async.png")
        return
    rows = [
        {
            "Semi-Prefill": event["semi_prefill_ms"] / 1000.0,
            "Summary Prefill": event["summary_prefill_ms"] / 1000.0,
            "Summary Decode": event["summary_decode_ms"] / 1000.0,
        }
        for event in events
    ]
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.48), 5.4))
    stacked_bars(
        ax,
        labels,
        rows,
        [("Semi-Prefill", "Semi-Prefill", COLORS["semi"]), ("Summary Prefill", "Summary Prefill", COLORS["sum_pf"]), ("Summary Decode", "Summary Decode", COLORS["sum_dec"])],
    )
    ax.set_title("Per-Event Sync Breakdown")
    ax.set_ylabel("Latency (s)")
    ax.tick_params(axis="x", rotation=55)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    savefig(charts / "per_event_sync_breakdown.png")

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.48), 5.4))
    x = list(range(len(labels)))
    width = 0.36
    async_values = [event["semi_prefill_ms"] / 1000.0 for event in events]
    sync_values = [(event["semi_prefill_ms"] + event["summary_prefill_ms"] + event["summary_decode_ms"]) / 1000.0 for event in events]
    ax.bar([i - width / 2 for i in x], async_values, width=width, color="#64B5F6", label="Async (SP only)")
    ax.bar([i + width / 2 for i in x], sync_values, width=width, color="#EF5350", label="Sync (SP+SumPF+SumDec)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right")
    ax.set_ylabel("Compression event cost (s)")
    ax.set_title("Per-Event Sync vs Async")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    savefig(charts / "per_event_sync_vs_async.png")


def plot_per_turn_latency(samples: list[dict[str, Any]], charts: Path) -> None:
    for sample in representative_samples(samples):
        turns = [int(item["turn"]) for item in sample["context_series"]]
        event_by_turn = defaultdict(float)
        sync_extra_by_turn = defaultdict(float)
        for event in sample["events"]:
            event_by_turn[int(event["turn"])] += event["semi_prefill_ms"]
            sync_extra_by_turn[int(event["turn"])] += event["summary_prefill_ms"] + event["summary_decode_ms"]
        full = []
        incr = []
        semi = []
        decode = []
        previous = None
        output_by_turn = defaultdict(float)
        # Approximate decode per turn by distributing output evenly over recorded context turns when raw timing is collapsed.
        avg_decode = sample["phase_async"]["decode_ms"] / max(len(turns), 1)
        for i, item in enumerate(sample["context_series"]):
            turn = int(item["turn"])
            prompt = item["prompt_tokens"]
            full.append((prompt * R_PF + C_FIXED) if i == 0 else 0.0)
            delta = max(prompt - (previous or prompt), 0.0) if i > 0 and turn not in event_by_turn else 0.0
            incr.append(delta * R_PF + C_FIXED if i > 0 and turn not in event_by_turn else 0.0)
            semi.append(event_by_turn[turn])
            decode.append(output_by_turn[turn] or avg_decode)
            previous = prompt if prompt > 0 else previous
        rows = [
            {"Full": a / 1000.0, "Incr": b / 1000.0, "Semi": c / 1000.0, "Decode": d / 1000.0}
            for a, b, c, d in zip(full, incr, semi, decode)
        ]
        fig, ax = plt.subplots(figsize=(max(9, len(turns) * 0.42), 4.8))
        stacked_bars(ax, [str(turn) for turn in turns], rows, [("Full", "Full Prefill", COLORS["full"]), ("Incr", "Incr. Prefill", COLORS["incr"]), ("Semi", "Semi-Prefill", COLORS["semi"]), ("Decode", "Decode", COLORS["decode"])])
        ax.set_title(f"Per-Turn Latency {sample['sample_label']} (Async)")
        ax.set_xlabel("Turn")
        ax.set_ylabel("Latency (s)")
        ax.legend(loc="upper left")
        ax.grid(True, axis="y", alpha=0.25)
        savefig(charts / f"per_turn_latency_{sample['sample_label']}.png")

        rows_sync = []
        rows_async = []
        for row, turn in zip(rows, turns):
            rows_async.append(row)
            rows_sync.append({**row, "Sum": sync_extra_by_turn[turn] / 1000.0})
        fig, axes = plt.subplots(2, 1, figsize=(max(9, len(turns) * 0.42), 7), sharex=True)
        labels = [str(turn) for turn in turns]
        stacked_bars(axes[0], labels, rows_async, [("Full", "Full", COLORS["full"]), ("Incr", "Incr", COLORS["incr"]), ("Semi", "Semi", COLORS["semi"]), ("Decode", "Decode", COLORS["decode"])])
        axes[0].set_title("Async")
        stacked_bars(axes[1], labels, rows_sync, [("Full", "Full", COLORS["full"]), ("Incr", "Incr", COLORS["incr"]), ("Semi", "Semi", COLORS["semi"]), ("Sum", "Summary", COLORS["sum_dec"]), ("Decode", "Decode", COLORS["decode"])])
        axes[1].set_title("Sync")
        axes[1].set_xlabel("Turn")
        for ax in axes:
            ax.set_ylabel("Latency (s)")
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend(loc="upper left", ncol=3, fontsize=8)
        fig.suptitle(f"Per-Turn Sync vs Async {sample['sample_label']}")
        savefig(charts / f"per_turn_latency_sync_async_{sample['sample_label']}.png")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False) if isinstance(row.get(key), (dict, list)) else row.get(key) for key in keys})


def fmt_i(value: Any) -> str:
    parsed = num(value)
    return "-" if parsed is None else f"{parsed:,.0f}"


def fmt_f(value: Any, digits: int = 1) -> str:
    parsed = num(value)
    return "-" if parsed is None else f"{parsed:,.{digits}f}"


def fmt_pct(value: Any, digits: int = 1) -> str:
    parsed = num(value)
    return "-" if parsed is None else f"{parsed:.{digits}f}%"


def table(rows: list[dict[str, Any]], cols: list[tuple[str, str, str]], limit: int | None = None) -> str:
    rows2 = rows[:limit] if limit else rows
    if not rows2:
        return "_无可用记录。_\n"
    lines = ["| " + " | ".join(name for _, name, _ in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in rows2:
        cells = []
        for key, _, kind in cols:
            value = row.get(key)
            if kind == "int":
                cells.append(fmt_i(value))
            elif kind == "float":
                cells.append(fmt_f(value, 2))
            elif kind == "pct":
                cells.append(fmt_pct(value, 1))
            else:
                cells.append("-" if value is None else str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    if limit and len(rows) > limit:
        lines.append(f"\n_仅展示前 {limit} 条；完整表格见 tables/*.csv。_")
    return "\n".join(lines) + "\n"


def render_report(dataset: Dataset, samples: list[dict[str, Any]], agg: dict[str, Any]) -> str:
    active = active_samples(samples)
    events = [event for sample in active for event in sample["events"]]
    first_cfg = active[0]["config"] if active else dataset.fallback_config
    sample_rows = [
        {
            "sample": sample["sample_label"],
            "run": sample["run_name"],
            "turns": sample["rounds"],
            "compressions": sample["n_compressions"],
            "compression_rate": sample["compression_rate"] * 100.0,
            "delta_median": sample["delta_median"],
            "ctx_max": sample["max_context"],
            "acc_ctx": sample["accumulated_context"],
            "ctx_mult": sample["context_multiplier"],
        }
        for sample in active
    ]
    event_rows = [
        {
            "sample": event["sample_label"],
            "event": event["event"],
            "turn": event["turn"],
            "A": event["A"],
            "B1": event["B1"],
            "B2": event["B2"],
            "C1": event["C1"],
            "B2_C1": event["B2_C1"],
            "sp_ms": event["semi_prefill_ms"],
            "sync_ms": event["semi_prefill_ms"] + event["summary_prefill_ms"] + event["summary_decode_ms"],
        }
        for event in events
    ]
    phase_rows = []
    for sample in active:
        a = sample["phase_async"]
        s = sample["phase_sync"]
        phase_rows.append(
            {
                "sample": sample["sample_label"],
                "async_s": a["total_ms"] / 1000.0,
                "sync_s": s["total_ms"] / 1000.0,
                "full_ms": a["full_prefill_ms"],
                "incr_ms": a["incremental_prefill_ms"],
                "semi_ms": a["semi_prefill_ms"],
                "decode_ms": a["decode_ms"],
                "sumdec_ms": s["summary_decode_ms"],
                "overhead_sync": sample["sync_overhead_pct"],
            }
        )

    sample_cols = [
        ("sample", "样本", "str"),
        ("turns", "总轮数", "int"),
        ("compressions", "压缩次数", "int"),
        ("compression_rate", "压缩率", "pct"),
        ("delta_median", "δ 中位数", "int"),
        ("ctx_max", "ctx max", "int"),
        ("acc_ctx", "累计 ctx", "int"),
        ("ctx_mult", "累计/首轮", "float"),
    ]
    event_cols = [
        ("sample", "样本", "str"),
        ("event", "事件", "int"),
        ("turn", "Turn", "int"),
        ("A", "A", "int"),
        ("B2", "B2", "int"),
        ("C1", "C1", "int"),
        ("B2_C1", "B2+C1", "int"),
        ("sp_ms", "Async SP ms", "int"),
        ("sync_ms", "Sync event ms", "int"),
    ]
    phase_cols = [
        ("sample", "样本", "str"),
        ("async_s", "Async s", "float"),
        ("sync_s", "Sync s", "float"),
        ("full_ms", "Full ms", "int"),
        ("incr_ms", "Incr ms", "int"),
        ("semi_ms", "SP ms", "int"),
        ("decode_ms", "Decode ms", "int"),
        ("sumdec_ms", "SumDec ms", "int"),
        ("overhead_sync", "Sync overhead", "pct"),
    ]
    sync_increase = (agg["phase_sync"]["total_ms"] / agg["phase_async"]["total_ms"] - 1.0) * 100.0 if agg["phase_async"]["total_ms"] else 0.0
    async_prefill_total = (
        agg["phase_async"]["full_prefill_ms"] + agg["phase_async"]["incremental_prefill_ms"] + agg["phase_async"]["semi_prefill_ms"]
    )
    sync_prefill_total = (
        agg["phase_sync"]["full_prefill_ms"]
        + agg["phase_sync"]["incremental_prefill_ms"]
        + agg["phase_sync"]["semi_prefill_ms"]
        + agg["phase_sync"]["summary_prefill_ms"]
        + agg["phase_sync"]["summary_decode_ms"]
    )
    sp_prefill_share = agg["phase_async"]["semi_prefill_ms"] / async_prefill_total * 100.0 if async_prefill_total else 0.0
    incr_prefill_share = agg["phase_async"]["incremental_prefill_ms"] / async_prefill_total * 100.0 if async_prefill_total else 0.0
    reps = representative_samples(samples)
    return f"""# Semi-Prefill Overhead 分析报告（{dataset.slug}，Prefix Caching 场景）

## 1 实验概述

| 参数 | 值 |
|------|-----|
| 模型 | {first_cfg.get('model', 'Llama-3.3-70B-Instruct')} |
| 上下文窗口 (cw) | {fmt_i(first_cfg.get('context_window'))} tokens |
| 压缩阈值 (threshold) | {fmt_i(first_cfg.get('threshold'))} tokens |
| 保留最近 (keep_recent) | {fmt_i(first_cfg.get('keep_recent_tokens'))} tokens |
| 摘要上限 (summary_max) | {fmt_i(first_cfg.get('summary_max_tokens'))} tokens |
| 预留 (reserve) | {fmt_i(first_cfg.get('reserve_tokens'))} tokens |
| 有效样本数 | {fmt_i(agg['active_sample_count'])} / {fmt_i(agg['sample_count'])} |
| 配置来源 | {first_cfg.get('source', 'unknown')} |

**硬件参数（理论模型，与 cw8000 参考报告保持一致）**

| 符号 | 含义 | 值 | 来源 |
|------|------|-----|------|
| $r_{{pf}}$ | Prefill 速率 | {R_PF} ms/tok | cw8000 参考报告 |
| $r_{{dec}}$ | Decode 速率 | {R_DEC} ms/tok | cw8000 参考报告 |
| $c_{{fix}}$ | 请求固定开销 | {C_FIXED:.0f} ms | cw8000 参考报告 |

**异步压缩 vs 同步压缩**

| 模式 | 含义 | 关键路径上的压缩成本 |
|------|------|-------------------|
| **Async（异步摘要）** | 摘要在后台/空闲时预生成，不阻塞用户请求 | 仅 Semi-Prefill：$(B_2+C_1) \\times r_{{pf}} + c_{{fix}}$ |
| **Sync（同步摘要）** | 摘要在压缩触发时同步生成，用户必须等待 | Summary Prefill $B_1 \\times r_{{pf}}$ + Summary Decode $B_2 \\times r_{{dec}}$ + Semi-Prefill |

> {dataset.caveat}

---

## 2 Agent 工作负载统计

### 2.1 基本统计

| 指标 | 值 |
|------|-----|
| 总轮数 | {fmt_i(agg['total_rounds'])} |
| LLM calls | {fmt_i(agg['total_steps'])} |
| 压缩次数 | {fmt_i(agg['compression_event_count'])} |
| 压缩率 | {fmt_pct(agg['compression_rate'] * 100.0, 1)} |
| 平均压缩/请求 | {fmt_f(agg['avg_compressions'], 2)} |
| 平均 rounds/请求 | {fmt_f(agg['avg_rounds'], 2)} |
| 平均累计上下文 | {fmt_i(agg['avg_accumulated_context'])} tokens |
| 累计/首轮范围 | {fmt_f(agg['min_context_multiplier'], 1)}×–{fmt_f(agg['max_context_multiplier'], 1)}× |

{table(sample_rows, sample_cols, limit=30)}

### 2.2 每轮新增 tokens（δ）

跨样本 δ 中位数约 **{fmt_i(agg['avg_delta_median'])} tokens/轮**。该值用于估计普通增量 prefill 基线：

$$\delta_{{med}} \\times r_{{pf}} + c_{{fix}} \\approx {fmt_i(agg['avg_delta_median'])} \\times {R_PF} + {C_FIXED:.0f} = {fmt_i(agg['avg_delta_median'] * R_PF + C_FIXED)}\\text{{ ms/轮}}$$

### 2.3 上下文长度分布

![Context Length Over Turns](charts/context_length_over_turns.png)

**特征**：图中蓝色为首轮 Full Prefill，绿色为 Prefix Cache 命中的增量轮，橙色为压缩后的 Semi-Prefill 轮。多轮 agentic request 平均跨越 **{fmt_f(agg['avg_rounds'], 2)} rounds**，平均累计上下文为 **{fmt_f(agg['avg_accumulated_context'] / 1000.0, 1)}K tokens**，是单轮请求的 **{fmt_f(agg['min_context_multiplier'], 1)}×–{fmt_f(agg['max_context_multiplier'], 1)}×**。

### 2.4 Context Churn Ratio

![Context Churn Ratio](charts/context_churn_ratio.png)

增量轮 churn 由新增 token 占当前上下文比例估计；压缩轮 churn 由 $(B_2+C_1)/L_{{post}}$ 估计。压缩轮通常出现明显尖峰，因为 B₂ 重写导致 C₁ 也需要重新 prefill。

---

## 3 Semi-Prefill 触发统计

### 3.1 每次压缩的 token 段分布

![Semi-Prefill Composition](charts/semi_prefill_composition.png)

{table(event_rows, event_cols, limit=50)}

**统计汇总**：

| 指标 | 值 |
|------|-----|
| B₂+C₁ 均值 | {fmt_i(agg['avg_b2_c1'])} tokens |
| B₂+C₁ 范围 | {fmt_i(agg['min_b2_c1'])} – {fmt_i(agg['max_b2_c1'])} tokens |
| C₁ 均值 | {fmt_i(agg['avg_c1'])} tokens |
| B₂ 均值 | {fmt_i(agg['avg_b2'])} tokens |
| Semi-prefill 均值 | {fmt_i(agg['avg_semi_prefill_ms'])} ms/event |
| Sync 单事件均值 | {fmt_i(agg['avg_sync_event_ms'])} ms/event |

### 3.2 Semi-Prefill 尖峰 vs 增量基线

![Semi-Prefill Spikes](charts/semi_prefill_spikes.png)

Async 下每次压缩事件的 semi-prefill 平均为 **{fmt_i(agg['avg_semi_prefill_ms'])} ms**，约为普通增量 prefill 基线的 **{fmt_f(agg['avg_semi_prefill_ms'] / max(agg['avg_delta_median'] * R_PF + C_FIXED, 1), 1)}×**。

---

## 4 执行时间分解（Prefix Caching 理论模型）

### 4.1 公式

| 轮类型 | Prefill 成本 | Decode 成本 | 备注 |
|--------|-------------|-------------|------|
| T0（Full Prefill） | $L_0 \\times r_{{pf}} + c_{{fix}}$ | $O \\times r_{{dec}}$ | 冷启动 |
| 增量轮（Cached） | $\delta \\times r_{{pf}} + c_{{fix}}$ | $O \\times r_{{dec}}$ | Prefix Cache 命中 |
| 压缩轮（Async） | $(B_2 + C_1) \\times r_{{pf}} + c_{{fix}}$ | $O \\times r_{{dec}}$ | 仅 Semi-Prefill |
| 压缩轮（Sync） | $B_1 \\times r_{{pf}} + B_2 \\times r_{{dec}} + (B_2+C_1) \\times r_{{pf}} + c_{{fix}}$ | $O \\times r_{{dec}}$ | 含摘要生成 |

### 4.2 总时间分解

![Phase Breakdown](charts/phase_breakdown_stacked.png)

{table(phase_rows, phase_cols, limit=30)}

![Phase Pie Charts](charts/phase_pie_charts.png)

### 4.3 Async vs Sync 对比

![Phase Breakdown Sync vs Async](charts/phase_breakdown_sync_vs_async.png)

同步摘要使理论总延迟从 **{fmt_f(agg['phase_async']['total_ms'] / 1000.0, 1)}s** 增加到 **{fmt_f(agg['phase_sync']['total_ms'] / 1000.0, 1)}s**，增幅 **{fmt_pct(sync_increase, 1)}**。新增成本主要来自 Summary Decode。

### 4.4 Prefill-Only 分解（排除 Decode）

![Async Prefill-Only Bar](charts/async_prefill_only_bar.png)

![Async Prefill-Only Pie](charts/async_prefill_only_pie.png)

**关键发现（Async）**：Semi-Prefill 只发生在 **{fmt_pct(agg['compression_rate'] * 100.0, 1)}** 的 rounds，却消耗了 **{fmt_pct(sp_prefill_share, 1)}** 的 Prefill-only 计算量；Incremental Prefill 占 **{fmt_pct(incr_prefill_share, 1)}**。

![Prefill-Only Sync vs Async](charts/prefill_only_sync_vs_async_bar.png)

Sync 下 Prefill-only 总量从 **{fmt_f(async_prefill_total / 1000.0, 1)}s** 增至 **{fmt_f(sync_prefill_total / 1000.0, 1)}s**，倍率 **{fmt_f(sync_prefill_total / max(async_prefill_total, 1), 1)}×**，其中 Summary Decode 占 Sync Prefill-only 的 **{fmt_pct(agg['summary_decode_share_in_sync_prefill'], 1)}**。

### 4.5 Per-Turn Latency Breakdown

前 3 个有效样本的逐轮图如下，文件命名与参考报告保持一致：

{chr(10).join([f"![Per-Turn Latency {sample['sample_label']}](charts/per_turn_latency_{sample['sample_label']}.png)" for sample in reps])}

{chr(10).join([f"![Per-Turn Sync vs Async {sample['sample_label']}](charts/per_turn_latency_sync_async_{sample['sample_label']}.png)" for sample in reps])}

---

## 5 同步 vs 异步摘要：时间占比对比

### 5.1 摘要生成的成本分解

![Per-Event Sync Breakdown](charts/per_event_sync_breakdown.png)

![Per-Event Sync vs Async](charts/per_event_sync_vs_async.png)

Async 下单次压缩事件平均只承担 **{fmt_f(agg['avg_semi_prefill_ms'] / 1000.0, 2)}s** 的 Semi-Prefill；Sync 下平均膨胀至 **{fmt_f(agg['avg_sync_event_ms'] / 1000.0, 2)}s**。

### 5.2 总延迟与占比

![Sync vs Async Stacked](charts/sync_vs_async_stacked.png)

![Sync vs Async Pie](charts/sync_vs_async_pie.png)

![Prefill-Only Breakdown](charts/sync_vs_async_prefill_only.png)

### 5.3 Overhead 对比

![Overhead Async vs Sync](charts/overhead_async_vs_sync.png)

| 场景 | 压缩 Overhead |
|------|-------------|
| Async | **{fmt_pct(agg['async_overhead_pct'], 2)}** |
| Sync | **{fmt_pct(agg['sync_overhead_pct'], 2)}** |

---

## 6 核心结论

| 结论 | 数据支撑 |
|------|----------|
| 单个 agentic request 平均触发 {fmt_f(agg['avg_compressions'], 2)} 次压缩 | {fmt_i(agg['compression_event_count'])} events / {fmt_i(agg['active_sample_count'])} active samples |
| 平均跨越 {fmt_f(agg['avg_rounds'], 2)} rounds | timing/summary 统计 |
| 平均累计上下文 {fmt_f(agg['avg_accumulated_context'] / 1000.0, 1)}K tokens | sum(prompt_tokens) |
| 累计上下文是单轮的 {fmt_f(agg['min_context_multiplier'], 1)}×–{fmt_f(agg['max_context_multiplier'], 1)}× | accumulated / first prompt |
| Async 压缩 overhead 为 {fmt_pct(agg['async_overhead_pct'], 2)} | 仅 Semi-Prefill 进关键路径 |
| Sync 压缩 overhead 为 {fmt_pct(agg['sync_overhead_pct'], 2)} | Summary Decode 进入关键路径 |

## 附录

- 生成脚本：`code/generate_cw8000_reference_analysis.py`
- JSON 结果：`analysis_results.json`
- 表格目录：`tables/`
- 图表目录：`charts/`
"""


def write_outputs(dataset: Dataset) -> dict[str, Any]:
    charts, tables, _ = ensure_dirs(dataset.output_dir)
    summaries_cache: dict[Path, dict[str, dict[str, Any]]] = {}
    samples = [build_sample(source, dataset, summaries_cache) for source in discover(dataset)]
    agg = aggregate(samples)
    events = [event for sample in samples for event in sample["events"]]
    sample_table = [
        {key: value for key, value in sample.items() if key not in {"context_series", "churn_series", "events", "phase_async", "phase_sync"}}
        for sample in samples
    ]
    phase_table = [
        {
            "sample_label": sample["sample_label"],
            "async_total_ms": sample["phase_async"]["total_ms"],
            "sync_total_ms": sample["phase_sync"]["total_ms"],
            **{f"async_{key}": value for key, value in sample["phase_async"].items()},
            **{f"sync_{key}": value for key, value in sample["phase_sync"].items()},
            "async_overhead_pct": sample["async_overhead_pct"],
            "sync_overhead_pct": sample["sync_overhead_pct"],
        }
        for sample in samples
    ]
    write_csv(tables / "workload_stats.csv", sample_table)
    write_csv(tables / "semi_prefill_events.csv", events)
    write_csv(tables / "phase_breakdown_sync_async.csv", phase_table)

    analysis = {"dataset": dataset.__dict__ | {"data_root": str(dataset.data_root), "output_dir": str(dataset.output_dir)}, "config": {"r_pf": R_PF, "r_dec": R_DEC, "c_fixed": C_FIXED}, "aggregate": agg, "samples": samples, "events": events}
    (dataset.output_dir / "analysis_results.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_context_length(samples, agg, charts, dataset.slug)
    plot_context_churn(samples, charts)
    plot_semi_prefill_composition(samples, charts)
    plot_semi_prefill_spikes(samples, agg, charts)
    plot_phase_breakdowns(samples, agg, charts)
    plot_prefill_only(samples, agg, charts)
    plot_sync_async_summary(samples, agg, charts)
    plot_event_costs(samples, charts)
    plot_per_turn_latency(samples, charts)

    report = render_report(dataset, samples, agg)
    (dataset.output_dir / dataset.report_name).write_text(report, encoding="utf-8")
    return analysis


def main() -> None:
    summary = []
    for dataset in DATASETS:
        analysis = write_outputs(dataset)
        agg = analysis["aggregate"]
        summary.append(
            {
                "slug": dataset.slug,
                "output_dir": str(dataset.output_dir),
                "active_samples": agg["active_sample_count"],
                "events": agg["compression_event_count"],
                "async_prefill_only_pie": str(dataset.output_dir / "charts/async_prefill_only_pie.png"),
                "context_length_over_turns": str(dataset.output_dir / "charts/context_length_over_turns.png"),
                "avg_compressions": round(agg["avg_compressions"], 3),
                "avg_rounds": round(agg["avg_rounds"], 3),
                "async_overhead_pct": round(agg["async_overhead_pct"], 3),
                "sync_overhead_pct": round(agg["sync_overhead_pct"], 3),
            }
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()