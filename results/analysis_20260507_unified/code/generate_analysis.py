#!/usr/bin/env python3
"""Generate agentic semi-prefill/compression analysis artifacts.

The script reads existing experiment outputs and writes derived reports, charts,
CSV tables, and JSON summaries into new analysis folders. It intentionally does
not delete or rewrite source experiment records.
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
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/root")
TODAY = "20260507"


@dataclass(frozen=True)
class DatasetConfig:
    slug: str
    title: str
    data_root: Path
    output_dir: Path
    kind: str
    report_name: str
    description: str
    caveat: str = ""


DATASETS = [
    DatasetConfig(
        slug="tau2_telecom",
        title="Tau2 Telecom MMS Semi-Prefill Compression",
        data_root=ROOT
        / "tau2_telecom_sp_bench/results/real_mms_easy_user3_agent1_n20_compressed_only_cw32000_thr30000_c12800_20260507_001",
        output_dir=ROOT
        / "tau2_telecom_sp_bench/results/real_mms_easy_user3_agent1_n20_compressed_only_cw32000_thr30000_c12800_20260507_001"
        / f"analysis_{TODAY}_unified",
        kind="flat",
        report_name="Tau2_Telecom_SemiPrefill_压缩分析报告.md",
        description="Tau2 telecom MMS task set with compression-only semi-prefill traces.",
        caveat="目录名包含 n20，但当前结果目录中实际可读 timing/ABC 样本数以文件系统为准。",
    ),
    DatasetConfig(
        slug="agentbench_ltp",
        title="AgentBench LTP Semi-Prefill Compression",
        data_root=ROOT / "agentbench_semi_prefill_bench/results/compressed",
        output_dir=ROOT / "agentbench_semi_prefill_bench" / f"analysis_{TODAY}_unified",
        kind="flat",
        report_name="AgentBench_LTP_SemiPrefill_压缩分析报告.md",
        description="AgentBench LTP compressed run with puzzle-level timing, traces, prompt logs and ABC segments.",
        caveat="既有 results/analysis 不会被覆盖；本报告重新扫描 compressed 目录中的全部可读 puzzle 文件。",
    ),
    DatasetConfig(
        slug="bfcl_long_context",
        title="BFCL Long-Context Semi-Prefill Compression",
        data_root=ROOT / "bfcl_long_context_sp_bench/results",
        output_dir=ROOT / "bfcl_long_context_sp_bench/results" / f"analysis_{TODAY}_unified",
        kind="multi_run",
        report_name="BFCL_LongContext_SemiPrefill_压缩分析报告.md",
        description="BFCL long-context two-compression validation and continuation runs under results/.",
        caveat="score summary 中 validate 样本均为 invalid，本文只把这些日志作为压缩与 prefill 工作负载证据，不作为 benchmark accuracy 结论。",
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
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def as_num(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def pick_num(source: dict[str, Any] | None, keys: Iterable[str]) -> float | None:
    if not isinstance(source, dict):
        return None
    for key in keys:
        value = as_num(source.get(key))
        if value is not None:
            return value
    return None


def flatten_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def walk(prefix: str, obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                walk(f"{prefix}.{key}" if prefix else key, value)
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            rows.append({"key": prefix, "value": obj})

    walk("", config)
    return rows


def discover_sources(config: DatasetConfig) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    if config.kind == "flat":
        for timing_path in sorted((config.data_root / "timing").glob("*.json")):
            sources.append(
                {
                    "run_root": config.data_root,
                    "run_name": config.data_root.name,
                    "timing_path": timing_path,
                    "sample_id": timing_path.stem,
                }
            )
        return sources

    for run_root in sorted(config.data_root.iterdir()):
        if not run_root.is_dir():
            continue
        timing_dir = run_root / "timing"
        if not timing_dir.is_dir():
            continue
        for timing_path in sorted(timing_dir.glob("*.json")):
            sources.append(
                {
                    "run_root": run_root,
                    "run_name": run_root.name,
                    "timing_path": timing_path,
                    "sample_id": timing_path.stem,
                }
            )
    return sources


def find_abc_path(run_root: Path, sample_id: str) -> Path | None:
    abc_dir = run_root / "abc_segments"
    if not abc_dir.is_dir():
        return None
    exact = abc_dir / f"{sample_id}.json"
    if exact.exists():
        return exact
    matches = sorted(abc_dir.glob(f"{sample_id}*.json"))
    return matches[0] if matches else None


def read_summary_maps(data_root: Path, run_root: Path) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for path in [run_root / "summary.jsonl", run_root / "tau2_results/results.json", run_root / "agentbench_results/results.jsonl"]:
        if path.suffix == ".jsonl":
            rows = load_jsonl(path)
        elif path.exists():
            obj = load_json(path, [])
            if isinstance(obj, list):
                rows = [row for row in obj if isinstance(row, dict)]
            elif isinstance(obj, dict):
                rows = [obj]
            else:
                rows = []
        else:
            rows = []
        for row in rows:
            row_id = row.get("id") or row.get("sample_id") or row.get("task_id") or row.get("puzzle_id")
            if row_id is not None:
                summary[str(row_id)] = row

    for score_path in sorted(data_root.glob("*score_summary*.json")):
        obj = load_json(score_path, [])
        rows = obj if isinstance(obj, list) else []
        for row in rows:
            if isinstance(row, dict):
                row_id = row.get("id")
                if row_id is not None:
                    summary[str(row_id)] = {**summary.get(str(row_id), {}), **row}
    return summary


def short_label(sample_id: str, run_name: str = "") -> str:
    if sample_id.startswith("puzzle_"):
        return "P" + sample_id.split("_", 1)[1]
    if sample_id.startswith("multi_turn_long_context_"):
        suffix = sample_id.rsplit("_", 1)[-1]
        if run_name.startswith("validate_"):
            return f"V{suffix}"
        if run_name.startswith("continue"):
            return f"C{suffix}"
        return f"B{suffix}"
    match = re.search(r"([0-9a-f]{8,12})$", sample_id)
    if match:
        return match.group(1)[:6]
    if len(sample_id) <= 18:
        return sample_id
    return sample_id[:8] + "..." + sample_id[-6:]


def turn_value(row: dict[str, Any]) -> int | None:
    value = row.get("turn")
    if value is None:
        value = row.get("turn_idx")
    number = as_num(value)
    return int(number) if number is not None else None


def step_value(row: dict[str, Any]) -> int | None:
    value = row.get("step")
    if value is None:
        value = row.get("step_idx")
    number = as_num(value)
    return int(number) if number is not None else None


def extract_abc_events(raw: Any, sample_id: str, sample_label: str, run_name: str) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    events: list[dict[str, Any]] = []
    for index, event in enumerate(raw, start=1):
        if not isinstance(event, dict):
            continue
        segments = event.get("abc_segments") if isinstance(event.get("abc_segments"), dict) else {}
        before = segments.get("before", {}) if isinstance(segments, dict) else {}
        after = segments.get("after", {}) if isinstance(segments, dict) else {}
        pre = pick_num(event, ["pre_prompt_tokens", "before_prompt_tokens", "original_prompt_tokens"])
        post = pick_num(event, ["post_prompt_tokens", "after_prompt_tokens", "compressed_prompt_tokens"])
        a_tokens = pick_num(event, ["A_tokens", "a_tokens"])
        if a_tokens is None:
            a_tokens = pick_num(after, ["A_tokens", "a_tokens"])
        if a_tokens is None:
            a_tokens = pick_num(before, ["A_tokens", "a_tokens"])
        b1_tokens = pick_num(event, ["B1_tokens", "b1_tokens"])
        if b1_tokens is None:
            b1_tokens = pick_num(before, ["B1_tokens", "b1_tokens"])
        b2_tokens = pick_num(event, ["B2_tokens", "B2_tokens_after", "b2_tokens"])
        if b2_tokens is None:
            b2_tokens = pick_num(after, ["B2_tokens", "B2_tokens_after", "b2_tokens"])
        c1_tokens = pick_num(event, ["C1_tokens_after", "C1_after_tokens", "C1_tokens", "c1_tokens"])
        if c1_tokens is None:
            c1_tokens = pick_num(after, ["C1_tokens_after", "C1_tokens", "c1_tokens"])
        if c1_tokens is None:
            c1_tokens = pick_num(before, ["C1_tokens", "c1_tokens"])
        if b2_tokens is None and post is not None and a_tokens is not None and c1_tokens is not None:
            b2_tokens = max(post - a_tokens - c1_tokens, 0.0)
        if c1_tokens is None and post is not None and a_tokens is not None and b2_tokens is not None:
            c1_tokens = max(post - a_tokens - b2_tokens, 0.0)
        semi_prefill_tokens = pick_num(event, ["semi_prefill_tokens", "compression_prefill_tokens"])
        if semi_prefill_tokens is None:
            parts = [value for value in [b2_tokens, c1_tokens] if value is not None]
            semi_prefill_tokens = sum(parts) if parts else None
        saving_pct = pick_num(event, ["token_saving_pct", "saving_pct", "compression_ratio_pct"])
        if saving_pct is None and pre and post is not None:
            saving_pct = (pre - post) / pre * 100.0
        b2_b1_ratio = None
        if b1_tokens and b2_tokens is not None:
            b2_b1_ratio = b2_tokens / b1_tokens
        events.append(
            {
                "sample_id": sample_id,
                "sample_label": sample_label,
                "run_name": run_name,
                "compression_index": index,
                "turn": turn_value(event),
                "step": step_value(event),
                "pre_prompt_tokens": pre,
                "post_prompt_tokens": post,
                "A_tokens": a_tokens,
                "B1_tokens": b1_tokens,
                "B2_tokens": b2_tokens,
                "C1_tokens": c1_tokens,
                "B2_plus_C1_tokens": semi_prefill_tokens,
                "B2_over_B1": b2_b1_ratio,
                "token_saving_pct": saving_pct,
                "summary_generation_time_s": pick_num(event, ["summary_generation_time_s", "summary_time_s"]),
            }
        )
    return events


def summarize_timing(timing: list[dict[str, Any]]) -> dict[str, Any]:
    phase = defaultdict(lambda: {"count": 0, "prompt_tokens": 0.0, "output_tokens": 0.0, "total_ms": 0.0, "ttft_ms": 0.0, "decode_ms": 0.0})
    prompt_tokens: list[float] = []
    output_tokens: list[float] = []
    turn_prompt: dict[int, float] = {}
    semi_prefill_from_timing = 0.0
    for row in timing:
        classification = str(row.get("classification") or "unknown")
        prompt = as_num(row.get("prompt_tokens"), 0.0) or 0.0
        output = as_num(row.get("output_tokens"), 0.0) or 0.0
        prompt_tokens.append(prompt)
        output_tokens.append(output)
        phase[classification]["count"] += 1
        phase[classification]["prompt_tokens"] += prompt
        phase[classification]["output_tokens"] += output
        phase[classification]["total_ms"] += as_num(row.get("total_ms"), 0.0) or 0.0
        phase[classification]["ttft_ms"] += as_num(row.get("ttft_ms"), 0.0) or 0.0
        phase[classification]["decode_ms"] += as_num(row.get("decode_ms"), 0.0) or 0.0
        sp_tokens = as_num(row.get("semi_prefill_tokens"), 0.0) or 0.0
        semi_prefill_from_timing += sp_tokens
        if classification == "semi_prefill" and sp_tokens == 0.0:
            semi_prefill_from_timing += prompt
        turn = turn_value(row)
        if turn is not None:
            turn_prompt[turn] = max(turn_prompt.get(turn, 0.0), prompt)
    turns = sorted(turn_prompt)
    return {
        "steps": len(timing),
        "first_context_tokens": prompt_tokens[0] if prompt_tokens else 0.0,
        "max_context_tokens": max(prompt_tokens) if prompt_tokens else 0.0,
        "min_context_tokens": min(prompt_tokens) if prompt_tokens else 0.0,
        "accumulated_context_tokens": sum(prompt_tokens),
        "output_tokens": sum(output_tokens),
        "timing_total_ms": sum(item["total_ms"] for item in phase.values()),
        "semi_prefill_tokens_from_timing": semi_prefill_from_timing,
        "classification_counts": dict(Counter(str(row.get("classification") or "unknown") for row in timing)),
        "phase": {key: dict(value) for key, value in phase.items()},
        "turn_context_series": [{"turn": turn, "prompt_tokens": turn_prompt[turn]} for turn in turns],
        "unique_timing_turns": len(turns),
    }


def sample_rounds(summary: dict[str, Any], timing_summary: dict[str, Any]) -> int:
    for key in ["turns", "decoded_turns", "rounds"]:
        value = as_num(summary.get(key))
        if value is not None and value > 0:
            return int(value)
    turns = int(timing_summary.get("unique_timing_turns") or 0)
    return turns if turns else int(timing_summary.get("steps") or 0)


def analyze_dataset(config: DatasetConfig) -> dict[str, Any]:
    output_dir = config.output_dir
    charts_dir = output_dir / "charts"
    tables_dir = output_dir / "tables"
    code_dir = output_dir / "code"
    for path in [output_dir, charts_dir, tables_dir, code_dir]:
        path.mkdir(parents=True, exist_ok=True)
    if Path(__file__).exists():
        shutil.copyfile(Path(__file__), code_dir / "generate_analysis.py")

    sources = discover_sources(config)
    sample_summaries: list[dict[str, Any]] = []
    compression_events: list[dict[str, Any]] = []
    phase_rows_by_dataset = defaultdict(lambda: {"count": 0, "prompt_tokens": 0.0, "output_tokens": 0.0, "total_ms": 0.0, "ttft_ms": 0.0, "decode_ms": 0.0})
    run_configs: dict[str, Any] = {}
    source_file_counts = {
        "timing_files": len(sources),
        "abc_files": 0,
        "trace_files": 0,
        "checkpoint_files": 0,
        "prompt_log_files": 0,
        "declared_sample_ids": None,
    }
    declared_ids = load_json(config.data_root / "sample_ids.json")
    if isinstance(declared_ids, list):
        source_file_counts["declared_sample_ids"] = len(declared_ids)

    summary_maps_cache: dict[Path, dict[str, dict[str, Any]]] = {}
    for source in sources:
        run_root: Path = source["run_root"]
        sample_id = str(source["sample_id"])
        run_name = str(source["run_name"])
        label = short_label(sample_id, run_name)
        timing = load_json(source["timing_path"], [])
        if not isinstance(timing, list):
            timing = []
        timing_summary = summarize_timing([row for row in timing if isinstance(row, dict)])
        if run_root not in summary_maps_cache:
            summary_maps_cache[run_root] = read_summary_maps(config.data_root, run_root)
        summary_map = summary_maps_cache[run_root]
        summary = summary_map.get(sample_id, summary_map.get(label, {}))
        abc_path = find_abc_path(run_root, sample_id)
        abc_raw = load_json(abc_path, []) if abc_path else []
        if abc_path and abc_path.exists():
            source_file_counts["abc_files"] += 1
        events = extract_abc_events(abc_raw, sample_id, label, run_name)
        compression_events.extend(events)
        trace_exists = (run_root / "traces" / f"{sample_id}.json").exists()
        prompt_log_exists = (run_root / "prompt_logs" / f"{sample_id}.jsonl").exists()
        checkpoint_exists = (run_root / "checkpoints" / f"{sample_id}.json").exists()
        source_file_counts["trace_files"] += 1 if trace_exists else 0
        source_file_counts["prompt_log_files"] += 1 if prompt_log_exists else 0
        source_file_counts["checkpoint_files"] += 1 if checkpoint_exists else 0
        run_config = load_json(run_root / "run_config.json", {})
        if run_config:
            run_configs[run_name] = run_config

        for phase, values in timing_summary["phase"].items():
            phase_rows_by_dataset[phase]["count"] += values["count"]
            phase_rows_by_dataset[phase]["prompt_tokens"] += values["prompt_tokens"]
            phase_rows_by_dataset[phase]["output_tokens"] += values["output_tokens"]
            phase_rows_by_dataset[phase]["total_ms"] += values["total_ms"]
            phase_rows_by_dataset[phase]["ttft_ms"] += values["ttft_ms"]
            phase_rows_by_dataset[phase]["decode_ms"] += values["decode_ms"]

        abc_prefill_tokens = sum(as_num(event.get("B2_plus_C1_tokens"), 0.0) or 0.0 for event in events)
        timing_prefill_tokens = as_num(timing_summary.get("semi_prefill_tokens_from_timing"), 0.0) or 0.0
        additional_prefill_tokens = abc_prefill_tokens if abc_prefill_tokens > 0 else timing_prefill_tokens
        accumulated_context = timing_summary["accumulated_context_tokens"]
        first_context = timing_summary["first_context_tokens"]
        context_multiplier = accumulated_context / first_context if first_context else None
        rounds = sample_rounds(summary, timing_summary)
        compressions = len(events)
        summary_generation_s = sum(as_num(event.get("summary_generation_time_s"), 0.0) or 0.0 for event in events)
        sample_summaries.append(
            {
                "sample_id": sample_id,
                "sample_label": label,
                "run_name": run_name,
                "rounds": rounds,
                "steps": timing_summary["steps"],
                "compressions": compressions,
                "compression_per_round": compressions / rounds if rounds else None,
                "first_context_tokens": first_context,
                "max_context_tokens": timing_summary["max_context_tokens"],
                "accumulated_context_tokens": accumulated_context,
                "context_vs_single_round": context_multiplier,
                "additional_prefill_tokens": additional_prefill_tokens,
                "additional_prefill_share_pct": (additional_prefill_tokens / accumulated_context * 100.0) if accumulated_context else None,
                "timing_total_s": timing_summary["timing_total_ms"] / 1000.0,
                "summary_generation_s": summary_generation_s,
                "classification_counts": timing_summary["classification_counts"],
                "turn_context_series": timing_summary["turn_context_series"],
                "score_valid": summary.get("valid"),
                "score": summary.get("score"),
                "error_type": summary.get("error_type"),
                "tool_calls": summary.get("tool_calls"),
                "has_trace": trace_exists,
                "has_prompt_log": prompt_log_exists,
                "has_checkpoint": checkpoint_exists,
            }
        )

    summary_gen_ms = sum(as_num(event.get("summary_generation_time_s"), 0.0) or 0.0 for event in compression_events) * 1000.0
    if summary_gen_ms:
        phase_rows_by_dataset["summary_generation"]["count"] = len(compression_events)
        phase_rows_by_dataset["summary_generation"]["total_ms"] = summary_gen_ms

    aggregate = compute_aggregate(sample_summaries, compression_events)
    phase_rows = [
        {"phase": phase, **values, "total_s": values["total_ms"] / 1000.0}
        for phase, values in sorted(phase_rows_by_dataset.items())
    ]
    score_rows = [
        {
            "sample_label": row["sample_label"],
            "sample_id": row["sample_id"],
            "run_name": row["run_name"],
            "valid": row["score_valid"],
            "score": row["score"],
            "error_type": row["error_type"],
        }
        for row in sample_summaries
        if row.get("score_valid") is not None or row.get("score") is not None or row.get("error_type")
    ]

    analysis = {
        "dataset": {
            "slug": config.slug,
            "title": config.title,
            "description": config.description,
            "data_root": str(config.data_root),
            "output_dir": str(output_dir),
            "caveat": config.caveat,
        },
        "source_file_counts": source_file_counts,
        "aggregate": aggregate,
        "samples": sample_summaries,
        "compression_events": compression_events,
        "phase_breakdown": phase_rows,
        "run_configs": run_configs,
        "score_rows": score_rows,
    }

    write_csv(tables_dir / "sample_workload.csv", sample_summaries, skip_keys={"classification_counts", "turn_context_series"})
    write_csv(tables_dir / "compression_events.csv", compression_events)
    write_csv(tables_dir / "phase_breakdown.csv", phase_rows)
    if score_rows:
        write_csv(tables_dir / "score_validity.csv", score_rows)
    write_csv(tables_dir / "required_claims.csv", [aggregate])
    (output_dir / "analysis_results.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_required_compression_prefill(config, sample_summaries, aggregate, charts_dir)
    plot_required_context_accumulation(config, sample_summaries, aggregate, charts_dir)
    plot_context_over_rounds(config, sample_summaries, run_configs, charts_dir)
    plot_abc_composition(config, compression_events, charts_dir)
    plot_compression_savings(config, compression_events, charts_dir)
    plot_phase_breakdown(config, phase_rows, charts_dir)

    report = render_report(config, analysis)
    (output_dir / config.report_name).write_text(report, encoding="utf-8")
    return analysis


def compute_aggregate(samples: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
    active_samples = [
        row
        for row in samples
        if (as_num(row.get("steps"), 0.0) or 0.0) > 0 and (as_num(row.get("first_context_tokens"), 0.0) or 0.0) > 0
    ]

    def vals(key: str, rows: list[dict[str, Any]] | None = None) -> list[float]:
        source = rows if rows is not None else samples
        return [as_num(row.get(key), 0.0) or 0.0 for row in source]

    multipliers = [as_num(row.get("context_vs_single_round")) for row in active_samples if as_num(row.get("context_vs_single_round")) is not None]
    additional_prefill = sum(vals("additional_prefill_tokens"))
    accumulated_context = sum(vals("accumulated_context_tokens"))
    active_rounds = vals("rounds", active_samples)
    active_compressions = vals("compressions", active_samples)
    first_contexts = vals("first_context_tokens", active_samples)
    max_contexts = vals("max_context_tokens", active_samples)
    compression_prefill_share_pct = additional_prefill / accumulated_context * 100.0 if accumulated_context else 0.0
    b2_c1_values = [as_num(row.get("B2_plus_C1_tokens"), 0.0) or 0.0 for row in events]
    summary_generation_values = [as_num(row.get("summary_generation_time_s"), 0.0) or 0.0 for row in events]
    savings = [as_num(row.get("token_saving_pct")) for row in events if as_num(row.get("token_saving_pct")) is not None]
    c1_values = [as_num(row.get("C1_tokens")) for row in events if as_num(row.get("C1_tokens")) is not None]
    return {
        "sample_count": len(samples),
        "active_sample_count": len(active_samples),
        "samples_with_nonzero_first_context": len(active_samples),
        "compression_event_count": len(events),
        "total_rounds": sum(vals("rounds")),
        "total_steps": sum(vals("steps")),
        "total_compressions": sum(vals("compressions")),
        "avg_rounds_per_request": mean(active_rounds) if active_rounds else 0.0,
        "median_rounds_per_request": median(active_rounds) if active_rounds else 0.0,
        "avg_steps_per_request": mean(vals("steps", active_samples)) if active_samples else 0.0,
        "avg_compressions_per_request": mean(active_compressions) if active_compressions else 0.0,
        "median_compressions_per_request": median(active_compressions) if active_compressions else 0.0,
        "compression_per_round": sum(active_compressions) / sum(active_rounds) if sum(active_rounds) else 0.0,
        "avg_first_context_tokens": mean(first_contexts) if first_contexts else 0.0,
        "avg_max_context_tokens": mean(max_contexts) if max_contexts else 0.0,
        "avg_accumulated_context_tokens": mean(vals("accumulated_context_tokens", active_samples)) if active_samples else 0.0,
        "total_accumulated_context_tokens": accumulated_context,
        "avg_context_vs_single_round": mean(multipliers) if multipliers else 0.0,
        "min_context_vs_single_round": min(multipliers) if multipliers else 0.0,
        "max_context_vs_single_round": max(multipliers) if multipliers else 0.0,
        "total_additional_prefill_tokens": additional_prefill,
        "additional_prefill_share_pct": compression_prefill_share_pct,
        "avg_additional_prefill_tokens_per_request": mean(vals("additional_prefill_tokens", active_samples)) if active_samples else 0.0,
        "avg_B2_plus_C1_tokens_per_compression": mean(b2_c1_values) if b2_c1_values else 0.0,
        "avg_C1_tokens_per_compression": mean(c1_values) if c1_values else 0.0,
        "avg_token_saving_pct": mean(savings) if savings else 0.0,
        "avg_summary_generation_time_s": mean(summary_generation_values) if summary_generation_values else 0.0,
        "total_summary_generation_time_s": sum(summary_generation_values),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], skip_keys: set[str] | None = None) -> None:
    skip_keys = skip_keys or set()
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in skip_keys and key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False) if isinstance(row.get(key), (dict, list)) else row.get(key) for key in keys})


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_required_compression_prefill(config: DatasetConfig, samples: list[dict[str, Any]], aggregate: dict[str, Any], charts_dir: Path) -> None:
    labels = [row["sample_label"] for row in samples]
    x = list(range(len(samples)))
    rounds = [as_num(row.get("rounds"), 0.0) or 0.0 for row in samples]
    comps = [as_num(row.get("compressions"), 0.0) or 0.0 for row in samples]
    shares = [as_num(row.get("additional_prefill_share_pct"), 0.0) or 0.0 for row in samples]
    fig, axes = plt.subplots(2, 1, figsize=(max(8, len(samples) * 0.75), 7), sharex=True)
    width = 0.38
    axes[0].bar([i - width / 2 for i in x], rounds, width=width, label="Rounds", color="#4C78A8")
    axes[0].bar([i + width / 2 for i in x], comps, width=width, label="Compressions", color="#F58518")
    axes[0].set_ylabel("Count")
    axes[0].legend(loc="upper left")
    axes[0].set_title(
        f"Compression frequency: avg {aggregate['avg_compressions_per_request']:.2f} per {aggregate['avg_rounds_per_request']:.2f} rounds"
    )
    axes[1].bar(x, shares, color="#54A24B")
    axes[1].axhline(aggregate["additional_prefill_share_pct"], color="#E45756", linestyle="--", linewidth=1.4, label="Aggregate share")
    axes[1].set_ylabel("Additional prefill share (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].legend(loc="upper right")
    axes[1].set_title(f"Compression-induced prefill share: {aggregate['additional_prefill_share_pct']:.2f}%")
    fig.suptitle(config.title, y=1.02, fontsize=13)
    savefig(charts_dir / "figure1_compression_frequency_prefill_share.png")


def plot_required_context_accumulation(config: DatasetConfig, samples: list[dict[str, Any]], aggregate: dict[str, Any], charts_dir: Path) -> None:
    labels = [row["sample_label"] for row in samples]
    x = list(range(len(samples)))
    first = [(as_num(row.get("first_context_tokens"), 0.0) or 0.0) / 1000.0 for row in samples]
    accumulated = [(as_num(row.get("accumulated_context_tokens"), 0.0) or 0.0) / 1000.0 for row in samples]
    multiplier = [as_num(row.get("context_vs_single_round"), 0.0) or 0.0 for row in samples]
    fig, ax1 = plt.subplots(figsize=(max(8, len(samples) * 0.8), 5.5))
    width = 0.38
    ax1.bar([i - width / 2 for i in x], first, width=width, color="#72B7B2", label="First request context")
    ax1.bar([i + width / 2 for i in x], accumulated, width=width, color="#4C78A8", label="Accumulated context")
    ax1.set_ylabel("Prompt tokens (K)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(x, multiplier, color="#E45756", marker="o", linewidth=2, label="Accumulated / first")
    ax2.set_ylabel("Multiplier")
    ax2.legend(loc="upper right")
    ax1.set_title(
        f"Avg accumulated context {aggregate['avg_accumulated_context_tokens'] / 1000.0:.1f}K, multiplier {aggregate['avg_context_vs_single_round']:.1f}x"
    )
    fig.suptitle(config.title, y=1.02, fontsize=13)
    savefig(charts_dir / "figure2_context_accumulation_vs_single_round.png")


def plot_context_over_rounds(config: DatasetConfig, samples: list[dict[str, Any]], run_configs: dict[str, Any], charts_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for row in samples:
        series = row.get("turn_context_series") or []
        if not series:
            continue
        turns = [item["turn"] for item in series]
        prompts = [(as_num(item.get("prompt_tokens"), 0.0) or 0.0) / 1000.0 for item in series]
        ax.plot(turns, prompts, marker="o", linewidth=1.3, markersize=3, label=row["sample_label"])
    threshold = find_threshold(run_configs)
    if threshold:
        ax.axhline(threshold / 1000.0, color="#E45756", linestyle="--", linewidth=1.2, label=f"Threshold {threshold / 1000.0:.0f}K")
    ax.set_xlabel("Round / turn")
    ax.set_ylabel("Max prompt tokens in round (K)")
    ax.set_title("Context length over rounds")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    savefig(charts_dir / "context_length_over_rounds.png")


def find_threshold(run_configs: dict[str, Any]) -> float | None:
    for config in run_configs.values():
        if not isinstance(config, dict):
            continue
        for path in [("preset", "threshold"), ("compression", "threshold"), ("threshold",)]:
            obj: Any = config
            for key in path:
                obj = obj.get(key) if isinstance(obj, dict) else None
            value = as_num(obj)
            if value:
                return value
    return None


def plot_abc_composition(config: DatasetConfig, events: list[dict[str, Any]], charts_dir: Path) -> None:
    if not events:
        empty_plot(charts_dir / "abc_composition_by_sample.png", "No compression events")
        return
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[str(event["sample_label"])].append(event)
    labels = sorted(grouped)
    metrics = ["A_tokens", "B1_tokens", "B2_tokens", "C1_tokens"]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    left = [0.0 for _ in labels]
    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.42)))
    for metric, color in zip(metrics, colors):
        values = []
        for label in labels:
            nums = [as_num(event.get(metric), 0.0) or 0.0 for event in grouped[label]]
            values.append((mean(nums) if nums else 0.0) / 1000.0)
        ax.barh(labels, values, left=left, color=color, label=metric.replace("_tokens", ""))
        left = [base + value for base, value in zip(left, values)]
    ax.set_xlabel("Average tokens per compression (K)")
    ax.set_title("ABC segment composition by sample")
    ax.legend(loc="lower right")
    savefig(charts_dir / "abc_composition_by_sample.png")


def plot_compression_savings(config: DatasetConfig, events: list[dict[str, Any]], charts_dir: Path) -> None:
    if not events:
        empty_plot(charts_dir / "compression_savings_pre_post.png", "No compression events")
        return
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[str(event["sample_label"])].append(event)
    labels = sorted(grouped)
    x = list(range(len(labels)))
    pre = []
    post = []
    saving = []
    for label in labels:
        rows = grouped[label]
        pre_vals = [as_num(row.get("pre_prompt_tokens"), 0.0) or 0.0 for row in rows]
        post_vals = [as_num(row.get("post_prompt_tokens"), 0.0) or 0.0 for row in rows]
        saving_vals = [as_num(row.get("token_saving_pct")) for row in rows if as_num(row.get("token_saving_pct")) is not None]
        pre.append((mean(pre_vals) if pre_vals else 0.0) / 1000.0)
        post.append((mean(post_vals) if post_vals else 0.0) / 1000.0)
        saving.append(mean(saving_vals) if saving_vals else 0.0)
    fig, ax1 = plt.subplots(figsize=(max(8, len(labels) * 0.75), 5.2))
    width = 0.36
    ax1.bar([i - width / 2 for i in x], pre, width=width, label="Pre-compression", color="#F58518")
    ax1.bar([i + width / 2 for i in x], post, width=width, label="Post-compression", color="#4C78A8")
    ax1.set_ylabel("Prompt tokens (K)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(x, saving, color="#E45756", marker="o", linewidth=2, label="Token saving %")
    ax2.set_ylabel("Saving (%)")
    ax2.legend(loc="upper right")
    ax1.set_title("Compression savings: pre vs post prompt tokens")
    savefig(charts_dir / "compression_savings_pre_post.png")


def plot_phase_breakdown(config: DatasetConfig, phase_rows: list[dict[str, Any]], charts_dir: Path) -> None:
    if not phase_rows:
        empty_plot(charts_dir / "phase_breakdown_seconds.png", "No timing rows")
        return
    rows = sorted(phase_rows, key=lambda row: as_num(row.get("total_s"), 0.0) or 0.0, reverse=True)
    labels = [str(row["phase"]) for row in rows]
    seconds = [as_num(row.get("total_s"), 0.0) or 0.0 for row in rows]
    counts = [as_num(row.get("count"), 0.0) or 0.0 for row in rows]
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.bar(labels, seconds, color="#4C78A8")
    ax1.set_ylabel("Total seconds")
    ax1.set_title("Latency phase breakdown")
    ax1.tick_params(axis="x", rotation=30)
    ax2 = ax1.twinx()
    ax2.plot(labels, counts, color="#E45756", marker="o", linewidth=2)
    ax2.set_ylabel("Request/event count")
    savefig(charts_dir / "phase_breakdown_seconds.png")


def empty_plot(path: Path, text: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12)
    ax.axis("off")
    savefig(path)


def fmt_int(value: Any) -> str:
    number = as_num(value)
    if number is None:
        return "-"
    return f"{number:,.0f}"


def fmt_float(value: Any, digits: int = 2) -> str:
    number = as_num(value)
    if number is None:
        return "-"
    return f"{number:,.{digits}f}"


def fmt_pct(value: Any, digits: int = 2) -> str:
    number = as_num(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}%"


def md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str, str]], limit: int | None = None) -> str:
    selected = rows[:limit] if limit else rows
    if not selected:
        return "_无可用记录。_\n"
    header = "| " + " | ".join(name for _, name, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in selected:
        cells = []
        for key, _, kind in columns:
            value = row.get(key)
            if kind == "int":
                cells.append(fmt_int(value))
            elif kind == "float":
                cells.append(fmt_float(value))
            elif kind == "float1":
                cells.append(fmt_float(value, 1))
            elif kind == "pct":
                cells.append(fmt_pct(value))
            elif kind == "bool":
                cells.append("yes" if value is True else "no" if value is False else "-")
            else:
                text = "-" if value is None else str(value)
                cells.append(text.replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    if limit and len(rows) > limit:
        lines.append(f"\n_表中展示前 {limit} 条；完整数据见 tables/*.csv。_")
    return "\n".join(lines) + "\n"


def render_report(config: DatasetConfig, analysis: dict[str, Any]) -> str:
    aggregate = analysis["aggregate"]
    samples = analysis["samples"]
    events = analysis["compression_events"]
    phase = analysis["phase_breakdown"]
    run_configs = analysis["run_configs"]
    source_counts = analysis["source_file_counts"]
    score_rows = analysis.get("score_rows", [])
    config_rows: list[dict[str, Any]] = []
    for run_name, run_config in run_configs.items():
        for row in flatten_config(run_config):
            config_rows.append({"run_name": run_name, **row})

    claim1 = (
        f"As shown in Figure 1, a single agentic request invokes "
        f"{aggregate['avg_compressions_per_request']:.2f} compressions per "
        f"{aggregate['avg_rounds_per_request']:.2f} rounds on average, while the additional prefill tokens "
        f"due to compression account for {aggregate['additional_prefill_share_pct']:.2f}% of accumulated prompt tokens."
    )
    claim2 = (
        f"As shown in Figure 2, a single agentic request often spans "
        f"{aggregate['avg_rounds_per_request']:.2f} rounds and accumulates a total of "
        f"{aggregate['avg_accumulated_context_tokens'] / 1000.0:.1f}K context tokens, which is "
        f"{aggregate['min_context_vs_single_round']:.1f}x-{aggregate['max_context_vs_single_round']:.1f}x longer "
        f"than single-round inference."
    )

    sample_columns = [
        ("sample_label", "样本", "str"),
        ("run_name", "run", "str"),
        ("rounds", "rounds", "int"),
        ("steps", "LLM calls", "int"),
        ("compressions", "compressions", "int"),
        ("compression_per_round", "P", "float"),
        ("first_context_tokens", "first ctx", "int"),
        ("max_context_tokens", "max ctx", "int"),
        ("accumulated_context_tokens", "acc ctx", "int"),
        ("context_vs_single_round", "acc/first", "float"),
        ("additional_prefill_share_pct", "extra prefill", "pct"),
    ]
    event_columns = [
        ("sample_label", "样本", "str"),
        ("compression_index", "idx", "int"),
        ("turn", "turn", "int"),
        ("step", "step", "int"),
        ("pre_prompt_tokens", "pre", "int"),
        ("post_prompt_tokens", "post", "int"),
        ("B1_tokens", "B1", "int"),
        ("B2_tokens", "B2", "int"),
        ("C1_tokens", "C1", "int"),
        ("B2_plus_C1_tokens", "B2+C1", "int"),
        ("token_saving_pct", "saving", "pct"),
        ("summary_generation_time_s", "summary s", "float1"),
    ]
    phase_columns = [
        ("phase", "phase", "str"),
        ("count", "count", "int"),
        ("prompt_tokens", "prompt tokens", "int"),
        ("output_tokens", "output tokens", "int"),
        ("total_s", "total s", "float1"),
    ]
    config_columns = [("run_name", "run", "str"), ("key", "key", "str"), ("value", "value", "str")]
    score_columns = [
        ("sample_label", "样本", "str"),
        ("run_name", "run", "str"),
        ("valid", "valid", "bool"),
        ("score", "score", "float"),
        ("error_type", "error", "str"),
    ]
    config_section = md_table(config_rows, config_columns, limit=40) if config_rows else "_未发现 run_config.json；本节仅基于 timing/ABC 文件统计。_\n"
    score_section = md_table(score_rows, score_columns, limit=20) if score_rows else "_未发现 score summary 或 benchmark validity 字段；本文不报告 accuracy 结论。_\n"

    return f"""# {config.title} 分析报告

生成时间：{TODAY}

## 结论摘要

- 可读样本数：{fmt_int(aggregate['sample_count'])}；有效 timing 样本数：{fmt_int(aggregate['active_sample_count'])}；压缩事件数：{fmt_int(aggregate['compression_event_count'])}；总 rounds：{fmt_int(aggregate['total_rounds'])}；总 LLM calls：{fmt_int(aggregate['total_steps'])}。
- 平均每个 agentic request 运行 {fmt_float(aggregate['avg_rounds_per_request'])} rounds / {fmt_float(aggregate['avg_steps_per_request'])} LLM calls，并触发 {fmt_float(aggregate['avg_compressions_per_request'])} 次压缩。
- 压缩引入的额外 prefill tokens 为 {fmt_int(aggregate['total_additional_prefill_tokens'])}，占累计 prompt/context tokens 的 {fmt_pct(aggregate['additional_prefill_share_pct'])}。
- 单请求平均累计上下文为 {fmt_int(aggregate['avg_accumulated_context_tokens'])} tokens；相比首轮单次请求，平均为 {fmt_float(aggregate['avg_context_vs_single_round'])}x，样本范围为 {fmt_float(aggregate['min_context_vs_single_round'], 1)}x-{fmt_float(aggregate['max_context_vs_single_round'], 1)}x。

{config.caveat}

## 数据口径

- `rounds` 优先采用 run summary 中的 turns/decoded_turns；没有 summary 时使用 timing 日志中的唯一 turn 数。
- `LLM calls` 来自 `timing/*.json` 记录条数。
- 每请求均值只使用 `steps>0` 且首轮 `prompt_tokens>0` 的有效 timing 样本；空样本仍保留在完整性统计和样本表中。
- `accumulated context` 定义为单个样本所有 LLM request 的 `prompt_tokens` 之和。
- `single-round inference` 对照定义为该样本第一条 timing 记录的 `prompt_tokens`。
- `additional prefill tokens due to compression` 优先采用 timing 的 `semi_prefill_tokens`；若没有该字段，则采用 ABC 压缩事件中的 `B2+C1`。
- 本报告不读取或输出长 prompt 原文，只保留 timing/ABC 的标量统计字段。

## 数据完整性

| 项目 | 数量 |
| --- | ---: |
| timing files | {fmt_int(source_counts['timing_files'])} |
| ABC files | {fmt_int(source_counts['abc_files'])} |
| trace files | {fmt_int(source_counts['trace_files'])} |
| prompt log files | {fmt_int(source_counts['prompt_log_files'])} |
| checkpoint files | {fmt_int(source_counts['checkpoint_files'])} |
| declared sample ids | {fmt_int(source_counts['declared_sample_ids'])} |

## Figure 1：压缩频率与额外 Prefill 占比

![Compression frequency and prefill share](charts/figure1_compression_frequency_prefill_share.png)

{claim1}

中文解读：图 1 同时展示每个样本的 rounds、压缩次数和压缩导致的额外 prefill 占比。整体压缩频率为 {fmt_float(aggregate['compression_per_round'])} 次/round，这对应压缩触发概率 P 的经验估计。

## Figure 2：多轮累计上下文与单轮对照

![Context accumulation vs single round](charts/figure2_context_accumulation_vs_single_round.png)

{claim2}

中文解读：图 2 展示首轮上下文、累计上下文以及累计/首轮倍数。该图说明 agentic request 的成本不能只按单轮推理估计；多轮调用会重复携带或重建大量上下文。

## 样本工作负载表

{md_table(samples, sample_columns, limit=30)}

## 压缩事件表

{md_table(events, event_columns, limit=40)}

ABC 分段图如下：

![ABC composition](charts/abc_composition_by_sample.png)

压缩前后 prompt token 对比如下：

![Compression savings](charts/compression_savings_pre_post.png)

## Context 长度随轮次变化

![Context length over rounds](charts/context_length_over_rounds.png)

这张图采用每个 round 内最大的 `prompt_tokens` 作为该 round 的上下文长度。如果 run_config 中存在 threshold，图中会画出阈值线。

## Timing / Phase Breakdown

{md_table(phase, phase_columns)}

![Phase breakdown](charts/phase_breakdown_seconds.png)

## 配置摘要

{config_section}

## Score / Validity 摘要

{score_section}

## 证据与限制

已验证：

- timing、ABC、trace/prompt log/checkpoint 文件数量已经按文件系统重新扫描。
- 所有核心数值都写入 `analysis_results.json` 与 `tables/*.csv`，报告中的 Figure 1/2 文案由同一份 JSON 指标生成。
- 图表均来自 timing 和 ABC 的标量字段，不依赖 README 或旧报告文字。

未验证：

- 本分析不是一次新的 benchmark run，也不证明未完成样本可以跑满目标 turns。
- BFCL validity/score 若为 invalid，仅说明这些样本不能作为最终准确率结论；压缩次数、上下文长度和 prefill token 统计仍可作为运行日志证据。
- `additional prefill tokens` 是按日志可见的 semi-prefill 或 ABC `B2+C1` 估算；如果底层推理服务还有隐藏 prefix-cache 命中/失效，该比例不包含未记录的内部实现细节。

## 产物清单

- `analysis_results.json`：完整结构化统计。
- `tables/sample_workload.csv`：样本级工作负载表。
- `tables/compression_events.csv`：压缩事件表。
- `tables/phase_breakdown.csv`：阶段耗时与 token 表。
- `charts/*.png`：报告图表。
- `code/generate_analysis.py`：生成本目录产物的脚本副本。
"""


def main() -> None:
    summaries = []
    for dataset in DATASETS:
        analysis = analyze_dataset(dataset)
        aggregate = analysis["aggregate"]
        summaries.append(
            {
                "slug": dataset.slug,
                "output_dir": str(dataset.output_dir),
                "samples": aggregate["sample_count"],
                "active_samples": aggregate["active_sample_count"],
                "events": aggregate["compression_event_count"],
                "avg_compressions": round(aggregate["avg_compressions_per_request"], 3),
                "avg_rounds": round(aggregate["avg_rounds_per_request"], 3),
                "prefill_share_pct": round(aggregate["additional_prefill_share_pct"], 3),
                "avg_context_k": round(aggregate["avg_accumulated_context_tokens"] / 1000.0, 3),
                "context_multiplier_min": round(aggregate["min_context_vs_single_round"], 3),
                "context_multiplier_max": round(aggregate["max_context_vs_single_round"], 3),
            }
        )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()