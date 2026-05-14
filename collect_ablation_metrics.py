#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean


MB_PATTERNS = [
    ("wire_bytes", re.compile(r"\bbytes=([0-9.]+)MB")),
    ("payload", re.compile(r"\bpayload=([0-9.]+)MB")),
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def iter_timing_rows(run_dir: Path):
    for path in sorted((run_dir / "timing").glob("*.json")):
        rows = load_json(path)
        if isinstance(rows, list):
            for row in rows:
                row = dict(row)
                row["sample_file"] = path.name
                yield row


def timing_summary(run_dir: Path, first_turn_only: bool) -> dict:
    rows = list(iter_timing_rows(run_dir))
    if first_turn_only:
        rows = [row for row in rows if row.get("turn") == 0]
    ttft = [float(row["ttft_ms"]) for row in rows if "ttft_ms" in row]
    total = [float(row["total_ms"]) for row in rows if "total_ms" in row]
    tpot = []
    for row in rows:
        output_tokens = int(row.get("output_tokens") or 0)
        if output_tokens <= 1:
            continue
        decode_ms = float(row.get("decode_ms") or 0.0)
        tpot.append(decode_ms / max(output_tokens - 1, 1))
    return {
        "steps": len(rows),
        "ttft_ms_mean": round(mean(ttft), 3) if ttft else None,
        "ttft_ms_values": ttft,
        "total_ms_mean": round(mean(total), 3) if total else None,
        "tpot_ms_mean": round(mean(tpot), 3) if tpot else None,
        "tpot_ms_values": [round(value, 3) for value in tpot],
    }


def score_summary(score_file: Path | None) -> dict:
    if score_file is None or not score_file.exists():
        return {"available": False}
    report = load_json(score_file)
    return {
        "available": True,
        "metric": report.get("metric"),
        "samples": report.get("samples"),
        "correct": report.get("correct"),
        "accuracy": report.get("accuracy"),
        "quality_breakdown": report.get("quality_breakdown"),
        "entries": report.get("entries", []),
    }


def log_comm_summary(log_dir: Path | None) -> dict:
    if log_dir is None or not log_dir.exists():
        return {"available": False}
    files = sorted(log_dir.rglob("*.log"))
    by_file = {}
    wire_total_mb = 0.0
    codec_payload_total_mb = 0.0
    for path in files:
        wire_values = []
        codec_payload_values = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            for kind, pattern in MB_PATTERNS:
                match = pattern.search(line)
                if match:
                    value = float(match.group(1))
                    if kind == "wire_bytes" or "[CommManager EH] Phase2" in line:
                        wire_values.append(value)
                    else:
                        codec_payload_values.append(value)
                    break
        if wire_values or codec_payload_values:
            file_wire_total = sum(wire_values)
            file_codec_total = sum(codec_payload_values)
            by_file[str(path.relative_to(log_dir))] = {
                "wire_or_direct_payload_events": len(wire_values),
                "wire_or_direct_payload_total_mb": round(file_wire_total, 3),
                "wire_or_direct_payload_mean_mb": round(mean(wire_values), 3) if wire_values else None,
                "wire_or_direct_payload_values_mb": [round(value, 3) for value in wire_values],
                "codec_payload_events": len(codec_payload_values),
                "codec_payload_total_mb": round(file_codec_total, 3),
                "codec_payload_mean_mb": round(mean(codec_payload_values), 3) if codec_payload_values else None,
                "codec_payload_values_mb": [round(value, 3) for value in codec_payload_values],
            }
            wire_total_mb += file_wire_total
            codec_payload_total_mb += file_codec_total
    return {
        "available": True,
        "log_dir": str(log_dir),
        "wire_or_direct_payload_total_mb": round(wire_total_mb, 3),
        "codec_payload_total_mb": round(codec_payload_total_mb, 3),
        "by_file": by_file,
        "note": "Use wire_or_direct_payload_total_mb for communication volume. codec_payload_total_mb is a codec-side estimate and is reported separately to avoid double-counting sender bytes.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect communication ablation metrics")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--score-file", type=Path, default=None)
    parser.add_argument("--vllm-log-dir", type=Path, default=None)
    parser.add_argument("--first-turn-only", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    report = {
        "run_dir": str(run_dir),
        "timing": timing_summary(run_dir, args.first_turn_only),
        "quality": score_summary(args.score_file.resolve() if args.score_file else None),
        "communication": log_comm_summary(args.vllm_log_dir.resolve() if args.vllm_log_dir else None),
    }

    output = args.output or (run_dir / "metrics" / "ablation_metrics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {output}")


if __name__ == "__main__":
    main()