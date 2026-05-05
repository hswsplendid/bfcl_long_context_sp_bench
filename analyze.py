import argparse
import json
import statistics as stats
from pathlib import Path

import sp_config as CFG


def load_run(run_dir: Path) -> dict:
    samples = {}
    trace_dir = run_dir / "traces"
    for trace_path in sorted(trace_dir.glob("*.json")):
        sample_id = trace_path.stem
        timing_path = run_dir / "timing" / f"{sample_id}.json"
        abc_path = run_dir / "abc_segments" / f"{sample_id}.json"
        samples[sample_id] = {
            "trace": json.load(open(trace_path, "r", encoding="utf-8")),
            "timing": json.load(open(timing_path, "r", encoding="utf-8")) if timing_path.exists() else [],
            "abc": json.load(open(abc_path, "r", encoding="utf-8")) if abc_path.exists() else [],
        }
    return samples


def describe(values: list[float]) -> dict:
    values = [value for value in values if value is not None]
    if not values:
        return {}
    report = {
        "min": min(values),
        "max": max(values),
        "mean": round(stats.mean(values), 3),
        "median": round(stats.median(values), 3),
    }
    if len(values) >= 4:
        quantiles = stats.quantiles(values, n=4)
        report["p25"] = round(quantiles[0], 3)
        report["p75"] = round(quantiles[2], 3)
    return report


def workload(samples: dict) -> dict:
    rows = []
    all_sp_lengths = []
    all_prompt_tokens = []
    all_new_tokens = []
    for sample_id, payload in samples.items():
        timing = payload["timing"]
        abc = payload["abc"]
        trace = payload["trace"]
        if not timing:
            continue
        prompt_tokens = [row.get("prompt_tokens", 0) for row in timing]
        output_tokens = [row.get("output_tokens") or 0 for row in timing]
        deltas = [prompt_tokens[index] - prompt_tokens[index - 1] for index in range(1, len(prompt_tokens))]
        all_prompt_tokens.extend(prompt_tokens)
        all_new_tokens.extend(output_tokens)
        churn = []
        for event in abc:
            before = event.get("pre_prompt_tokens") or 0
            after = event.get("post_prompt_tokens") or 0
            if before > 0:
                churn.append((before - after) / before)
        sp_lengths = [event.get("B2_plus_C1_tokens", 0) for event in abc]
        all_sp_lengths.extend(sp_lengths)
        compression_rate = len(abc) / len(timing)
        rows.append(
            {
                "id": sample_id,
                "turns": trace.get("total_turns", len(trace.get("turns", []))),
                "steps": len(timing),
                "tool_calls": sum(row.get("tool_calls_in_response", 0) for row in timing),
                "compressions": len(abc),
                "P": round(compression_rate, 4),
                "context_lengths": prompt_tokens,
                "new_tokens_per_step": output_tokens,
                "prompt_token_deltas": deltas,
                "max_prompt_tokens": max(prompt_tokens),
                "mean_new_tokens_per_step": round(stats.mean(output_tokens), 2) if output_tokens else 0.0,
                "mean_delta_per_step": round(stats.mean(deltas), 2) if deltas else 0.0,
                "churn_ratio_mean": round(stats.mean(churn), 4) if churn else None,
                "min_c1": min((event.get("C1_tokens_after", 0) for event in abc), default=None),
                "max_c1": max((event.get("C1_tokens_after", 0) for event in abc), default=None),
                "semi_prefill_count": sum(1 for row in timing if row.get("classification") == "semi_prefill"),
                "semi_prefill_lengths": sp_lengths,
                "semi_prefill_events": [
                    {
                        "turn": event.get("turn"),
                        "step": event.get("step"),
                        "tokens": event.get("B2_plus_C1_tokens", 0),
                        "c1_tokens": event.get("C1_tokens_after", 0),
                    }
                    for event in abc
                ],
            }
        )

    summary = {
        "n_samples": len(rows),
        "turns": describe([row["turns"] for row in rows]),
        "steps": describe([row["steps"] for row in rows]),
        "tool_calls": describe([row["tool_calls"] for row in rows]),
        "compressions": describe([row["compressions"] for row in rows]),
        "P": describe([row["P"] for row in rows]),
        "max_prompt_tokens": describe([row["max_prompt_tokens"] for row in rows]),
        "context_length_distribution": describe(all_prompt_tokens),
        "new_tokens_per_step": describe(all_new_tokens),
        "mean_delta_per_step": describe([row["mean_delta_per_step"] for row in rows]),
        "churn_ratio_mean": describe([row["churn_ratio_mean"] for row in rows]),
        "samples_P_le_1_5": sum(1 for row in rows if 0 < row["P"] <= CFG.P_TARGET_1_5),
        "samples_P_le_1_6": sum(1 for row in rows if 0 < row["P"] <= CFG.P_TARGET_1_6),
        "samples_P_gt_1_5": sum(1 for row in rows if row["P"] > CFG.P_TARGET_1_5),
        "samples_P_gt_1_6": sum(1 for row in rows if row["P"] > CFG.P_TARGET_1_6),
        "samples_P_zero": sum(1 for row in rows if row["P"] == 0),
        "samples_C1_lt_2000": sum(1 for row in rows if row["min_c1"] is not None and row["min_c1"] < CFG.C1_MIN_TOKENS),
        "samples_C1_gt_3000": sum(1 for row in rows if row["max_c1"] is not None and row["max_c1"] > CFG.C1_MAX_TOKENS),
    }
    if all_sp_lengths:
        summary["semi_prefill_lengths"] = describe(all_sp_lengths)
        summary["semi_prefill_lengths"]["n_events"] = len(all_sp_lengths)
    return {"summary": summary, "per_sample": rows}


def time_breakdown(samples: dict) -> dict:
    aggregate = {
        "prefill_ms": 0.0,
        "semi_prefill_ms": 0.0,
        "summary_gen_ms": 0.0,
        "incremental_ms": 0.0,
        "decode_ms": 0.0,
    }
    per_sample = []
    for sample_id, payload in samples.items():
        timing = payload["timing"]
        abc = payload["abc"]
        pref = semi = incr = decode = 0.0
        summary_ms = sum((event.get("summary_generation_time_s") or 0.0) * 1000.0 for event in abc)
        for row in timing:
            cls = row.get("classification")
            ttft = row.get("ttft_ms") or 0.0
            decode += row.get("decode_ms") or 0.0
            if cls == "full_prefill":
                pref += ttft
            elif cls == "semi_prefill":
                semi += ttft
            else:
                incr += ttft
        aggregate["prefill_ms"] += pref
        aggregate["semi_prefill_ms"] += semi
        aggregate["summary_gen_ms"] += summary_ms
        aggregate["incremental_ms"] += incr
        aggregate["decode_ms"] += decode
        total = pref + semi + incr + decode + summary_ms
        per_sample.append(
            {
                "id": sample_id,
                "prefill_ms": round(pref, 2),
                "semi_prefill_ms": round(semi, 2),
                "summary_gen_ms": round(summary_ms, 2),
                "incremental_ms": round(incr, 2),
                "decode_ms": round(decode, 2),
                "total_ms": round(total, 2),
                "semi_prefill_total_ms": round(semi + summary_ms, 2),
                "pct": {
                    "prefill": round(pref / total * 100, 2) if total else 0.0,
                    "semi_prefill": round((semi + summary_ms) / total * 100, 2) if total else 0.0,
                    "semi_prefill_io": round(semi / total * 100, 2) if total else 0.0,
                    "summary_gen": round(summary_ms / total * 100, 2) if total else 0.0,
                    "incremental": round(incr / total * 100, 2) if total else 0.0,
                    "decode": round(decode / total * 100, 2) if total else 0.0,
                },
            }
        )
    total_ms = sum(aggregate.values())
    pct = {
        "prefill": round(aggregate["prefill_ms"] / total_ms * 100, 2) if total_ms else 0.0,
        "semi_prefill": round((aggregate["semi_prefill_ms"] + aggregate["summary_gen_ms"]) / total_ms * 100, 2) if total_ms else 0.0,
        "semi_prefill_io": round(aggregate["semi_prefill_ms"] / total_ms * 100, 2) if total_ms else 0.0,
        "summary_gen": round(aggregate["summary_gen_ms"] / total_ms * 100, 2) if total_ms else 0.0,
        "incremental": round(aggregate["incremental_ms"] / total_ms * 100, 2) if total_ms else 0.0,
        "decode": round(aggregate["decode_ms"] / total_ms * 100, 2) if total_ms else 0.0,
    }
    aggregate["total_ms"] = round(total_ms, 2)
    aggregate["pct"] = pct
    return {"aggregate": aggregate, "per_sample": per_sample}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    samples = load_run(args.run_dir)
    report = {
        "run_dir": str(args.run_dir),
        "workload": workload(samples),
        "time_breakdown": time_breakdown(samples),
    }
    out_path = args.run_dir / "analysis.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    workload_summary = report["workload"]["summary"]
    time_summary = report["time_breakdown"]["aggregate"]
    print(f"samples: {workload_summary['n_samples']}")
    print(f"P<=1/5: {workload_summary['samples_P_le_1_5']}/{workload_summary['n_samples']}")
    print(f"P<=1/6: {workload_summary['samples_P_le_1_6']}/{workload_summary['n_samples']}")
    print(f"P>1/5: {workload_summary['samples_P_gt_1_5']}")
    print(f"P>1/6: {workload_summary['samples_P_gt_1_6']}")
    print(f"P=0: {workload_summary['samples_P_zero']}")
    print(f"C1<2000: {workload_summary['samples_C1_lt_2000']}")
    print(f"C1>3000: {workload_summary['samples_C1_gt_3000']}")
    print(f"context length distribution: {workload_summary.get('context_length_distribution', {})}")
    print(f"new tokens per step: {workload_summary.get('new_tokens_per_step', {})}")
    print(f"prefill: {time_summary['pct']['prefill']}%")
    print(f"semi-prefill: {time_summary['pct']['semi_prefill']}%")
    print(f"incremental: {time_summary['pct']['incremental']}%")
    print(f"decode: {time_summary['pct']['decode']}%")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
