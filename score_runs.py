#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

BENCH_ROOT = Path(__file__).parent.resolve()
BFCL_ROOT = Path("/root/gorilla/berkeley-function-call-leaderboard")
sys.path.insert(0, str(BFCL_ROOT))
sys.path.insert(0, str(BENCH_ROOT))

from bfcl_eval.eval_checker.eval_runner import _evaluate_single_multi_turn_entry  # noqa: E402
from bfcl_eval.model_handler.utils import (  # noqa: E402
    convert_to_function_call,
    default_decode_execute_prompting,
)

import sp_config as CFG  # noqa: E402
from run import load_dataset  # noqa: E402


class MixedModelResultHandler:
    def decode_execute(self, result, has_tool_call_tag: bool = False):
        if isinstance(result, list):
            return convert_to_function_call(result)
        if isinstance(result, str):
            return default_decode_execute_prompting(result, has_tool_call_tag)
        raise TypeError(f"unsupported model result item type: {type(result).__name__}")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_possible_answers() -> dict[str, list[list[str]]]:
    path = CFG.BFCL_ROOT / "bfcl_eval" / "data" / "possible_answer" / f"BFCL_v4_{CFG.TEST_CATEGORY}.json"
    return {row["id"]: row["ground_truth"] for row in load_jsonl(path)}


def resolve_data_file(run_dir: Path, explicit_data_file: Path | None) -> Path:
    if explicit_data_file is not None:
        return explicit_data_file
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        data_file = config.get("data_file")
        if data_file:
            return Path(data_file)
    return CFG.DATA_FILE


def load_model_results(run_dir: Path, sample_ids: list[str] | None) -> list[dict]:
    model_result_dir = run_dir / "model_results"
    if not model_result_dir.exists():
        raise FileNotFoundError(f"model_results directory not found: {model_result_dir}")

    if sample_ids:
        paths = [model_result_dir / f"{sample_id}.json" for sample_id in sample_ids]
    else:
        paths = sorted(model_result_dir.glob("*.json"))

    rows = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"model result not found: {path}")
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def pad_ground_truth(ground_truth: list[list[str]], result_turns: int) -> tuple[list[list[str]], int]:
    if result_turns <= len(ground_truth):
        return ground_truth, 0
    padding = result_turns - len(ground_truth)
    return ground_truth + [[] for _ in range(padding)], padding


def json_safe(value):
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def score_run(run_dir: Path, data_file: Path, sample_ids: list[str] | None) -> dict:
    dataset = load_dataset(data_file)
    possible_answers = load_possible_answers()
    handler = MixedModelResultHandler()
    result_rows = load_model_results(run_dir, sample_ids)

    entries = []
    correct_count = 0
    for model_result in result_rows:
        sample_id = model_result["id"]
        raw_result = model_result["result"]
        if sample_id not in dataset:
            raise KeyError(f"sample {sample_id} not found in data file {data_file}")
        if sample_id not in possible_answers:
            raise KeyError(f"sample {sample_id} not found in BFCL possible_answer")

        ground_truth, padded_turns = pad_ground_truth(possible_answers[sample_id], len(raw_result))
        entry_result = _evaluate_single_multi_turn_entry(
            handler,
            sample_id,
            raw_result,
            ground_truth,
            deepcopy(dataset[sample_id]),
            "Llama-3.3-70B-Instruct",
            CFG.TEST_CATEGORY,
        )
        valid = bool(entry_result.get("valid"))
        if valid:
            correct_count += 1

        score_entry = {
            "id": sample_id,
            "valid": valid,
            "score": 1 if valid else 0,
            "result_turns": len(raw_result),
            "ground_truth_turns_original": len(possible_answers[sample_id]),
            "ground_truth_turns_eval": len(ground_truth),
            "synthetic_ground_truth_padding_turns": padded_turns,
        }
        if not valid:
            error = entry_result.get("error", {})
            score_entry["error_type"] = error.get("error_type")
            score_entry["error_message"] = error.get("error_message")
            score_entry["details"] = json_safe(error.get("details"))
        entries.append(score_entry)

    accuracy = correct_count / len(entries) if entries else 0.0
    return {
        "run_dir": str(run_dir),
        "data_file": str(data_file),
        "samples": len(entries),
        "correct": correct_count,
        "accuracy": accuracy,
        "entries": entries,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Score native BFCL model_results saved by this runner")
    parser.add_argument("run_dir", type=Path, help="Path to results/<run_name>")
    parser.add_argument("--data-file", type=Path, default=None, help="Dataset used by the run; defaults to run_config.json")
    parser.add_argument("--ids", nargs="+", default=None, help="Optional sample ids to score")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    data_file = resolve_data_file(run_dir, args.data_file).resolve()
    report = score_run(run_dir, data_file, args.ids)

    output = args.output or (run_dir / "scores" / "bfcl_scores.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"saved {output}")
    print(f"accuracy {report['correct']}/{report['samples']} = {report['accuracy']:.4f}")
    for entry in report["entries"]:
        if entry["valid"]:
            print(f"{entry['id']} score=1")
        else:
            print(f"{entry['id']} score=0 error_type={entry.get('error_type')}")


if __name__ == "__main__":
    main()