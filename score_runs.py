#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
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


def _safe_literal(value):
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def _call_name(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def parse_execution_call(call: str) -> dict:
    """Parse `func(a=1)` into a normalized diagnostic structure."""
    tree = ast.parse(call, mode="eval")
    node = tree.body
    if not isinstance(node, ast.Call):
        raise ValueError(f"not a function call: {call}")
    args = {}
    for index, arg in enumerate(node.args):
        args[f"__arg{index}"] = _safe_literal(ast.unparse(arg))
    for keyword in node.keywords:
        if keyword.arg is None:
            continue
        args[keyword.arg] = _safe_literal(ast.unparse(keyword.value))
    return {"name": _call_name(node.func), "args": args, "raw": call}


def _call_signature(call: dict) -> tuple:
    args = tuple(sorted((key, repr(value)) for key, value in call.get("args", {}).items()))
    return call.get("name", ""), args


def _f1(predicted: Counter, expected: Counter) -> dict:
    pred_total = sum(predicted.values())
    exp_total = sum(expected.values())
    true_positive = sum((predicted & expected).values())
    precision = true_positive / pred_total if pred_total else (1.0 if exp_total == 0 else 0.0)
    recall = true_positive / exp_total if exp_total else (1.0 if pred_total == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "true_positive": true_positive,
        "predicted": pred_total,
        "expected": exp_total,
    }


def decode_model_calls_by_turn(handler: MixedModelResultHandler, raw_result: list) -> tuple[list[list[str]], list[dict]]:
    decoded_turns = []
    errors = []
    for turn_index, turn_result in enumerate(raw_result):
        step_results = turn_result if isinstance(turn_result, list) else [turn_result]
        turn_calls = []
        for step_index, step_result in enumerate(step_results):
            try:
                turn_calls.extend(handler.decode_execute(step_result))
            except Exception as exc:
                errors.append(
                    {
                        "turn": turn_index,
                        "step": step_index,
                        "error": str(exc),
                        "raw_result": json_safe(step_result),
                    }
                )
        decoded_turns.append(turn_calls)
    return decoded_turns, errors


def fine_grained_quality(raw_result: list, expected_answers: list[list[str]], handler: MixedModelResultHandler) -> dict:
    decoded_turns, decode_errors = decode_model_calls_by_turn(handler, raw_result)

    parsed_predicted = []
    parsed_expected = []
    parse_errors = []
    for turn_index, calls in enumerate(decoded_turns):
        for call in calls:
            try:
                item = parse_execution_call(call)
                item["turn"] = turn_index
                parsed_predicted.append(item)
            except Exception as exc:
                parse_errors.append({"turn": turn_index, "call": call, "error": str(exc)})
    for turn_index, calls in enumerate(expected_answers):
        for call in calls:
            try:
                item = parse_execution_call(call)
                item["turn"] = turn_index
                parsed_expected.append(item)
            except Exception as exc:
                parse_errors.append({"turn": turn_index, "expected_call": call, "error": str(exc)})

    predicted_names = Counter(call["name"] for call in parsed_predicted)
    expected_names = Counter(call["name"] for call in parsed_expected)
    predicted_signatures = Counter(_call_signature(call) for call in parsed_predicted)
    expected_signatures = Counter(_call_signature(call) for call in parsed_expected)

    predicted_arg_keys = Counter(
        (call["name"], key) for call in parsed_predicted for key in call.get("args", {})
    )
    expected_arg_keys = Counter(
        (call["name"], key) for call in parsed_expected for key in call.get("args", {})
    )
    predicted_arg_values = Counter(
        (call["name"], key, repr(value))
        for call in parsed_predicted
        for key, value in call.get("args", {}).items()
    )
    expected_arg_values = Counter(
        (call["name"], key, repr(value))
        for call in parsed_expected
        for key, value in call.get("args", {}).items()
    )

    predicted_sequence = [
        [call["name"] for call in parsed_predicted if call["turn"] == turn_index]
        for turn_index in range(len(decoded_turns))
    ]
    expected_sequence = [
        [call["name"] for call in parsed_expected if call["turn"] == turn_index]
        for turn_index in range(len(expected_answers))
    ]

    return {
        "decoded_model_calls": decoded_turns,
        "expected_calls": expected_answers,
        "decode_error_count": len(decode_errors),
        "decode_errors": decode_errors,
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors,
        "predicted_call_count": len(parsed_predicted),
        "expected_call_count": len(parsed_expected),
        "function_name": _f1(predicted_names, expected_names),
        "call_exact": _f1(predicted_signatures, expected_signatures),
        "argument_key": _f1(predicted_arg_keys, expected_arg_keys),
        "argument_value": _f1(predicted_arg_values, expected_arg_values),
        "call_count_exact": len(parsed_predicted) == len(parsed_expected),
        "function_order_exact": predicted_sequence == expected_sequence,
    }


def average_quality_breakdown(entries: list[dict]) -> dict:
    metrics = [entry.get("fine_grained_quality", {}) for entry in entries]
    result = {}
    for key in ("function_name", "call_exact", "argument_key", "argument_value"):
        f1_values = [m.get(key, {}).get("f1") for m in metrics if m.get(key, {}).get("f1") is not None]
        result[f"{key}_f1_mean"] = round(sum(f1_values) / len(f1_values), 6) if f1_values else None
    result["call_count_exact_rate"] = (
        sum(1 for m in metrics if m.get("call_count_exact")) / len(metrics) if metrics else 0.0
    )
    result["function_order_exact_rate"] = (
        sum(1 for m in metrics if m.get("function_order_exact")) / len(metrics) if metrics else 0.0
    )
    return result


def score_run(
    run_dir: Path,
    data_file: Path,
    sample_ids: list[str] | None,
    first_turn_only: bool = False,
) -> dict:
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

        eval_result = raw_result[:1] if first_turn_only else raw_result
        expected_answers = possible_answers[sample_id][:1] if first_turn_only else possible_answers[sample_id]
        eval_entry = deepcopy(dataset[sample_id])
        if first_turn_only:
            eval_entry["question"] = eval_entry.get("question", [])[:1]

        ground_truth, padded_turns = pad_ground_truth(expected_answers, len(eval_result))
        entry_result = _evaluate_single_multi_turn_entry(
            handler,
            sample_id,
            eval_result,
            ground_truth,
            eval_entry,
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
            "metric": "first_turn_accuracy" if first_turn_only else "full_multi_turn_accuracy",
            "result_turns": len(eval_result),
            "ground_truth_turns_original": len(expected_answers),
            "ground_truth_turns_eval": len(ground_truth),
            "synthetic_ground_truth_padding_turns": padded_turns,
            "fine_grained_quality": fine_grained_quality(eval_result, expected_answers, handler),
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
        "metric": "first_turn_accuracy" if first_turn_only else "full_multi_turn_accuracy",
        "quality_breakdown": average_quality_breakdown(entries),
        "entries": entries,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Score native BFCL model_results saved by this runner")
    parser.add_argument("run_dir", type=Path, help="Path to results/<run_name>")
    parser.add_argument("--data-file", type=Path, default=None, help="Dataset used by the run; defaults to run_config.json")
    parser.add_argument("--ids", nargs="+", default=None, help="Optional sample ids to score")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--first-turn-only", action="store_true", help="Score only turn 0 against turn-0 ground truth")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    data_file = resolve_data_file(run_dir, args.data_file).resolve()
    report = score_run(run_dir, data_file, args.ids, first_turn_only=args.first_turn_only)

    output = args.output or (run_dir / "scores" / "bfcl_scores.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"saved {output}")
    print(f"{report['metric']} {report['correct']}/{report['samples']} = {report['accuracy']:.4f}")
    for entry in report["entries"]:
        if entry["valid"]:
            print(f"{entry['id']} score=1")
        else:
            print(f"{entry['id']} score=0 error_type={entry.get('error_type')}")


if __name__ == "__main__":
    main()