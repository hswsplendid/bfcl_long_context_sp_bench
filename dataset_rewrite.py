#!/usr/bin/env python3
"""
Build an augmented BFCL multi_turn_long_context dataset for cw=32000 tests.

The transform keeps each original BFCL task intact, but adds deterministic,
varied background context blocks to user turns and pads short samples to at least
six turns with no-op context-maintenance turns.  The purpose is overhead and
compression-trigger measurement, not an official BFCL accuracy submission.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Iterable

BENCH_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(BENCH_ROOT))

import sp_config as CFG

FUNC_DOC_FILE_MAPPING = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
    "MemoryAPI_kv": "memory_kv.json",
    "MemoryAPI_vector": "memory_vector.json",
    "MemoryAPI_rec_sum": "memory_rec_sum.json",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_tokenizer(model_key: str):
    from transformers import AutoTokenizer

    model_info = CFG.MODEL_REGISTRY[model_key]
    return AutoTokenizer.from_pretrained(model_info["tokenizer_path"], trust_remote_code=True)


def token_count(tokenizer, text: str) -> int:
    if tokenizer is None:
        return max(1, len(text) // 4)
    return len(tokenizer.tokenize(text))


def message_text(messages: list[dict]) -> str:
    chunks = []
    for message in messages:
        chunks.append(str(message.get("role", "")))
        if message.get("content"):
            chunks.append(str(message["content"]))
        if message.get("tool_calls"):
            chunks.append(json.dumps(message["tool_calls"], ensure_ascii=False))
        if message.get("tool_call_id"):
            chunks.append(str(message["tool_call_id"]))
    return "\n".join(chunks)


def message_tokens(tokenizer, messages: list[dict]) -> int:
    return token_count(tokenizer, message_text(messages))


def populate_functions(entries: list[dict]) -> list[dict]:
    for entry in entries:
        if "involved_classes" not in entry:
            continue
        functions = []
        for class_name in entry["involved_classes"]:
            file_name = FUNC_DOC_FILE_MAPPING.get(class_name)
            if not file_name:
                continue
            doc_path = CFG.FUNC_DOC_DIR / file_name
            with doc_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        functions.append(json.loads(line))
        if functions:
            entry["function"] = functions
    return entries


def function_tokens(tokenizer, entry: dict) -> int:
    functions = entry.get("function") or []
    return token_count(tokenizer, json.dumps(functions, ensure_ascii=False))


def build_reference_block(sample_id: str, block_index: int, target_tokens: int, tokenizer) -> str:
    """Create a varied synthetic audit/context block close to target_tokens."""
    if target_tokens <= 0:
        return ""

    header = (
        f"\n\n[Long-context audit block {block_index} for {sample_id}]\n"
        "This block is background state for latency and compression testing. "
        "It is not a new tool request. Preserve the actual task instruction outside this block.\n"
    )
    rows = [header]
    token_budget = max(1, target_tokens)
    current_tokens = token_count(tokenizer, header)
    row_index = 0
    statuses = ["queued", "validated", "reconciled", "archived", "deferred"]
    while current_tokens < token_budget:
        status = statuses[(row_index + block_index) % len(statuses)]
        row = (
            f"Record {block_index:02d}-{row_index:04d}: entity=user_{(row_index * 17 + block_index) % 997}, "
            f"operation=state_sync, status={status}, ledger={sample_id}, "
            f"inputs=[alpha:{row_index % 31}, beta:{(row_index * 7) % 43}, gamma:{(row_index * 13) % 59}], "
            f"note='retain this as inert historical context; do not call tools for this record'.\n"
        )
        rows.append(row)
        current_tokens += token_count(tokenizer, row)
        row_index += 1
    return "".join(rows)


def extend_to_min_turns(entry: dict, min_turns: int) -> None:
    questions = entry.setdefault("question", [])
    while len(questions) < min_turns:
        synthetic_index = len(questions)
        questions.append(
            [
                {
                    "role": "user",
                    "content": (
                        f"Context-maintenance turn {synthetic_index}: no new external action is requested. "
                        "Briefly acknowledge that the previous task state and audit context remain available. "
                        "Do not call tools unless a previous tool call is still unresolved."
                    ),
                }
            ]
        )


def augment_entry(entry: dict, tokenizer, initial_target: int, growth_tokens: int, min_turns: int) -> dict:
    augmented = deepcopy(entry)
    extend_to_min_turns(augmented, min_turns)

    sample_id = augmented["id"]
    tool_schema_tokens = function_tokens(tokenizer, augmented)
    question_messages = [message for turn in augmented["question"] for message in turn]
    base_message_tokens = message_tokens(tokenizer, question_messages[:1])
    initial_context_gap = max(0, initial_target - tool_schema_tokens - base_message_tokens)

    first_user = augmented["question"][0][0]
    first_user["content"] = (
        first_user.get("content", "")
        + build_reference_block(sample_id, 0, initial_context_gap, tokenizer)
    )

    for turn_index, turn_messages in enumerate(augmented["question"][1:], start=1):
        if not turn_messages:
            continue
        growth_block = build_reference_block(sample_id, turn_index, growth_tokens, tokenizer)
        turn_messages[0]["content"] = turn_messages[0].get("content", "") + growth_block

    augmented["sp_bench_transform"] = {
        "source_id": entry["id"],
        "initial_target_tokens_including_tools": initial_target,
        "turn_growth_tokens": growth_tokens,
        "min_turns": min_turns,
        "tool_schema_tokens_est": tool_schema_tokens,
        "intended_context_window": 32000,
        "intended_threshold": 30000,
        "intended_c1_range": [CFG.C1_MIN_TOKENS, CFG.C1_MAX_TOKENS],
        "purpose": "semi-prefill overhead/compression-trigger workload measurement",
    }
    return augmented


def build_dataset(args) -> list[dict]:
    rng = random.Random(args.seed)
    tokenizer = load_tokenizer(args.model) if not args.no_tokenizer else None
    entries = populate_functions(load_jsonl(args.input))
    if args.ids:
        id_set = set(args.ids)
        entries = [entry for entry in entries if entry["id"] in id_set]
    if args.shuffle:
        rng.shuffle(entries)
    if args.max_samples and args.max_samples > 0:
        entries = entries[: args.max_samples]

    return [
        augment_entry(
            entry,
            tokenizer=tokenizer,
            initial_target=args.initial_target_tokens,
            growth_tokens=args.turn_growth_tokens,
            min_turns=args.min_turns,
        )
        for entry in entries
    ]


def summarize(rows: list[dict], tokenizer, model_key: str) -> list[dict]:
    report = []
    for entry in populate_functions(deepcopy(rows)):
        messages = [message for turn in entry.get("question", []) for message in turn]
        report.append(
            {
                "id": entry["id"],
                "turns": len(entry.get("question", [])),
                "tool_schema_tokens_est": function_tokens(tokenizer, entry),
                "first_turn_message_tokens_est": message_tokens(tokenizer, entry.get("question", [[]])[0]),
                "all_user_message_tokens_est": message_tokens(tokenizer, messages),
                "model": model_key,
            }
        )
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Build augmented BFCL long-context dataset")
    parser.add_argument("--input", type=Path, default=CFG.ORIGINAL_DATA_FILE)
    parser.add_argument("--output", type=Path, default=CFG.AUGMENTED_DATA_FILE)
    parser.add_argument("--sample-ids-out", type=Path, default=CFG.SAMPLE_IDS_FILE)
    parser.add_argument("--summary-out", type=Path, default=CFG.BENCH_ROOT / "data" / "dataset_summary.json")
    parser.add_argument("--model", choices=sorted(CFG.MODEL_REGISTRY), default=CFG.DEFAULT_MODEL)
    parser.add_argument("--initial-target-tokens", type=int, default=CFG.AUGMENT_INITIAL_TARGET_TOKENS)
    parser.add_argument("--turn-growth-tokens", type=int, default=CFG.AUGMENT_TURN_GROWTH_TOKENS)
    parser.add_argument("--min-turns", type=int, default=CFG.AUGMENT_MIN_TURNS)
    parser.add_argument("--max-samples", type=int, default=CFG.AUGMENT_MAX_SAMPLES)
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=CFG.AUGMENT_SEED)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--no-tokenizer", action="store_true", help="Use chars/4 estimate instead of model tokenizer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_dataset(args)
    write_jsonl(args.output, rows)
    args.sample_ids_out.parent.mkdir(parents=True, exist_ok=True)
    with args.sample_ids_out.open("w", encoding="utf-8") as handle:
        json.dump([row["id"] for row in rows], handle, indent=2, ensure_ascii=False)

    tokenizer = None if args.no_tokenizer else load_tokenizer(args.model)
    summary = summarize(rows[: min(20, len(rows))], tokenizer, args.model)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_out.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "output": str(args.output),
                "samples": len(rows),
                "preview": summary,
                "target": {
                    "context_window": 32000,
                    "threshold": 30000,
                    "c1_min": CFG.C1_MIN_TOKENS,
                    "c1_max": CFG.C1_MAX_TOKENS,
                    "p_max_1_5": CFG.P_TARGET_1_5,
                    "p_max_1_6": CFG.P_TARGET_1_6,
                },
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(f"wrote {len(rows)} samples -> {args.output}")
    print(f"wrote sample ids -> {args.sample_ids_out}")
    print(f"wrote summary -> {args.summary_out}")


if __name__ == "__main__":
    main()
