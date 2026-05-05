import argparse
import json
import sys
from pathlib import Path

from bench_handler import BFCLLongContextSemiPrefillBench, resolve_preset, run_dirs
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


def populate_functions(entries: list[dict]) -> list[dict]:
    doc_root = CFG.BFCL_ROOT / "bfcl_eval" / "data" / "multi_turn_func_doc"
    for entry in entries:
        if "involved_classes" not in entry:
            continue
        functions = []
        for class_name in entry["involved_classes"]:
            file_name = FUNC_DOC_FILE_MAPPING.get(class_name)
            if not file_name:
                continue
            doc_path = doc_root / file_name
            with open(doc_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    functions.append(json.loads(line))
        if functions:
            entry["function"] = functions
    return entries


def load_dataset(data_file: Path | None = None) -> dict:
    entries = []
    source = data_file or CFG.DATA_FILE
    if not source.exists() and source == CFG.AUGMENTED_DATA_FILE:
        print(f"[data] augmented dataset not found, falling back to original: {CFG.ORIGINAL_DATA_FILE}")
        source = CFG.ORIGINAL_DATA_FILE
    with open(source, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entries.append(json.loads(line))
    entries = populate_functions(entries)
    return {entry["id"]: entry for entry in entries}


def load_triggered_ids(data_file: Path | None = None) -> list[str]:
    if CFG.SAMPLE_IDS_FILE.exists():
        with open(CFG.SAMPLE_IDS_FILE, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return list(load_dataset(data_file).keys())


def configure_model(model_key: str) -> dict:
    model_info = CFG.MODEL_REGISTRY[model_key]
    CFG.MODEL_KEY = model_key
    CFG.MODEL_PATH = model_info["model_path"]
    CFG.TOKENIZER_PATH = model_info["tokenizer_path"]
    return model_info


def maybe_prepare_data(args) -> None:
    if not args.prepare_data:
        return
    from dataset_rewrite import build_dataset, write_jsonl, load_tokenizer, summarize

    rewrite_args = argparse.Namespace(
        input=CFG.ORIGINAL_DATA_FILE,
        output=args.data_file,
        sample_ids_out=CFG.SAMPLE_IDS_FILE,
        summary_out=CFG.BENCH_ROOT / "data" / "dataset_summary.json",
        model=args.model,
        initial_target_tokens=args.initial_target_tokens,
        turn_growth_tokens=args.turn_growth_tokens,
        min_turns=args.min_turns,
        max_samples=args.prepare_max_samples,
        ids=args.ids if getattr(args, "ids", None) else None,
        seed=CFG.AUGMENT_SEED,
        shuffle=False,
        no_tokenizer=False,
    )
    rows = build_dataset(rewrite_args)
    write_jsonl(args.data_file, rows)
    CFG.SAMPLE_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = []
    if CFG.SAMPLE_IDS_FILE.exists():
        with open(CFG.SAMPLE_IDS_FILE, "r", encoding="utf-8") as handle:
            existing_ids = json.load(handle)
    sample_ids = [row["id"] for row in rows] or existing_ids
    with open(CFG.SAMPLE_IDS_FILE, "w", encoding="utf-8") as handle:
        json.dump(sample_ids, handle, indent=2, ensure_ascii=False)
    tokenizer = load_tokenizer(args.model)
    summary = summarize(rows[: min(20, len(rows))], tokenizer, args.model)
    rewrite_args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    with open(rewrite_args.summary_out, "w", encoding="utf-8") as handle:
        json.dump({"samples": len(rows), "preview": summary}, handle, indent=2, ensure_ascii=False)
    print(f"[prepare-data] wrote {len(rows)} samples -> {args.data_file}")


def write_run_meta(run_name: str, preset: dict, sample_ids: list[str], model_key: str, data_file: Path) -> None:
    dirs = run_dirs(run_name)
    dirs["base"].mkdir(parents=True, exist_ok=True)
    with open(dirs["config_file"], "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": model_key,
                "model_path": CFG.MODEL_REGISTRY[model_key]["model_path"],
                "data_file": str(data_file),
                "proxy_url": CFG.PROXY_URL,
                "preset": preset,
                "targets": {
                    "p_max_1_5": CFG.P_TARGET_1_5,
                    "p_max_1_6": CFG.P_TARGET_1_6,
                    "c1_min_tokens": CFG.C1_MIN_TOKENS,
                    "c1_max_tokens": CFG.C1_MAX_TOKENS,
                },
            },
            handle,
            indent=2,
        )
    with open(dirs["samples_file"], "w", encoding="utf-8") as handle:
        json.dump(sample_ids, handle, indent=2)


def run_once(run_name: str, preset_name: str, sample_ids: list[str], force: bool, model_key: str, data_file: Path) -> None:
    preset = resolve_preset(preset_name)
    write_run_meta(run_name, preset, sample_ids, model_key, data_file)
    bench = BFCLLongContextSemiPrefillBench(run_name, preset, model_key=model_key)
    dataset = load_dataset(data_file)

    summaries = []
    for sample_id in sample_ids:
        if sample_id not in dataset:
            print(f"[skip] missing sample {sample_id}")
            continue
        if bench.sample_finished(sample_id) and not force:
            print(f"[skip] {sample_id} already complete")
            continue
        print(f"[run] {run_name} :: {sample_id}")
        summary = bench.run_sample(dataset[sample_id])
        summaries.append(summary)

    summary_path = run_dirs(run_name)["summary_file"]
    with open(summary_path, "w", encoding="utf-8") as handle:
        for row in summaries:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[done] {run_name}")


def cmd_search(args):
    configure_model(args.model)
    maybe_prepare_data(args)
    presets = args.presets or [preset[0] for preset in CFG.PRESETS]
    sample_ids = CFG.SEARCH_SAMPLE_IDS[: args.limit] if args.limit else list(CFG.SEARCH_SAMPLE_IDS)
    for preset in presets:
        run_once(f"search_{args.model}_{preset}", preset, sample_ids, force=args.force, model_key=args.model, data_file=args.data_file)


def cmd_full(args):
    configure_model(args.model)
    maybe_prepare_data(args)
    sample_ids = args.ids or load_triggered_ids(args.data_file)
    if args.limit:
        sample_ids = sample_ids[: args.limit]
    run_name = args.run_name or f"full_{args.model}_{args.preset}"
    run_once(run_name, args.preset, sample_ids, force=args.force, model_key=args.model, data_file=args.data_file)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(run_parser):
        run_parser.add_argument("--model", choices=sorted(CFG.MODEL_REGISTRY), default=CFG.DEFAULT_MODEL)
        run_parser.add_argument("--data-file", type=Path, default=CFG.DATA_FILE)
        run_parser.add_argument("--results-root", type=Path, default=None)
        run_parser.add_argument("--prepare-data", action="store_true", help="Build augmented dataset before running")
        run_parser.add_argument("--prepare-max-samples", type=int, default=CFG.AUGMENT_MAX_SAMPLES)
        run_parser.add_argument("--initial-target-tokens", type=int, default=CFG.AUGMENT_INITIAL_TARGET_TOKENS)
        run_parser.add_argument("--turn-growth-tokens", type=int, default=CFG.AUGMENT_TURN_GROWTH_TOKENS)
        run_parser.add_argument("--min-turns", type=int, default=CFG.AUGMENT_MIN_TURNS)

    search = sub.add_parser("search")
    add_common(search)
    search.add_argument("--presets", nargs="+", default=None)
    search.add_argument("--limit", type=int, default=None)
    search.add_argument("--force", action="store_true")

    full = sub.add_parser("full")
    add_common(full)
    full.add_argument("--preset", default=CFG.PRIMARY_PRESET)
    full.add_argument("--ids", nargs="+", default=None)
    full.add_argument("--limit", type=int, default=None)
    full.add_argument("--run-name", default=None)
    full.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if getattr(args, "results_root", None):
        CFG.RESULTS_DIR = args.results_root.resolve()
    if args.cmd == "search":
        cmd_search(args)
    elif args.cmd == "full":
        cmd_full(args)
    else:
        print(f"unknown command: {args.cmd}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
