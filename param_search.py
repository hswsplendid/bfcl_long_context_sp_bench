import argparse
import json
import statistics as stats
from pathlib import Path

import sp_config as CFG


def scan_run(run_dir: Path) -> list[dict]:
    rows = []
    timing_dir = run_dir / "timing"
    abc_dir = run_dir / "abc_segments"
    for timing_path in sorted(timing_dir.glob("*.json")):
        sample_id = timing_path.stem
        timing = json.load(open(timing_path, "r", encoding="utf-8"))
        abc_path = abc_dir / f"{sample_id}.json"
        abc = json.load(open(abc_path, "r", encoding="utf-8")) if abc_path.exists() else []
        steps = len(timing)
        if steps == 0:
            continue
        rate = len(abc) / steps
        rows.append(
            {
                "id": sample_id,
                "steps": steps,
                "compressions": len(abc),
                "P": rate,
                "P_pct": round(rate * 100, 2),
                "min_c1": min((event.get("C1_tokens_after", 0) for event in abc), default=None),
                "max_c1": max((event.get("C1_tokens_after", 0) for event in abc), default=None),
                "max_prompt_tokens": max((row.get("prompt_tokens", 0) for row in timing), default=0),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=CFG.RESULTS_DIR)
    parser.add_argument("--presets", nargs="+", default=None)
    args = parser.parse_args()

    presets = args.presets or [preset[0] for preset in CFG.PRESETS]
    table = {}
    for preset in presets:
        run_dir = args.results_dir / f"search_{preset}"
        if not run_dir.exists():
            continue
        table[preset] = scan_run(run_dir)

    print(f"{'preset':<28}{'N':>4}{'mean_P%':>10}{'med_P%':>10}{'max_P%':>10}{'<=1/5':>8}{'<=1/6':>8}{'min_C1':>10}{'max_C1':>10}{'C1>2000':>8}")
    for preset in presets:
        rows = table.get(preset, [])
        if not rows:
            continue
        pcts = [row["P_pct"] for row in rows]
        le_1_5 = sum(1 for row in rows if 0 < row["P"] <= CFG.P_TARGET_1_5)
        le_1_6 = sum(1 for row in rows if 0 < row["P"] <= CFG.P_TARGET_1_6)
        min_c1 = min((row["min_c1"] for row in rows if row["min_c1"] is not None), default=None)
        max_c1 = max((row.get("max_c1") for row in rows if row.get("max_c1") is not None), default=None)
        c1_ok = sum(
            1
            for row in rows
            if row.get("min_c1") is not None
            and row["min_c1"] > CFG.C1_MIN_TOKENS
        )
        print(
            f"{preset:<28}{len(rows):>4}{stats.mean(pcts):>10.2f}{stats.median(pcts):>10.2f}{max(pcts):>10.2f}"
            f"{le_1_5:>8}/{len(rows):<1}{le_1_6:>8}/{len(rows):<1}{str(min_c1):>10}{str(max_c1):>10}{c1_ok:>8}/{len(rows):<1}"
        )

    sample_ids = sorted({row["id"] for rows in table.values() for row in rows})
    if not sample_ids:
        return

    print("\nP% matrix")
    header = f"{'sample':<32}" + "".join(f"{preset:>16}" for preset in presets)
    print(header)
    for sample_id in sample_ids:
        line = f"{sample_id:<32}"
        for preset in presets:
            row = next((item for item in table.get(preset, []) if item["id"] == sample_id), None)
            line += f"{(str(row['P_pct']) if row else '-'):>16}"
        print(line)


if __name__ == "__main__":
    main()
