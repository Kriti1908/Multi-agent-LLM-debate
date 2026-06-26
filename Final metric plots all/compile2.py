#!/usr/bin/env python3
"""
Second-stage compiler.

Reads the JSON files produced by compile.py inside:

model_value_pair/
  Reasoning vs Non-reasoning/
  Runs/
  Small vs Large/

and aggregates them by:

    (model_1, reasoning_effort_1, model_2, reasoning_effort_2)

For each category folder, it writes one JSON file per pair directly inside:

model_pair/
  Reasoning vs Non-reasoning/
  Runs/
  Small vs Large/

Each output JSON contains the sums of all count fields across all value1/value2
pairs for that model/effort pair, plus:

    model_1_wins = number of debates with final_verdict == "Action_1" or "Action 1"
    model_2_wins = number of debates with final_verdict == "Action_2" or "Action 2"
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent
INPUT_ROOT = ROOT / "model_value_pair"
OUTPUT_ROOT = ROOT / "model_pair"

CATEGORIES = [
    "Reasoning vs Non-reasoning",
    "Runs",
    "Small vs Large",
]

COUNT_FIELDS = [
    "model_1_strong_arguments_type",
    "model_1_weak_arguments_type",
    "model_2_strong_arguments_type",
    "model_2_weak_arguments_type",
    "model_1_disagreed_arguments_response",
    "model_1_partiallyagreed_arguments_response",
    "model_1_agreed_arguments_response",
    "model_1_ignored_arguments_response",
    "model_2_disagreed_arguments_response",
    "model_2_partiallyagreed_arguments_response",
    "model_2_agreed_arguments_response",
    "model_2_ignored_arguments_response",
]


def safe_load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def normalize_key(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def parse_win(verdict: Any) -> Optional[str]:
    """
    Accepts Action_1, Action 1, Action1, Action_2, Action 2, Action2.
    Returns "model_1", "model_2", or None.
    """
    key = normalize_key(verdict)
    if key == "action1":
        return "model_1"
    if key == "action2":
        return "model_2"
    return None


def model_effort_key(data: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(data.get("model_1", "")),
        str(data.get("reasoning_effort_1", "")),
        str(data.get("model_2", "")),
        str(data.get("reasoning_effort_2", "")),
    )


def sanitize_filename_part(value: str) -> str:
    value = value.strip()
    value = re.sub(r'[\\/:*?"<>|]+', "_", value)
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "unknown"


def output_filename_from_key(key: Tuple[str, str, str, str]) -> str:
    model_1, effort_1, model_2, effort_2 = key
    parts = [
        sanitize_filename_part(model_1),
        sanitize_filename_part(effort_1),
        sanitize_filename_part(model_2),
        sanitize_filename_part(effort_2),
    ]
    return "_".join(parts) + ".json"


def iter_json_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.json"):
        if path.name == "compiled.json":
            continue
        yield path


def accumulate_group(files: List[Path]) -> Dict[str, Any]:
    totals = {field: 0 for field in COUNT_FIELDS}
    model_1_name = ""
    model_2_name = ""
    effort_1_name = ""
    effort_2_name = ""
    model_1_wins = 0
    model_2_wins = 0

    for path in files:
        data = safe_load_json(path)
        if not isinstance(data, dict):
            print(f"[skip] Not a JSON object: {path}")
            continue

        m1, e1, m2, e2 = model_effort_key(data)
        if not model_1_name and not model_2_name and not effort_1_name and not effort_2_name:
            model_1_name, effort_1_name, model_2_name, effort_2_name = m1, e1, m2, e2
        elif (m1, e1, m2, e2) != (model_1_name, effort_1_name, model_2_name, effort_2_name):
            print(f"[warn] Model/effort mismatch inside group for file: {path}")

        for field in COUNT_FIELDS:
            value = data.get(field, 0)
            if isinstance(value, (int, float)):
                totals[field] += int(value)

        entries = data.get("list", [])
        if not isinstance(entries, list):
            print(f"[skip] 'list' is not a list in: {path}")
            continue

        for entry in entries:
            if not isinstance(entry, dict):
                continue

            for field in COUNT_FIELDS:
                value = entry.get(field, 0)
                if isinstance(value, (int, float)):
                    totals[field] += int(value)

            winner = parse_win(entry.get("final_verdict"))
            if winner == "model_1":
                model_1_wins += 1
            elif winner == "model_2":
                model_2_wins += 1

    output: Dict[str, Any] = {
        "model_1": model_1_name,
        "reasoning_effort_1": effort_1_name,
        "model_2": model_2_name,
        "reasoning_effort_2": effort_2_name,
        **totals,
        "model_1_wins": model_1_wins,
        "model_2_wins": model_2_wins,
    }
    return output


def main() -> None:
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_ROOT}")

    for category in CATEGORIES:
        category_in = INPUT_ROOT / category
        if not category_in.exists():
            print(f"[skip] Missing category: {category_in}")
            continue

        category_out = OUTPUT_ROOT / category
        category_out.mkdir(parents=True, exist_ok=True)

        grouped: DefaultDict[Tuple[str, str, str, str], List[Path]] = defaultdict(list)

        for path in iter_json_files(category_in):
            data = safe_load_json(path)
            if not isinstance(data, dict):
                print(f"[skip] Not a JSON object: {path}")
                continue

            key = model_effort_key(data)
            grouped[key].append(path)

        for (model_1, effort_1, model_2, effort_2), files in sorted(grouped.items(), key=lambda item: item[0]):
            aggregated = accumulate_group(files)
            out_file = category_out / output_filename_from_key((model_1, effort_1, model_2, effort_2))
            safe_dump_json(out_file, aggregated)
            print(f"[ok] {out_file} ({len(files)} source file(s))")


if __name__ == "__main__":
    main()