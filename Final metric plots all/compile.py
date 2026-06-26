#!/usr/bin/env python3
"""
Compile debate + argument JSON files into merged JSONs grouped strictly by:

    (model_1, model_2, reasoning_effort_1, reasoning_effort_2, values1, values2)

Input layout (relative to this script):
../Archive/
  Reasoning vs Non-reasoning/
    <debate-folder>/
      0.json
      1.json
      ...
      arguments/
        0.arguments.json
        1.arguments.json
        ...
  Runs/
    Untitled/
      <debate-folder>/
        ...
  Small vs Large/
    <debate-folder>/
      ...

Output layout (created in the current directory):
Reasoning vs Non-reasoning/
  <compiled json files directly here>
Runs/Untitled/
  <compiled json files directly here>
Small vs Large/
  <compiled json files directly here>

Each compiled.json has:
{
  "model_1": ...,
  "model_2": ...,
  "reasoning_effort_1": ...,
  "reasoning_effort_2": ...,
  "values1": ...,
  "values2": ...,
  "list": [
    {
      "debate_id": ...,
      "final_verdict": ...,
      "model_1_strong_arguments_type": ...,
      "model_1_weak_arguments_type": ...,
      "model_2_strong_arguments_type": ...,
      "model_2_weak_arguments_type": ...,
      "model_1_disagreed_arguments_response": ...,
      "model_1_partiallyagreed_arguments_response": ...,
      "model_1_agreed_arguments_response": ...,
      "model_1_ignored_arguments_response": ...,
      "model_2_disagreed_arguments_response": ...,
      "model_2_partiallyagreed_arguments_response": ...,
      "model_2_agreed_arguments_response": ...,
      "model_2_ignored_arguments_response": ...
    },
    ...
  ]
}
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent
ARCHIVE_ROOT = ROOT.parent / "Archive"

TOP_LEVEL_CATEGORIES = [
    ("Reasoning vs Non-reasoning", ["Reasoning vs Non-reasoning"]),
    ("Runs", ["Runs/Untitled", "Runs"]),
    ("Small vs Large", ["Small vs Large"]),
]


def safe_load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_key(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def normalize_response(value: Any) -> str:
    key = normalize_key(value)
    mapping = {
        "disagreed": "disagreed",
        "partiallyagreed": "partiallyagreed",
        "partiallyagree": "partiallyagreed",
        "partiallyagreedto": "partiallyagreed",
        "agreed": "agreed",
        "ignored": "ignored",
    }
    return mapping.get(key, "other")


def sort_key(path: Path) -> Tuple[int, str]:
    stem = path.stem
    m = re.match(r"^(\d+)", stem)
    if m:
        return (int(m.group(1)), stem)
    return (10**18, stem)


def iter_debate_files(folder: Path) -> List[Path]:
    files = [
        p
        for p in folder.glob("*.json")
        if not p.name.endswith(".arguments.json") and p.name != "compiled.json"
    ]
    return sorted(files, key=sort_key)


def tuple_key_from_debate(debate: Dict[str, Any]) -> Tuple[str, str, str, str, str, str]:
    return (
        str(debate.get("model_1", "")),
        str(debate.get("model_2", "")),
        str(debate.get("reasoning_effort_1", "")),
        str(debate.get("reasoning_effort_2", "")),
        str(debate.get("values1", "")),
        str(debate.get("values2", "")),
    )


def compute_counts(arguments: List[Any]) -> Dict[str, int]:
    counts = Counter()

    for item in arguments:
        if not isinstance(item, dict):
            continue

        agent = str(item.get("agent", "")).strip().lower()
        if agent.startswith("agent1"):
            prefix = "model_1"
        elif agent.startswith("agent2"):
            prefix = "model_2"
        else:
            continue

        arg_type = str(item.get("type", "")).strip().lower()
        response = normalize_response(item.get("response"))

        if arg_type == "strong":
            counts[f"{prefix}_strong_arguments_type"] += 1
        elif arg_type == "weak":
            counts[f"{prefix}_weak_arguments_type"] += 1
        else:
            counts[f"{prefix}_other_arguments_type"] += 1

        counts[f"{prefix}_{response}_arguments_response"] += 1

    for prefix in ("model_1", "model_2"):
        counts.setdefault(f"{prefix}_strong_arguments_type", 0)
        counts.setdefault(f"{prefix}_weak_arguments_type", 0)
        counts.setdefault(f"{prefix}_other_arguments_type", 0)
        for response in ("disagreed", "partiallyagreed", "agreed", "ignored", "other"):
            counts.setdefault(f"{prefix}_{response}_arguments_response", 0)

    return dict(counts)


def build_compiled_entry(debate: Dict[str, Any], arguments: Dict[str, Any]) -> Dict[str, Any]:
    counts = compute_counts(arguments.get("arguments", []))

    return {
        "debate_id": debate.get("debate_id", arguments.get("debate_id")),
        "final_verdict": arguments.get("final_verdict"),
        "model_1_strong_arguments_type": counts["model_1_strong_arguments_type"],
        "model_1_weak_arguments_type": counts["model_1_weak_arguments_type"],
        "model_2_strong_arguments_type": counts["model_2_strong_arguments_type"],
        "model_2_weak_arguments_type": counts["model_2_weak_arguments_type"],
        "model_1_disagreed_arguments_response": counts["model_1_disagreed_arguments_response"],
        "model_1_partiallyagreed_arguments_response": counts["model_1_partiallyagreed_arguments_response"],
        "model_1_agreed_arguments_response": counts["model_1_agreed_arguments_response"],
        "model_1_ignored_arguments_response": counts["model_1_ignored_arguments_response"],
        "model_2_disagreed_arguments_response": counts["model_2_disagreed_arguments_response"],
        "model_2_partiallyagreed_arguments_response": counts["model_2_partiallyagreed_arguments_response"],
        "model_2_agreed_arguments_response": counts["model_2_agreed_arguments_response"],
        "model_2_ignored_arguments_response": counts["model_2_ignored_arguments_response"],
    }


def find_leaf_debate_folders(category_root: Path) -> List[Path]:
    if not category_root.exists():
        return []

    leaf_folders: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(category_root):
        current = Path(dirpath)
        has_debate_json = any(
            name.endswith(".json") and not name.endswith(".arguments.json") and name != "compiled.json"
            for name in filenames
        )
        if has_debate_json and (current / "arguments").is_dir():
            leaf_folders.append(current)

    return sorted(leaf_folders, key=lambda p: str(p))


def resolve_source_category_root(archive_root: Path, candidates: List[str]) -> Optional[Path]:
    for rel in candidates:
        p = archive_root / rel
        if p.exists():
            return p
    return None


def sanitize_filename_part(value: str) -> str:
    value = value.strip()
    value = re.sub(r'[\\/:*?"<>|]+', "_", value)
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "unknown"


def output_filename_from_key(key: Tuple[str, str, str, str, str, str]) -> str:
    model_1, model_2, effort_1, effort_2, values1, values2 = key
    parts = [
        sanitize_filename_part(model_1),
        sanitize_filename_part(effort_1),
        sanitize_filename_part(model_2),
        sanitize_filename_part(effort_2),
        sanitize_filename_part(values1),
        sanitize_filename_part(values2),
    ]
    return "_".join(parts) + ".json"


def main() -> None:
    if not ARCHIVE_ROOT.exists():
        raise FileNotFoundError(f"Archive folder not found: {ARCHIVE_ROOT}")

    total_compiled = 0

    for category_name, candidate_rels in TOP_LEVEL_CATEGORIES:
        source_category_root = resolve_source_category_root(ARCHIVE_ROOT, candidate_rels)
        if source_category_root is None:
            print(f"[skip] Missing category folder under {ARCHIVE_ROOT}: {candidate_rels}")
            continue

        output_category_root = ROOT / "model_value_pair"/ category_name
        output_category_root.mkdir(parents=True, exist_ok=True)

        grouped: DefaultDict[
            Tuple[str, str, str, str, str, str],
            List[Dict[str, Any]]
        ] = defaultdict(list)

        for debate_folder in find_leaf_debate_folders(source_category_root):
            arguments_folder = debate_folder / "arguments"

            for debate_path in iter_debate_files(debate_folder):
                stem = debate_path.stem
                arguments_path = arguments_folder / f"{stem}.arguments.json"

                if not arguments_path.exists():
                    print(f"[skip] Missing arguments file for debate {debate_path.name}: {arguments_path}")
                    continue

                debate = safe_load_json(debate_path)
                arguments = safe_load_json(arguments_path)

                if not isinstance(debate, dict):
                    print(f"[skip] Debate file is not a JSON object: {debate_path}")
                    continue
                if not isinstance(arguments, dict):
                    print(f"[skip] Arguments file is not a JSON object: {arguments_path}")
                    continue

                debate_id = debate.get("debate_id")
                arguments_debate_id = arguments.get("debate_id")
                if debate_id and arguments_debate_id and debate_id != arguments_debate_id:
                    print(
                        f"[warn] debate_id mismatch in {debate_path.name}: "
                        f"debate={debate_id} arguments={arguments_debate_id}"
                    )

                key = tuple_key_from_debate(debate)
                grouped[key].append(build_compiled_entry(debate, arguments))

        for key, merged_list in sorted(grouped.items(), key=lambda item: item[0]):
            model_1, model_2, effort_1, effort_2, values1, values2 = key

            compiled_output = {
                "model_1": model_1,
                "model_2": model_2,
                "reasoning_effort_1": effort_1,
                "reasoning_effort_2": effort_2,
                "values1": values1,
                "values2": values2,
                "list": merged_list,
            }

            out_path = output_category_root / output_filename_from_key(key)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(compiled_output, f, indent=2, ensure_ascii=False)
                f.write("\n")

            total_compiled += len(merged_list)
            print(f"[ok] {out_path} ({len(merged_list)} debates)")

    print(f"Done. Compiled {total_compiled} debate entries.")


if __name__ == "__main__":
    main()