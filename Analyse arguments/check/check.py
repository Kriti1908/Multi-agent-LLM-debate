#!/usr/bin/env python3

import json
from pathlib import Path

# ============================================================
# CHANGE THIS
# ============================================================

INPUT_FOLDER = "../../Archive/Runs/Untitled/reasoning_gemma-3-12b-it_gemma-3-12b-it/arguments"

# ============================================================
# CHECK JSON FILES
# ============================================================

input_path = Path(INPUT_FOLDER)

json_files = sorted(input_path.glob("*.json"))

print(f"\nFound {len(json_files)} JSON files\n")

bad_files = []

for json_file in json_files:

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

    except Exception as e:
        print(f"[JSON LOAD ERROR] {json_file}")
        print(f"  -> {e}\n")
        bad_files.append(json_file)
        continue

    arguments = data.get("arguments", None)

    # --------------------------------------------------------
    # Check arguments field itself
    # --------------------------------------------------------

    if not isinstance(arguments, list):

        print(f"[BAD ARGUMENTS FIELD] {json_file}")
        print(f"  arguments type = {type(arguments)}\n")

        bad_files.append(json_file)
        continue

    # --------------------------------------------------------
    # Check each argument entry
    # --------------------------------------------------------

    file_has_issue = False

    for idx, arg in enumerate(arguments):

        # Must be dict
        if not isinstance(arg, dict):

            print(f"[BAD ENTRY] {json_file}")
            print(f"  index = {idx}")
            print(f"  type  = {type(arg)}")
            print(f"  value = {arg}\n")

            file_has_issue = True
            continue

        # Must contain string argument
        text = arg.get("argument", None)

        if not isinstance(text, str):

            print(f"[BAD ARGUMENT TEXT] {json_file}")
            print(f"  index = {idx}")
            print(f"  argument type = {type(text)}")
            print(f"  value = {text}\n")

            file_has_issue = True
            continue

    if file_has_issue:
        bad_files.append(json_file)

# ============================================================
# SUMMARY
# ============================================================

print("===================================================")
print("CHECK COMPLETE")
print("===================================================")

print(f"\nTotal problematic files: {len(set(bad_files))}\n")

if len(bad_files) > 0:
    print("Problematic files:\n")

    for bf in sorted(set(bad_files)):
        print(bf)
else:
    print("No malformed files found.")