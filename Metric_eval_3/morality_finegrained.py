from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


MODEL_LABELS = {
	"google/gemma-3-12b-it": "Gemma 3 12B",
	"openai/gpt-oss-20b": "GPT-Oss 20B",
	"meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 8B",
	"Qwen/Qwen3-8B": "Qwen3 8B",
}

TARGET_MODEL_ORDER = [
	"google/gemma-3-12b-it",
	"openai/gpt-oss-20b",
	"meta-llama/Llama-3.1-8B-Instruct",
	"Qwen/Qwen3-8B",
]

VALUE_ORDER = [
	"Care/Protection",
	"Communication/Cooperation",
	"Creativity",
	"Justice/Equality/Freedom/Respect",
	"Privacy",
	"Professionalism",
	"Sustainability",
	"Truthfulness",
	"Wisdom/Learning",
]


def load_json(path: Path) -> dict:
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def normalise_text(value: object) -> str:
	if value is None:
		return ""
	return str(value).strip().lower()


def action_from_verdict(verdict: object, values1: str, values2: str) -> int | None:
	verdict_text = normalise_text(verdict)

	if verdict_text in {"action 1", "action1", "1"}:
		return 1
	if verdict_text in {"action 2", "action2", "2"}:
		return 2
	if verdict_text == values1.lower():
		return 1
	if verdict_text == values2.lower():
		return 2
	return None


def is_relevant_folder(folder: Path) -> bool:
	if not folder.is_dir() or not folder.name.startswith("reasoning_"):
		return False
	if folder.name.endswith("_v3"):
		return False
	return True


def canonical_folder_name(folder: Path) -> str:
	name = folder.name
	if name.endswith("_v2"):
		return name[:-3]
	return name


def model_label(model_id: str) -> str:
	return MODEL_LABELS.get(model_id, model_id)


def value_pair_key(value_a: str, value_b: str) -> tuple[str, str] | None:
	if value_a not in VALUE_ORDER or value_b not in VALUE_ORDER:
		return None
	ia = VALUE_ORDER.index(value_a)
	ib = VALUE_ORDER.index(value_b)
	if ia >= ib:
		return value_a, value_b
	return value_b, value_a


def collect_group_dirs(root: Path) -> dict[str, tuple[Path, Path]]:
	folders = sorted(folder for folder in root.iterdir() if is_relevant_folder(folder))
	groups: dict[str, dict[str, Path | None]] = {}
	for folder in folders:
		group_name = canonical_folder_name(folder)
		entry = groups.setdefault(group_name, {"base": None, "v2": None})
		if folder.name.endswith("_v2"):
			entry["v2"] = folder
		else:
			entry["base"] = folder

	valid_groups: dict[str, tuple[Path, Path]] = {}
	for group_name, entry in groups.items():
		base_folder = entry["base"]
		v2_folder = entry["v2"]
		if base_folder is None or v2_folder is None:
			continue
		valid_groups[group_name] = (base_folder, v2_folder)
	return valid_groups


def process_group(
	base_folder: Path,
	v2_folder: Path,
	numerator: dict[tuple[str, str], dict[tuple[str, str], int]],
	denominator: dict[tuple[str, str], dict[tuple[str, str], int]],
) -> None:
	base_arguments_dir = base_folder / "arguments"
	v2_arguments_dir = v2_folder / "arguments"

	for debate_file in sorted(base_folder.glob("*.json")):
		if debate_file.name.endswith(".failed.json"):
			continue

		stem = debate_file.stem
		base_arguments_file = base_arguments_dir / f"{stem}.arguments.json"
		v2_debate_file = v2_folder / f"{stem}.json"
		v2_arguments_file = v2_arguments_dir / f"{stem}.arguments.json"
		if not (base_arguments_file.exists() and v2_debate_file.exists() and v2_arguments_file.exists()):
			continue

		base_data = load_json(debate_file)
		v2_data = load_json(v2_debate_file)
		base_arguments = load_json(base_arguments_file)
		v2_arguments = load_json(v2_arguments_file)

		model_1 = base_data.get("model_1")
		model_2 = base_data.get("model_2")
		if not isinstance(model_1, str) or not isinstance(model_2, str):
			continue
		if model_1 == model_2:
			# User asked for across-model pairs; skip self-play groups.
			continue

		base_values1 = base_data.get("values1")
		base_values2 = base_data.get("values2")
		v2_values1 = v2_data.get("values1")
		v2_values2 = v2_data.get("values2")
		if not all(isinstance(v, str) for v in [base_values1, base_values2, v2_values1, v2_values2]):
			continue

		cell_key = value_pair_key(str(base_values1), str(base_values2))
		if cell_key is None:
			continue

		base_action = action_from_verdict(base_arguments.get("final_verdict"), str(base_values1), str(base_values2))
		v2_action = action_from_verdict(v2_arguments.get("final_verdict"), str(v2_values1), str(v2_values2))
		if base_action is None or v2_action is None:
			continue

		pair_key = tuple(sorted((model_label(model_1), model_label(model_2))))
		denominator[pair_key][cell_key] += 1
		if base_action == v2_action:
			numerator[pair_key][cell_key] += 1


def build_matrix(
	pair_key: tuple[str, str],
	numerator: dict[tuple[str, str], dict[tuple[str, str], int]],
	denominator: dict[tuple[str, str], dict[tuple[str, str], int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
	bias = pd.DataFrame(np.nan, index=VALUE_ORDER, columns=VALUE_ORDER)
	counts = pd.DataFrame(0, index=VALUE_ORDER, columns=VALUE_ORDER, dtype=int)

	for row_idx, row_value in enumerate(VALUE_ORDER):
		for col_idx, col_value in enumerate(VALUE_ORDER):
			if row_idx < col_idx:
				continue
			cell_key = (row_value, col_value)
			den = denominator[pair_key].get(cell_key, 0)
			num = numerator[pair_key].get(cell_key, 0)
			counts.loc[row_value, col_value] = den
			if den > 0:
				bias.loc[row_value, col_value] = (num / den) * 100

	return bias, counts


def plot_lower_triangle(bias: pd.DataFrame, counts: pd.DataFrame, title: str, output_path: Path) -> None:
	sns.set_theme(style="white")
	fig, ax = plt.subplots(figsize=(13, 11))

	mask = np.triu(np.ones_like(bias, dtype=bool), k=1)
	annot = np.empty(bias.shape, dtype=object)
	for i in range(bias.shape[0]):
		for j in range(bias.shape[1]):
			if mask[i, j]:
				annot[i, j] = ""
				continue
			value = bias.iat[i, j]
			count = counts.iat[i, j]
			if np.isnan(value):
				annot[i, j] = f"n={count}" if count > 0 else ""
			else:
				annot[i, j] = f"{value:.1f}%\n(n={count})"

	sns.heatmap(
		bias,
		mask=mask,
		cmap="YlGnBu",
		vmin=0,
		vmax=100,
		linewidths=0.5,
		linecolor="white",
		annot=annot,
		fmt="",
		cbar_kws={"label": "Morality bias (%)"},
		ax=ax,
	)

	ax.set_title(title)
	ax.set_xlabel("Value B")
	ax.set_ylabel("Value A")
	plt.xticks(rotation=35, ha="right")
	plt.yticks(rotation=0)
	plt.tight_layout()
	fig.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close(fig)


def pair_slug(pair_key: tuple[str, str]) -> str:
	left = pair_key[0].lower().replace(" ", "_").replace("/", "-")
	right = pair_key[1].lower().replace(" ", "_").replace("/", "-")
	return f"{left}__vs__{right}"


def model_pair_order() -> list[tuple[str, str]]:
	labels = [model_label(model_id) for model_id in TARGET_MODEL_ORDER]
	pairs: list[tuple[str, str]] = []
	for i in range(len(labels)):
		for j in range(i + 1, len(labels)):
			pairs.append((labels[i], labels[j]))
	return pairs


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Generate fine-grained morality-bias 9x9 lower-triangle heatmaps per model pair. "
			"Numerator counts normal-vs-v2 samples where action outputs are identical."
		),
	)
	parser.add_argument(
		"roots",
		type=Path,
		nargs="*",
		default=[Path("Base"), Path("rerun"), Path("rerun2")],
		help="Root folders containing reasoning runs.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("outputs"),
		help="Directory where fine-grained plots and CSVs will be written.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	output_dir = args.output_dir
	output_dir.mkdir(parents=True, exist_ok=True)

	numerator: dict[tuple[str, str], dict[tuple[str, str], int]] = defaultdict(lambda: defaultdict(int))
	denominator: dict[tuple[str, str], dict[tuple[str, str], int]] = defaultdict(lambda: defaultdict(int))

	for root in args.roots:
		if not root.exists() or not root.is_dir():
			print(f"Skipping missing root: {root}")
			continue
		for _, (base_folder, v2_folder) in collect_group_dirs(root).items():
			process_group(base_folder, v2_folder, numerator, denominator)

	pair_tables: list[pd.DataFrame] = []
	for pair_key in model_pair_order():
		bias, counts = build_matrix(pair_key, numerator, denominator)
		slug = pair_slug(pair_key)
		plot_path = output_dir / f"morality_finegrained_{slug}.png"
		counts_path = output_dir / f"morality_finegrained_{slug}_counts.csv"
		bias_path = output_dir / f"morality_finegrained_{slug}.csv"

		plot_lower_triangle(
			bias,
			counts,
			title=f"Morality Bias (Action Match %) - {pair_key[0]} vs {pair_key[1]}",
			output_path=plot_path,
		)
		bias.to_csv(bias_path)
		counts.to_csv(counts_path)

		print(f"Saved plot: {plot_path}")
		print(f"Saved bias matrix: {bias_path}")
		print(f"Saved count matrix: {counts_path}")

		table = (
			bias.stack()
			.rename("morality_bias")
			.reset_index()
			.rename(columns={"level_0": "value_a", "level_1": "value_b"})
		)
		table["sample_count"] = [counts.loc[a, b] for a, b in zip(table["value_a"], table["value_b"])]
		table["pair"] = f"{pair_key[0]} vs {pair_key[1]}"
		table = table[table["value_a"].map(VALUE_ORDER.index) >= table["value_b"].map(VALUE_ORDER.index)]
		pair_tables.append(table)

	if pair_tables:
		all_cells = pd.concat(pair_tables, ignore_index=True)
		all_cells.to_csv(output_dir / "morality_finegrained_all_pairs_cells.csv", index=False)
		print(f"Saved combined cell table: {output_dir / 'morality_finegrained_all_pairs_cells.csv'}")


if __name__ == "__main__":
	main()
