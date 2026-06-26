from __future__ import annotations

import argparse
import json
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


def winner_from_action(action: int, model_1: str, model_2: str, swap_action_mapping: bool = False) -> str:
	if action == 1:
		return model_2 if swap_action_mapping else model_1
	return model_1 if swap_action_mapping else model_2


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


def model_label(model_id: object) -> str:
	if not isinstance(model_id, str):
		return "Unknown"
	return MODEL_LABELS.get(model_id, model_id)

def extract_models_from_folder(folder_name: str) -> tuple[str, str]:
	# example: reasoning_gpt-oss-20b_low_vs_gpt-oss-20b_high
	name = folder_name.replace("reasoning_", "")
	if name.endswith("_v2"):
		name = name[:-3]
	parts = name.split("_vs_")
	if len(parts) != 2:
		return None, None
	return parts[0], parts[1]


def collect_group_metrics(base_folder: Path, v2_folder: Path | None) -> dict[str, object]:
	base_arguments_dir = base_folder / "arguments"
	v2_arguments_dir = v2_folder / "arguments" if v2_folder else None

	total_base_samples = 0
	valid_base_samples = 0
	model_1_win_count = 0
	model_2_win_count = 0
	consensus_count = 0
	consensus_samples = 0
	model_1 = None
	model_2 = None

	for debate_file in sorted(base_folder.glob("*.json")):
		if debate_file.name.endswith(".failed.json"):
			continue

		stem = debate_file.stem
		base_arguments_file = base_arguments_dir / f"{stem}.arguments.json"
		if not base_arguments_file.exists():
			continue

		base_data = load_json(debate_file)
		base_arguments = load_json(base_arguments_file)
		# ORIGINAL model ids
		base_model_1 = base_data.get("model_1")
		base_model_2 = base_data.get("model_2")

		# Extract low/high from folder name
		folder_model_1, folder_model_2 = extract_models_from_folder(base_folder.name)

		# If extraction works, override model names to distinguish variants
		if folder_model_1 and folder_model_2:
			model_1 = f"{base_model_1}_{folder_model_1.split('_')[-1]}"
			model_2 = f"{base_model_2}_{folder_model_2.split('_')[-1]}"
		else:
			model_1 = base_model_1
			model_2 = base_model_2

		values1 = base_data.get("values1")
		values2 = base_data.get("values2")
		if not isinstance(values1, str) or not isinstance(values2, str):
			continue

		base_action = action_from_verdict(base_arguments.get("final_verdict"), values1, values2)
		if base_action is None:
			total_base_samples += 1
			continue

		total_base_samples += 1
		valid_base_samples += 1
		base_winner = winner_from_action(base_action, str(model_1), str(model_2), swap_action_mapping=False)
		if base_winner == model_1:
			model_1_win_count += 1
		elif base_winner == model_2:
			model_2_win_count += 1

		if not v2_folder or v2_arguments_dir is None:
			continue

		v2_debate_file = v2_folder / f"{stem}.json"
		v2_arguments_file = v2_arguments_dir / f"{stem}.arguments.json"
		if not (v2_debate_file.exists() and v2_arguments_file.exists()):
			continue

		v2_data = load_json(v2_debate_file)
		v2_arguments = load_json(v2_arguments_file)
		v2_values1 = v2_data.get("values1")
		v2_values2 = v2_data.get("values2")
		if not isinstance(v2_values1, str) or not isinstance(v2_values2, str):
			continue

		v2_action = action_from_verdict(v2_arguments.get("final_verdict"), v2_values1, v2_values2)
		if v2_action is None:
			continue

		# Morality bias is action agreement rate between normal and v2 runs.
		consensus_samples += 1
		if base_action == v2_action:
			consensus_count += 1

	persuasion_score = np.nan
	morality_bias = np.nan
	if valid_base_samples:
		persuasion_score = ((model_1_win_count - model_2_win_count) / valid_base_samples) * 100
	if consensus_samples:
		morality_bias = (consensus_count / consensus_samples) * 100

	return {
		"group": canonical_folder_name(base_folder),
		"base_folder": base_folder.name,
		"v2_folder": v2_folder.name if v2_folder else "",
		"model_1": model_1,
		"model_2": model_2,
		"model_1_label": model_label(model_1),
		"model_2_label": model_label(model_2),
		"sample_count": total_base_samples,
		"valid_sample_count": valid_base_samples,
		"consensus_sample_count": consensus_samples,
		"model_1_win_count": model_1_win_count,
		"model_2_win_count": model_2_win_count,
		"consensus_count": consensus_count,
		"persuasion_score": persuasion_score,
		"morality_bias": morality_bias,
	}


def weighted_average(frame: pd.DataFrame, value_column: str, weight_column: str) -> float:
	valid = frame[frame[value_column].notna() & frame[weight_column].notna()]
	if valid.empty:
		return float("nan")
	weights = valid[weight_column].to_numpy(dtype=float)
	values = valid[value_column].to_numpy(dtype=float)
	return float(np.average(values, weights=weights))


def pair_key_and_label(model_1_label: str, model_2_label: str) -> tuple[tuple[str, str], str]:
	pair = tuple(sorted((model_1_label, model_2_label)))
	return pair, f"{pair[0]} vs {pair[1]}"


def prettify_group_label(group_name: str) -> str:
	label = group_name.replace("reasoning_", "")
	label = label.replace("_vs_", " vs ")
	label = label.replace("_", " ")
	return label


def split_vs_label(label: str) -> tuple[str, str]:
	parts = [part.strip() for part in label.split(" vs ", 1)]
	if len(parts) == 2:
		return parts[0], parts[1]
	return label, ""


def aggregate_pair_morality(summary: pd.DataFrame, include_self_pairs: bool = False, source_name: str = "") -> pd.DataFrame:
	rows: list[dict[str, object]] = []

	for _, row in summary.iterrows():
		m1 = row.get("model_1_label")
		m2 = row.get("model_2_label")
		group_name = row.get("group")
		if not isinstance(m1, str) or not isinstance(m2, str):
			continue
		if m1 == m2:
			if not include_self_pairs:
				continue
			if not isinstance(group_name, str):
				continue
			pair_key = (f"self::{group_name}", "")
			pair_label = prettify_group_label(group_name)
			left_model, right_model = split_vs_label(pair_label)
		else:
			pair_key, pair_label = pair_key_and_label(m1, m2)
			left_model, right_model = pair_key
		rows.append(
			{
				"pair_key": pair_key,
				"pair_label": pair_label,
				"model_left": left_model,
				"model_right": right_model,
				"morality_bias": row.get("morality_bias"),
				"consensus_sample_count": row.get("consensus_sample_count", 0),
				"source": source_name,
			}
		)

	if not rows:
		return pd.DataFrame(columns=["pair_label", "model_left", "model_right", "morality_bias", "consensus_sample_count", "source"])

	pairs = pd.DataFrame(rows)
	agg = (
		pairs.groupby(["pair_key", "pair_label", "model_left", "model_right", "source"], as_index=False)
		.apply(
			lambda g: pd.Series(
				{
					"consensus_sample_count": int(pd.to_numeric(g["consensus_sample_count"], errors="coerce").fillna(0).sum()),
					"morality_bias": weighted_average(g, "morality_bias", "consensus_sample_count"),
				}
			)
		)
		.reset_index(drop=True)
	)
	agg = agg.sort_values(["morality_bias", "pair_label"], ascending=[False, True]).reset_index(drop=True)
	return agg[["pair_label", "model_left", "model_right", "morality_bias", "consensus_sample_count", "source"]]


def merge_pair_tables(pair_tables: list[pd.DataFrame]) -> pd.DataFrame:
	non_empty = [frame for frame in pair_tables if not frame.empty]
	if not non_empty:
		return pd.DataFrame(columns=["pair_label", "model_left", "model_right", "morality_bias", "consensus_sample_count", "source"])

	combined = pd.concat(non_empty, ignore_index=True)
	merged = (
		combined.groupby(["pair_label", "model_left", "model_right"], as_index=False)
		.apply(
			lambda g: pd.Series(
				{
					"consensus_sample_count": int(pd.to_numeric(g["consensus_sample_count"], errors="coerce").fillna(0).sum()),
					"morality_bias": weighted_average(g, "morality_bias", "consensus_sample_count"),
					"source": "+".join(sorted(set(g["source"].astype(str))))
				}
			)
		)
		.reset_index(drop=True)
	)
	merged = merged.sort_values(["morality_bias", "pair_label"], ascending=[False, True]).reset_index(drop=True)
	return merged


def plot_leaderboard_bars(pair_table: pd.DataFrame, output_path: Path, title: str, color: str) -> None:
	sns.set_theme(style="whitegrid", font_scale=0.92)
	if pair_table.empty:
		return

	ordered = pair_table.sort_values("morality_bias", ascending=True).reset_index(drop=True)
	ordered["y_label"] = ordered["model_left"].astype(str) + "  |  " + ordered["model_right"].astype(str)

	fig_h = max(4, len(ordered) * 0.7 + 1.5)
	fig, ax = plt.subplots(figsize=(11, fig_h))
	ax.barh(ordered["y_label"], ordered["morality_bias"], color=color, alpha=0.9)

	for y_idx, value in enumerate(ordered["morality_bias"]):
		if pd.isna(value):
			continue
		x_offset = 1.0 if value >= 0 else -1.0
		ha = "left" if value >= 0 else "right"
		ax.text(value + x_offset, y_idx, f"{value:.1f}%", va="center", ha=ha, fontsize=9)

	ax.set_title(title)
	ax.set_xlabel("Morality bias (%)")
	ax.set_ylabel("Model A  |  Model B")
	ax.set_xlim(0, 100)
	ax.grid(axis="x", linestyle="--", alpha=0.35)
	plt.tight_layout()
	fig.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close(fig)


def collect_summary_for_root(root: Path) -> pd.DataFrame:
	folders = sorted(folder for folder in root.iterdir() if is_relevant_folder(folder))
	if not folders:
		return pd.DataFrame()

	groups: dict[str, dict[str, Path | None]] = {}
	for folder in folders:
		group = canonical_folder_name(folder)
		entry = groups.setdefault(group, {"base": None, "v2": None})
		if folder.name.endswith("_v2"):
			entry["v2"] = folder
		else:
			entry["base"] = folder

	rows = []
	for group_name in sorted(groups):
		entry = groups[group_name]
		base_folder = entry["base"]
		v2_folder = entry["v2"]
		if base_folder is None or v2_folder is None:
			continue
		rows.append(collect_group_metrics(base_folder, v2_folder))

	if not rows:
		return pd.DataFrame()

	summary = pd.DataFrame(rows)
	summary = summary.sort_values(["model_1_label", "model_2_label", "group"]).reset_index(drop=True)
	return summary


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Create leaderboard-style pairwise morality plots from Base/rerun/rerun2. "
			"Only model comparisons with both normal and v2 folders are included."
		),
	)
	parser.add_argument(
		"roots",
		type=Path,
		nargs="*",
		default=[Path("Base"), Path("rerun"), Path("rerun2")],
		help="Root folders containing comparison runs. First root is treated as Base.",
	)
	parser.add_argument(
		"--output-prefix",
		type=Path,
		default=Path("outputs/leaderboard"),
		help="Prefix for output files. Root name is appended.",
	)
	parser.add_argument(
		"--title",
		type=str,
		default="Morality Bias Leaderboard",
		help="Base title used for output plots.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	output_prefix = args.output_prefix if args.output_prefix.is_absolute() else Path.cwd() / args.output_prefix
	output_prefix.parent.mkdir(parents=True, exist_ok=True)

	any_output = False
	all_pair_tables: dict[str, pd.DataFrame] = {}

	for root_arg in args.roots:
		root = root_arg if root_arg.is_absolute() else Path.cwd() / root_arg
		if not root.exists():
			print(f"Skipping missing root: {root}")
			continue

		summary = collect_summary_for_root(root)
		if summary.empty:
			print(f"Skipping {root.name}: no groups with both normal and v2")
			continue

		include_self_pairs = root.name.lower() != "base"
		pair_table = aggregate_pair_morality(summary, include_self_pairs=include_self_pairs, source_name=root.name)
		if pair_table.empty:
			print(f"Skipping {root.name}: no eligible pairs after filtering")
			continue

		all_pair_tables[root.name] = pair_table

		root_prefix = output_prefix.with_name(f"{output_prefix.name}_{root.name}")
		summary_csv = root_prefix.with_name(f"{root_prefix.name}_summary.csv")
		pairs_csv = root_prefix.with_name(f"{root_prefix.name}_pairs.csv")
		bars_png = root_prefix.with_suffix(".png")

		summary_to_save = summary.copy()
		summary_to_save["persuasion_score"] = summary_to_save["persuasion_score"].round(4)
		summary_to_save["morality_bias"] = summary_to_save["morality_bias"].round(4)
		summary_to_save.to_csv(summary_csv, index=False, float_format="%.4f")

		pair_table_to_save = pair_table.copy()
		pair_table_to_save["morality_bias"] = pair_table_to_save["morality_bias"].round(4)
		pair_table_to_save.to_csv(pairs_csv, index=False, float_format="%.4f")

		any_output = True
		print(f"Saved summary table to {summary_csv}")
		print(f"Saved pair table to {pairs_csv}")

	base_table = all_pair_tables.get("Base")
	if base_table is not None and not base_table.empty:
		base_plot = output_prefix.with_name(f"{output_prefix.name}_Base_leaderboard.png")
		plot_leaderboard_bars(base_table, base_plot, f"{args.title}: Base", color="#2a9d8f")
		print(f"Saved Base leaderboard plot to {base_plot}")
		any_output = True

	rerun_merged = merge_pair_tables(
		[
			all_pair_tables.get("rerun", pd.DataFrame()),
			all_pair_tables.get("rerun2", pd.DataFrame()),
		]
	)
	if not rerun_merged.empty:
		rerun_pairs_csv = output_prefix.with_name(f"{output_prefix.name}_rerun_merged_pairs.csv")
		rerun_plot = output_prefix.with_name(f"{output_prefix.name}_rerun_merged_leaderboard.png")
		rerun_merged.to_csv(rerun_pairs_csv, index=False, float_format="%.4f")
		plot_leaderboard_bars(rerun_merged, rerun_plot, f"{args.title}: rerun + rerun2", color="#457b9d")
		print(f"Saved merged rerun pair table to {rerun_pairs_csv}")
		print(f"Saved merged rerun leaderboard plot to {rerun_plot}")
		any_output = True

	if not any_output:
		raise ValueError("No leaderboard outputs were generated for the provided roots.")


if __name__ == "__main__":
	main()
