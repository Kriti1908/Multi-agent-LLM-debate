from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, List, Optional, Tuple

try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ARCHIVE_ROOT = SCRIPT_DIR.parent / "Archive"

CATEGORY_INPUTS = {
    "Reasoning vs Non-reasoning": ARCHIVE_ROOT / "Reasoning vs Non-reasoning",
    "Runs/Untitled": ARCHIVE_ROOT / "Runs" / "Untitled",
    "Small vs Large": ARCHIVE_ROOT / "Small vs Large",
}

CATEGORY_OUTPUTS = {
    "Reasoning vs Non-reasoning": SCRIPT_DIR / "Reasoning vs Non-reasoning",
    "Runs/Untitled": SCRIPT_DIR / "Runs" / "Untitled",
    "Small vs Large": SCRIPT_DIR / "Small vs Large",
}

VERSION_ORDER = {"none": 0, "v2": 1, "v3": 2}


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

ACTION_RE = re.compile(r"\bAction\s*([12])\b", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def mean_or_none(values: List[float]) -> Optional[float]:
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return None
    return float(fmean(values))


def extract_action_label(text: Any) -> Optional[str]:
    if text is None:
        return None
    m = ACTION_RE.search(str(text))
    if not m:
        return None
    return f"Action {m.group(1)}"


def normalize_text_key(text: Any) -> str:
    if text is None:
        return ""
    s = str(text).strip().lower()
    s = SPACE_RE.sub(" ", s)
    return s


def response_to_score(response: Any) -> float:
    """
    Map:
      agreed -> +1
      partially agreed -> +0.5
      ignored -> 0
      disagreed -> -1
    """
    s = normalize_text_key(response)

    if s == "agreed":
        return 1.0
    if s in {"partially agreed", "partially_agreed", "partial agreed"}:
        return 0.5
    if s == "ignored":
        return 0.0
    if s == "disagreed":
        return -1.0

    if "partially" in s and "agree" in s:
        return 0.5
    if "agree" in s:
        return 1.0
    if "ignore" in s:
        return 0.0
    if "disagree" in s:
        return -1.0
    return 0.0


def strength_to_score(kind: Any) -> float:
    """
    Map:
      strong -> +1
      weak -> 0
    """
    s = normalize_text_key(kind)
    if s == "strong":
        return 1.0
    if s == "weak":
        return 0.0
    if "strong" in s:
        return 1.0
    return 0.0


def count_tokens(text: Any, model_name: Optional[str] = None) -> int:
    """
    Try to count tokens with tiktoken if available.
    Fall back to a simple whitespace-based approximation.
    """
    if text is None:
        return 0
    s = str(text)

    if tiktoken is not None:
        try:
            if model_name:
                try:
                    enc = tiktoken.encoding_for_model(model_name)
                except Exception:
                    enc = tiktoken.get_encoding("cl100k_base")
            else:
                enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(s))
        except Exception:
            pass

    return len([tok for tok in SPACE_RE.split(s.strip()) if tok])


def parse_version_from_folder(folder_name: str) -> str:
    m = re.search(r"_(v\d+)$", folder_name)
    if m:
        return m.group(1)
    return "none"


def is_run_folder(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "arguments").is_dir():
        return False

    for p in path.iterdir():
        if p.is_file() and p.suffix == ".json" and not p.name.endswith(".arguments.json"):
            if re.fullmatch(r"\d+\.json", p.name):
                return True
    return False


def list_debate_files(run_dir: Path) -> List[Path]:
    files = [
        p for p in run_dir.iterdir()
        if p.is_file()
        and p.suffix == ".json"
        and not p.name.endswith(".arguments.json")
        and re.fullmatch(r"\d+\.json", p.name)
    ]
    files.sort(key=lambda p: int(p.stem))
    return files


def list_argument_files(arguments_dir: Path) -> Dict[str, Path]:
    """
    Scan the arguments directory directly and map debate stem -> matching file.
    Example:
      2000.arguments.json -> key "2000"
    """
    mapping: Dict[str, Path] = {}
    if not arguments_dir.is_dir():
        return mapping

    for p in arguments_dir.iterdir():
        if (
            p.is_file()
            and p.suffix == ".json"
            and p.name.endswith(".arguments.json")
        ):
            stem = p.name[: -len(".arguments.json")]
            mapping[stem] = p

    return mapping


def build_agent_mapping(arguments: Any) -> Dict[str, str]:
    if not isinstance(arguments, list):
        return {}

    unique_agents: List[str] = []
    for arg in arguments:
        if not isinstance(arg, dict):
            continue
        agent = str(arg.get("agent", "")).strip()
        if agent and agent not in unique_agents:
            unique_agents.append(agent)

    mapping: Dict[str, str] = {}

    for agent in unique_agents:
        low = agent.lower()
        if "agent1" in low or "model1" in low or re.search(r"\b1\b", low):
            mapping[agent] = "model1"
        elif "agent2" in low or "model2" in low or re.search(r"\b2\b", low):
            mapping[agent] = "model2"

    unresolved = [a for a in unique_agents if a not in mapping]
    if unresolved and len(unique_agents) == 2:
        if unique_agents[0] not in mapping:
            mapping[unique_agents[0]] = "model1"
        if unique_agents[1] not in mapping:
            mapping[unique_agents[1]] = "model2"

    return mapping


def winner_from_verdict(final_verdict: Optional[str]) -> Optional[str]:
    action = extract_action_label(final_verdict)
    if action == "Action 1":
        return "model1"
    if action == "Action 2":
        return "model2"
    return None



def build_message_agent_mapping(messages: Any) -> Dict[str, str]:
    if not isinstance(messages, list):
        return {}

    unique_agents: List[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip().lower()
        name = str(msg.get("name", "")).strip()
        if role == "user" and name in ("", "UserProxy"):
            continue
        if not name:
            continue
        if name not in unique_agents:
            unique_agents.append(name)

    mapping: Dict[str, str] = {}
    for agent in unique_agents:
        low = agent.lower()
        if "agent1" in low or "model1" in low or re.search(r"\b1\b", low):
            mapping[agent] = "model1"
        elif "agent2" in low or "model2" in low or re.search(r"\b2\b", low):
            mapping[agent] = "model2"

    unresolved = [a for a in unique_agents if a not in mapping]
    if unresolved and len(unique_agents) == 2:
        if unique_agents[0] not in mapping:
            mapping[unique_agents[0]] = "model1"
        if unique_agents[1] not in mapping:
            mapping[unique_agents[1]] = "model2"

    return mapping


def get_message_round_number(msg: Dict[str, Any], fallback_round: int) -> int:
    round_value = msg.get("round")
    try:
        if round_value is None:
            return fallback_round
        return int(round_value)
    except Exception:
        return fallback_round


def extract_text_from_message(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(parts)
    return str(content)


def compute_debate_behavior_metrics(
    debate: Dict[str, Any],
    model1_name: str,
    model2_name: str,
) -> Dict[str, Any]:
    """
    New debate-based metrics computed directly from debate messages.

    These are computed from model1's perspective:
      - NC: no change vs previous valid round
      - AR: alignment with model2 in the same round
      - FR: invalid / missing action label rate
      - ArgsPerSample_*: counts of valid rounds by stance/alignment
      - Agr/DisAgr/Ign: frequencies over model1 rounds
      - Verbosity_*: mean token length of model1 messages split by stance
    """
    messages = debate.get("messages", []) or []
    agent_mapping = build_message_agent_mapping(messages)

    per_model: Dict[str, List[Tuple[int, str, Optional[str], int]]] = {
        "model1": [],
        "model2": [],
    }
    fallback_rounds = {"model1": 0, "model2": 0}

    for msg in messages:
        role = str(msg.get("role", "")).strip().lower()
        name = str(msg.get("name", "")).strip()
        if role == "user" and name in ("", "UserProxy"):
            continue
        if not name:
            continue

        model_key = agent_mapping.get(name)
        if model_key not in {"model1", "model2"}:
            continue

        fallback_rounds[model_key] += 1
        round_no = get_message_round_number(msg, fallback_rounds[model_key])

        content = extract_text_from_message(msg.get("content"))
        action = extract_action_label(content)
        model_name_for_tokens = model1_name if model_key == "model1" else model2_name
        tokens = count_tokens(content, model_name_for_tokens)

        per_model[model_key].append((round_no, content, action, tokens))

    model1_msgs = sorted(per_model["model1"], key=lambda x: (x[0], x[1]))
    model2_by_round = {r: action for r, _, action, _ in per_model["model2"]}

    # Base stance for model1 is action1 from the debate file.
    base_action = extract_action_label(debate.get("action1"))
    if base_action is None:
        for _, _, action, _ in model1_msgs:
            if action is not None:
                base_action = action
                break

    total_rounds = len(model1_msgs)
    valid_rounds = [(r, c, a, t) for (r, c, a, t) in model1_msgs if a is not None]
    valid_count = len(valid_rounds)
    invalid_count = total_rounds - valid_count

    # NC: no change from one valid round to the next.
    same_transition_count = 0
    transition_count = 0
    prev_action: Optional[str] = None
    for _, _, action, _ in model1_msgs:
        if action is None:
            continue
        if prev_action is not None:
            transition_count += 1
            if action == prev_action:
                same_transition_count += 1
        prev_action = action

    nc_rate = (same_transition_count / transition_count) if transition_count > 0 else None

    # Debate alignment from model1's perspective.
    aligned_count = 0
    opposed_count = 0
    ignored_count = invalid_count
    aligned_tokens: List[float] = []
    opposed_tokens: List[float] = []

    fav_si_count = 0
    opp_si_count = 0
    fav_si_tokens: List[float] = []
    opp_si_tokens: List[float] = []

    for round_no, _, action, tokens in model1_msgs:
        if action is None:
            ignored_count += 1
            continue

        if base_action is not None and action == base_action:
            fav_si_count += 1
            fav_si_tokens.append(float(tokens))
        else:
            opp_si_count += 1
            opp_si_tokens.append(float(tokens))

        other_action = model2_by_round.get(round_no)
        if other_action is None:
            ignored_count += 1
            continue

        if action == other_action:
            aligned_count += 1
            aligned_tokens.append(float(tokens))
        else:
            opposed_count += 1
            opposed_tokens.append(float(tokens))

    comparisons = aligned_count + opposed_count
    ar_rate = (aligned_count / comparisons) if comparisons > 0 else None

    fr_rate = (invalid_count / total_rounds) if total_rounds > 0 else None

    fav_dbt_count = float(aligned_count)
    opp_dbt_count = float(opposed_count)

    # Response-style frequencies based on same/aligned vs different/opposed vs invalid.
    agr_rate = (aligned_count / total_rounds) if total_rounds > 0 else None
    disagr_rate = (opposed_count / total_rounds) if total_rounds > 0 else None
    ign_rate = (ignored_count / total_rounds) if total_rounds > 0 else None

    verbosity_fav = mean_or_none(fav_si_tokens)
    verbosity_opp = mean_or_none(opp_si_tokens)

    return {
        "NC_Rate": nc_rate,
        "AR_Rate": ar_rate,
        "FR_Rate": fr_rate,

        "ArgsPerSample_Fav_SI": float(fav_si_count),
        "ArgsPerSample_Opp_SI": float(opp_si_count),
        "ArgsPerSample_Fav_Dbt": fav_dbt_count,
        "ArgsPerSample_Opp_Dbt": opp_dbt_count,

        "Agr_Rate": agr_rate,
        "DisAgr_Rate": disagr_rate,
        "Ign_Rate": ign_rate,

        "Verbosity_Fav": verbosity_fav,
        "Verbosity_Opp": verbosity_opp,
    }


def detect_consensus(
    messages: List[Dict[str, Any]],
    final_verdict: Optional[str],
) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Consensus detection with three tiers:

    1. TERMINATE found (outside <think> blocks) → (True, round_no, action)
       Action from the TERMINATE message itself (after keyword, then anywhere in msg).

    2. No TERMINATE, total rounds < 10 → (True, total_rounds, action)
       Action from final_verdict.

    3. No TERMINATE, total rounds == 10 → check for implicit consensus:
       Find the first round where BOTH agents output a pure action label
       (entire content outside <think> is just "Action 1" or "Action 2")
       and they agree. If found → (True, round_no, action).
       If not found → (False, None, None).
    """
    TERMINATE_RE = re.compile(r"\bTERMINATE\b", re.IGNORECASE)
    THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
    # strip markdown bold/italic markers and unicode non-breaking spaces
    CLEAN_RE = re.compile(r"[\*\_]+")

    def extract_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            return "\n".join(parts)
        return str(content)

    def strip_think(text: str) -> str:
        return THINK_RE.sub("", text)

    def clean_text(text: str) -> str:
        """Strip think blocks, markdown, unicode spaces, and whitespace."""
        text = strip_think(text)
        text = CLEAN_RE.sub("", text)
        # normalize unicode non-breaking / narrow spaces to regular space
        text = re.sub(r"[\u00a0\u202f\u2009\u2008\u2007\u2006\u2005\u2004\u2003\u2002\u2001\u200b]+", " ", text)
        return text.strip()

    def is_pure_action(text: str) -> Optional[str]:
        """
        Return action label if the entire cleaned text is just an action label,
        else None.
        """
        c = clean_text(text)
        m = ACTION_RE.fullmatch(c)
        if m:
            return f"Action {m.group(1)}"
        # allow minor surrounding noise (e.g. "We must argue... Action 1" is NOT pure)
        # but "**Action 1**" after stripping is pure
        m = ACTION_RE.search(c)
        if m and len(c.replace(m.group(0), "").strip()) == 0:
            return f"Action {m.group(1)}"
        return None

    def get_round_number(terminate_idx: int) -> int:
        first_agent_name = None
        round_no = 0
        for i in range(terminate_idx + 1):
            msg = messages[i]
            role = str(msg.get("role", "")).strip().lower()
            name = str(msg.get("name", "")).strip()
            if role == "user" and name in ("", "UserProxy"):
                continue
            if first_agent_name is None and name:
                first_agent_name = name
            if name == first_agent_name:
                round_no += 1
        return max(round_no, 1)

    def count_total_rounds() -> int:
        first_agent_name = None
        round_no = 0
        for msg in messages:
            role = str(msg.get("role", "")).strip().lower()
            name = str(msg.get("name", "")).strip()
            if role == "user" and name in ("", "UserProxy"):
                continue
            if first_agent_name is None and name:
                first_agent_name = name
            if name == first_agent_name:
                round_no += 1
        return max(round_no, 1)

    def find_implicit_consensus_round() -> Tuple[Optional[int], Optional[str]]:
        """
        Walk through agent-pair exchanges. For each round, check if both
        agents output a pure action label and they agree.
        Returns (round_no, action) of the first such round, or (None, None).
        """
        first_agent_name = None
        second_agent_name = None
        round_no = 0
        round_msgs: Dict[int, Dict[str, str]] = {}  # round_no -> {agent_name: raw_text}

        for msg in messages:
            role = str(msg.get("role", "")).strip().lower()
            name = str(msg.get("name", "")).strip()
            if role == "user" and name in ("", "UserProxy"):
                continue
            if not name:
                continue

            if first_agent_name is None:
                first_agent_name = name

            if name == first_agent_name:
                round_no += 1
                round_msgs[round_no] = {}

            if second_agent_name is None and name != first_agent_name:
                second_agent_name = name

            raw = extract_text(msg.get("content"))
            round_msgs.setdefault(round_no, {})[name] = raw

        # Now find first round where both agents agree on a pure action
        for rno in sorted(round_msgs.keys()):
            pair = round_msgs[rno]
            if first_agent_name not in pair or second_agent_name not in pair:
                continue
            a1 = is_pure_action(pair[first_agent_name])
            a2 = is_pure_action(pair[second_agent_name])
            if a1 is not None and a1 == a2:
                return rno, a1

        return None, None

    # ----------------------------------------------------------------
    # Tier 1: look for real TERMINATE (outside <think>)
    # ----------------------------------------------------------------
    first_terminate_idx: Optional[int] = None
    for idx, msg in enumerate(messages):
        full_text = extract_text(msg.get("content"))
        if TERMINATE_RE.search(strip_think(full_text)):
            first_terminate_idx = idx
            break

    if first_terminate_idx is not None:
        round_number = get_round_number(first_terminate_idx)
        full_text = extract_text(messages[first_terminate_idx].get("content"))
        c = strip_think(full_text)
        after_terminate = TERMINATE_RE.split(c, maxsplit=1)[-1]
        action = extract_action_label(after_terminate)
        if action is None:
            action = extract_action_label(c)
        return True, round_number, action

    # ----------------------------------------------------------------
    # Tier 2: no TERMINATE, rounds < 10 → implicit consensus
    # ----------------------------------------------------------------
    total_rounds = count_total_rounds()
    if total_rounds < 10:
        action = extract_action_label(final_verdict)
        return True, total_rounds, action

    # ----------------------------------------------------------------
    # Tier 3: no TERMINATE, rounds == 10 → check pure-action agreement
    # ----------------------------------------------------------------
    round_no, action = find_implicit_consensus_round()
    if round_no is not None:
        return True, round_no, action

    return False, None, None

# ---------------------------------------------------------------------
# Per-debate processing
# ---------------------------------------------------------------------


def process_debate(
    debate: Dict[str, Any],
    arguments_data: Dict[str, Any],
    version: str,
) -> Dict[str, Any]:
    model1_name = str(debate.get("model_1", ""))
    model2_name = str(debate.get("model_2", ""))

    # If reasoning effort fields exist (Reasoning vs Non-reasoning runs),
    # append them to model names.
    re1 = str(debate.get("reasoning_effort_1", "")).strip()
    re2 = str(debate.get("reasoning_effort_2", "")).strip()
    if re1:
        model1_name = f"{model1_name}_{re1}"
    if re2:
        model2_name = f"{model2_name}_{re2}"

    # Changed here: value1/value2 come from values1/values2, not reasoning_effort_1/2
    value1 = str(debate.get("values1", ""))
    value2 = str(debate.get("values2", ""))

    final_verdict = arguments_data.get("final_verdict")
    if final_verdict is None:
        final_verdict = debate.get("final_verdict")

    messages = debate.get("messages", []) or []
    consensus_reached, turns_to_consensus, consensus_action = detect_consensus(messages, final_verdict)
    winner = winner_from_verdict(final_verdict)

    arguments = arguments_data.get("arguments", []) or []
    agent_mapping = build_agent_mapping(arguments)

    token_counts = {"model1": [], "model2": []}
    arg_result_sum = {"model1": 0.0, "model2": 0.0}
    arg_strength_sum = {"model1": 0.0, "model2": 0.0}

    if isinstance(arguments, list):
        for arg in arguments:
            if not isinstance(arg, dict):
                continue

            agent = str(arg.get("agent", "")).strip()
            model_key = agent_mapping.get(agent)

            if model_key is None:
                low = agent.lower()
                if "agent1" in low or "model1" in low:
                    model_key = "model1"
                elif "agent2" in low or "model2" in low:
                    model_key = "model2"

            if model_key not in {"model1", "model2"}:
                continue

            arg_text = arg.get("argument", "")
            model_name_for_tokens = model1_name if model_key == "model1" else model2_name
            token_counts[model_key].append(count_tokens(arg_text, model_name_for_tokens))
            arg_result_sum[model_key] += response_to_score(arg.get("response"))
            arg_strength_sum[model_key] += strength_to_score(arg.get("type"))

    token_cost_model1 = mean_or_none([float(x) for x in token_counts["model1"]])
    token_cost_model2 = mean_or_none([float(x) for x in token_counts["model2"]])

    drift_model1 = None
    drift_model2 = None
    if version == "v2":
        if winner == "model1":
            drift_model1 = 1.0
            drift_model2 = 0.0
        elif winner == "model2":
            drift_model1 = 0.0
            drift_model2 = 1.0
        else:
            drift_model1 = 0.0
            drift_model2 = 0.0

    # New metrics computed from debate messages, model1 perspective.
    debate_metrics = compute_debate_behavior_metrics(debate, model1_name, model2_name)

    return {
        "Model1": model1_name,
        "Value1": value1,
        "Model2": model2_name,
        "Value2": value2,
        "Version": version,
        "Drift_Model1": drift_model1,
        "Drift_Model2": drift_model2,
        "Persuasion_Drift": (
            (drift_model1 - drift_model2) if drift_model1 is not None and drift_model2 is not None else None
        ),
        "TokenCost_Model1": token_cost_model1,
        "TokenCost_Model2": token_cost_model2,
        "Consensus_Reached": 1.0 if consensus_reached else 0.0,
        "Turns_To_Consensus": float(turns_to_consensus) if turns_to_consensus is not None else None,
        "ArgumentResult_Model1": arg_result_sum["model1"],
        "ArgumentResult_Model2": arg_result_sum["model2"],
        "ArgumentStrength_Model1": arg_strength_sum["model1"],
        "ArgumentStrength_Model2": arg_strength_sum["model2"],
        "Consensus_Winner": winner,
        **debate_metrics,
        "Steps": float(turns_to_consensus) if turns_to_consensus is not None else None,
    }


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------


def aggregate_category(category_name: str, input_root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not input_root.exists():
        print(f"[WARN] Missing input folder: {input_root}", file=sys.stderr)
        return [], []

    run_dirs = sorted(
        [p for p in input_root.rglob("*") if is_run_folder(p)],
        key=lambda p: str(p).lower(),
    )

    grouped: Dict[Tuple[str, str, str, str, str], Dict[str, Any]] = {}
    missing_argument_files: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        version = parse_version_from_folder(run_dir.name)

        debate_files = list_debate_files(run_dir)
        if not debate_files:
            continue

        arguments_dir = run_dir / "arguments"
        argument_files = list_argument_files(arguments_dir)

        for debate_path in debate_files:
            try:
                debate = load_json(debate_path)
            except Exception as e:
                print(f"[WARN] Failed to load {debate_path}: {e}", file=sys.stderr)
                continue

            args_path = argument_files.get(debate_path.stem)
            if args_path is None or not args_path.exists():
                expected = arguments_dir / f"{debate_path.stem}.arguments.json"
                print(f"[WARN] Missing arguments file: {expected}", file=sys.stderr)
                missing_argument_files.append({
                    "debate_file": debate_path.name,
                    "debate_path": str(debate_path),
                    "expected_arguments_file": expected.name,
                    "expected_arguments_path": str(expected),
                    "run_folder": str(run_dir),
                })
                continue

            try:
                arguments_data = load_json(args_path)
            except Exception as e:
                print(f"[WARN] Failed to load {args_path}: {e}", file=sys.stderr)
                continue

            row = process_debate(debate, arguments_data, version)

            key = (
                row["Model1"],
                row["Value1"],
                row["Model2"],
                row["Value2"],
                row["Version"],
            )

            if key not in grouped:
                grouped[key] = {
                    "Model1": row["Model1"],
                    "Value1": row["Value1"],
                    "Model2": row["Model2"],
                    "Value2": row["Value2"],
                    "Version": row["Version"],
                    "Debates": 0,
                    "Drift_Model1": [],
                    "Drift_Model2": [],
                    "Persuasion_Drift": [],
                    "TokenCost_Model1": [],
                    "TokenCost_Model2": [],
                    "Consensus_Reached": [],
                    "Turns_To_Consensus": [],
                    "ArgumentResult_Model1": [],
                    "ArgumentResult_Model2": [],
                    "ArgumentStrength_Model1": [],
                    "ArgumentStrength_Model2": [],
                    "Consensus_Winner_Model1": 0,
                    "Consensus_Winner_Model2": 0,
                    "Consensus_Winner_Unknown": 0,
                    "NC_Rate": [],
                    "AR_Rate": [],
                    "FR_Rate": [],
                    "ArgsPerSample_Fav_SI": [],
                    "ArgsPerSample_Opp_SI": [],
                    "ArgsPerSample_Fav_Dbt": [],
                    "ArgsPerSample_Opp_Dbt": [],
                    "Agr_Rate": [],
                    "DisAgr_Rate": [],
                    "Ign_Rate": [],
                    "Verbosity_Fav": [],
                    "Verbosity_Opp": [],
                    "Steps": [],
                }

            bucket = grouped[key]
            bucket["Debates"] += 1

            bucket["Drift_Model1"].append(row["Drift_Model1"])
            bucket["Drift_Model2"].append(row["Drift_Model2"])
            bucket["Persuasion_Drift"].append(row["Persuasion_Drift"])

            bucket["TokenCost_Model1"].append(row["TokenCost_Model1"])
            bucket["TokenCost_Model2"].append(row["TokenCost_Model2"])

            bucket["Consensus_Reached"].append(row["Consensus_Reached"])
            bucket["Turns_To_Consensus"].append(row["Turns_To_Consensus"])

            bucket["ArgumentResult_Model1"].append(row["ArgumentResult_Model1"])
            bucket["ArgumentResult_Model2"].append(row["ArgumentResult_Model2"])

            bucket["ArgumentStrength_Model1"].append(row["ArgumentStrength_Model1"])
            bucket["ArgumentStrength_Model2"].append(row["ArgumentStrength_Model2"])

            bucket["NC_Rate"].append(row["NC_Rate"])
            bucket["AR_Rate"].append(row["AR_Rate"])
            bucket["FR_Rate"].append(row["FR_Rate"])

            bucket["ArgsPerSample_Fav_SI"].append(row["ArgsPerSample_Fav_SI"])
            bucket["ArgsPerSample_Opp_SI"].append(row["ArgsPerSample_Opp_SI"])
            bucket["ArgsPerSample_Fav_Dbt"].append(row["ArgsPerSample_Fav_Dbt"])
            bucket["ArgsPerSample_Opp_Dbt"].append(row["ArgsPerSample_Opp_Dbt"])

            bucket["Agr_Rate"].append(row["Agr_Rate"])
            bucket["DisAgr_Rate"].append(row["DisAgr_Rate"])
            bucket["Ign_Rate"].append(row["Ign_Rate"])

            bucket["Verbosity_Fav"].append(row["Verbosity_Fav"])
            bucket["Verbosity_Opp"].append(row["Verbosity_Opp"])

            bucket["Steps"].append(row["Steps"])

            # Only count consensus winners when a consensus was actually reached.
            # This keeps Consensus_Wins_* <= Consensus_Debates and makes win-share
            # metrics mathematically valid.
            if row["Consensus_Reached"] == 1.0:
                winner = row["Consensus_Winner"]
                if winner == "model1":
                    bucket["Consensus_Winner_Model1"] += 1
                elif winner == "model2":
                    bucket["Consensus_Winner_Model2"] += 1
                else:
                    bucket["Consensus_Winner_Unknown"] += 1

    records: List[Dict[str, Any]] = []
    for bucket in grouped.values():
        consensus_count = int(sum(1 for x in bucket["Consensus_Reached"] if x == 1.0))

        if bucket["Version"] == "v2":
            drift_model1_count = int(sum(1 for x in bucket["Drift_Model1"] if x == 1.0))
            drift_model2_count = int(sum(1 for x in bucket["Drift_Model2"] if x == 1.0))
            drift_model1_rate = drift_model1_count / bucket["Debates"] if bucket["Debates"] else None
            drift_model2_rate = drift_model2_count / bucket["Debates"] if bucket["Debates"] else None
            persuasion_drift = (
                (drift_model1_rate - drift_model2_rate)
                if drift_model1_rate is not None and drift_model2_rate is not None
                else None
            )
        else:
            drift_model1_count = None
            drift_model2_count = None
            drift_model1_rate = None
            drift_model2_rate = None
            persuasion_drift = None

        consensus_wins_model1 = bucket["Consensus_Winner_Model1"]
        consensus_wins_model2 = bucket["Consensus_Winner_Model2"]
        consensus_wins_unknown = bucket["Consensus_Winner_Unknown"]
        consensus_no_winner = max(0, consensus_count - consensus_wins_model1 - consensus_wins_model2)

        record = {
            "Model1": bucket["Model1"],
            "Value1": bucket["Value1"],
            "Model2": bucket["Model2"],
            "Value2": bucket["Value2"],
            "Version": bucket["Version"],
            "Debates": bucket["Debates"],
            "Drift_Model1_Count": drift_model1_count,
            "Drift_Model2_Count": drift_model2_count,
            "Drift_Model1_Rate": drift_model1_rate,
            "Drift_Model2_Rate": drift_model2_rate,
            "Persuasion_Drift": persuasion_drift,
            "TokenCost_Model1": mean_or_none(bucket["TokenCost_Model1"]),
            "TokenCost_Model2": mean_or_none(bucket["TokenCost_Model2"]),
            "Consensus_Rate": mean_or_none(bucket["Consensus_Reached"]),
            "Avg_Turns_To_Consensus": mean_or_none(
                [float(x) for x in bucket["Turns_To_Consensus"] if x is not None]
            ),
            "ArgumentResult_Model1": mean_or_none(bucket["ArgumentResult_Model1"]),
            "ArgumentResult_Model2": mean_or_none(bucket["ArgumentResult_Model2"]),
            "ArgumentStrength_Model1": mean_or_none(bucket["ArgumentStrength_Model1"]),
            "ArgumentStrength_Model2": mean_or_none(bucket["ArgumentStrength_Model2"]),
            "Consensus_Wins_Model1": consensus_wins_model1,
            "Consensus_Wins_Model2": consensus_wins_model2,
            "Consensus_Wins_Unknown": consensus_wins_unknown,
            "Consensus_NoWinner": consensus_no_winner,
            "Consensus_WinRate_Model1": (
                consensus_wins_model1 / consensus_count if consensus_count > 0 else None
            ),
            "Consensus_WinRate_Model2": (
                consensus_wins_model2 / consensus_count if consensus_count > 0 else None
            ),
            "Consensus_NoWinner_Rate": (
                consensus_no_winner / consensus_count if consensus_count > 0 else None
            ),
            "Consensus_Debates": consensus_count,

            "NC_Rate": mean_or_none(bucket["NC_Rate"]),
            "AR_Rate": mean_or_none(bucket["AR_Rate"]),
            "FR_Rate": mean_or_none(bucket["FR_Rate"]),

            "ArgsPerSample_Fav_SI": mean_or_none(bucket["ArgsPerSample_Fav_SI"]),
            "ArgsPerSample_Opp_SI": mean_or_none(bucket["ArgsPerSample_Opp_SI"]),
            "ArgsPerSample_Fav_Dbt": mean_or_none(bucket["ArgsPerSample_Fav_Dbt"]),
            "ArgsPerSample_Opp_Dbt": mean_or_none(bucket["ArgsPerSample_Opp_Dbt"]),

            "Agr_Rate": mean_or_none(bucket["Agr_Rate"]),
            "DisAgr_Rate": mean_or_none(bucket["DisAgr_Rate"]),
            "Ign_Rate": mean_or_none(bucket["Ign_Rate"]),

            "Verbosity_Fav": mean_or_none(bucket["Verbosity_Fav"]),
            "Verbosity_Opp": mean_or_none(bucket["Verbosity_Opp"]),

            "Avg_Steps": mean_or_none(bucket["Steps"]),
        }

        records.append(record)

    records.sort(key=lambda r: (
        r["Model1"],
        r["Value1"],
        r["Model2"],
        r["Value2"],
        VERSION_ORDER.get(r["Version"], 99),
    ))

    return records, missing_argument_files


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

def main() -> None:
    if not ARCHIVE_ROOT.exists():
        raise FileNotFoundError(f"Archive folder not found at: {ARCHIVE_ROOT}")

    for category_name, input_root in CATEGORY_INPUTS.items():
        output_root = CATEGORY_OUTPUTS[category_name]
        output_root.mkdir(parents=True, exist_ok=True)

        records, missing_argument_files = aggregate_category(category_name, input_root)

        output_path = output_root / "metrics_new.json"
        dump_json(output_path, records)

        missing_output_path = output_root / "missing_arguments_files_new.json"
        dump_json(missing_output_path, missing_argument_files)

        print(f"[OK] Wrote {len(records)} records to: {output_path}")
        print(f"[OK] Wrote {len(missing_argument_files)} missing-file records to: {missing_output_path}")


if __name__ == "__main__":
    main()