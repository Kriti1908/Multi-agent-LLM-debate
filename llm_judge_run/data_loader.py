"""
data_loader.py
--------------
Loads all debate JSON and arguments JSON files from the nested folder structure.

Folder structure expected:
  <base_dir>/
    reasoning_<model_a>_<model_b>/
      <id>.json                          (debate transcript)
      arguments/
        <id>.arguments.json              (argument analysis)
    ...

Returns a list of DebateRecord dataclasses ready for metric computation.
"""

import os
import re
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Message:
    role: str
    name: str
    content_raw: str          # original (may contain <think>...</think>)
    content_clean: str        # <think> blocks stripped
    word_count: int           # word count of clean content
    char_count: int           # char count of clean content


@dataclass
class Argument:
    argument: str
    agent: str
    arg_type: str             # "Strong" | "Weak"
    response: str             # "Agreed" | "Disagreed" | "Ignored" | "Partial"
    justification_type: str
    justification_response: str


@dataclass
class DebateRecord:
    # ── Identifiers ──────────────────────────────────────────────────────────
    debate_id: str
    folder: str               # e.g. "reasoning_qwen3-8b_llama3.1-8b"
    file_path: str

    # ── Participants ─────────────────────────────────────────────────────────
    model_1: str              # full model string from JSON
    model_2: str
    values1: str              # value defended for action1
    values2: str
    action1: str
    action2: str
    dilemma: str
    context: str

    # ── Agent mapping ────────────────────────────────────────────────────────
    # agent_map[agent_name] = {model, value, action_label}
    # Derived from first-appearance ordering in messages.
    agent_map: dict = field(default_factory=dict)

    # ── Messages ─────────────────────────────────────────────────────────────
    messages: list = field(default_factory=list)      # list[Message]

    # ── Arguments (from arguments file) ──────────────────────────────────────
    arguments: list = field(default_factory=list)     # list[Argument]
    final_verdict: Optional[str] = None               # "Action 1" | "Action 2" | "No Consensus"

    # ── Derived per-agent stats ───────────────────────────────────────────────
    # Filled by compute_basic_stats()
    per_agent_stats: dict = field(default_factory=dict)

    # ── Folder-level metadata ─────────────────────────────────────────────────
    temperature_tag: Optional[str] = None             # "low" | "high" | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)


def strip_think(text: str) -> str:
    return THINK_RE.sub('', text).strip()


def _extract_agent_name_tag(agent_name: str) -> str:
    """'Agent_Qwen' -> 'qwen',  'Agent_Llama' -> 'llama', etc."""
    return agent_name.replace('Agent_', '').lower()


def _model_matches_tag(model_str: str, tag: str) -> bool:
    """Check if a loose tag like 'qwen' appears in the model string."""
    model_lower = model_str.lower()
    # handle common aliases
    aliases = {
        'qwen': ['qwen'],
        'llama': ['llama'],
        'gemma': ['gemma'],
        'gpt': ['gpt', 'openai'],
        'gptoss': ['gpt', 'openai'],
    }
    candidates = aliases.get(tag, [tag])
    return any(c in model_lower for c in candidates)


def _build_agent_map(debate: dict) -> dict:
    """
    Attach reasoning mode per side using reasoning_effort_1/2 so LOW/HIGH
    distinctions are preserved even when model_1 == model_2.
    """
    seen_agents = []
    for m in debate.get('messages', []):
        if m.get('name') != 'UserProxy' and m.get('name') not in seen_agents:
            seen_agents.append(m.get('name'))
        if len(seen_agents) == 2:
            break

    if len(seen_agents) < 2:
        logger.warning("Debate %s has fewer than 2 agents", debate.get('debate_id', '?'))
        return {}

    agent1, agent2 = seen_agents[0], seen_agents[1]

    def resolve_model(agent_name, m1, m2):
        tag = _extract_agent_name_tag(agent_name)
        if _model_matches_tag(m1, tag):
            return m1
        if _model_matches_tag(m2, tag):
            return m2
        return None

    m1, m2 = debate.get('model_1', ''), debate.get('model_2', '')

    # Authoritative per-side reasoning effort from JSON
    mode1 = debate.get('reasoning_effort_1')
    mode2 = debate.get('reasoning_effort_2')

    mode1 = mode1.lower().strip() if isinstance(mode1, str) else None
    mode2 = mode2.lower().strip() if isinstance(mode2, str) else None
    mode1 = mode1 if mode1 in ('low', 'high') else None
    mode2 = mode2 if mode2 in ('low', 'high') else None

    # Store canonical model keys WITH mode (this is what downstream metrics should use)
    model1_base = normalize_model_name(resolve_model(agent1, m1, m2) or m1)
    model2_base = normalize_model_name(resolve_model(agent2, m1, m2) or m2)

    # normalize_model_name() might already have :low/:high (if provided in string); don't double-append
    def ensure_mode(model_name: str, m: Optional[str]) -> str:
        if m in ('low', 'high') and (not model_name.endswith(':low')) and (not model_name.endswith(':high')):
            return f"{model_name}:{m}"
        return model_name

    return {
        agent1: {
            'model': ensure_mode(model1_base, mode1),
            'model_base': model1_base.split(':', 1)[0],
            'mode': mode1,
            'value': debate.get('values1', 'unknown'),
            'action_label': debate.get('action1', 'unknown'),
            'action_number': 'Action 1',
        },
        agent2: {
            'model': ensure_mode(model2_base, mode2),
            'model_base': model2_base.split(':', 1)[0],
            'mode': mode2,
            'value': debate.get('values2', 'unknown'),
            'action_label': debate.get('action2', 'unknown'),
            'action_number': 'Action 2',
        },
    }
# ...existing code...


def _parse_messages(raw_messages: list) -> list:
    msgs = []
    for m in raw_messages:
        raw = m.get('content', '')
        
        # 1. Safely handle cases where 'content' is a list of blocks
        if isinstance(raw, list):
            extracted_text = []
            for item in raw:
                if isinstance(item, str):
                    extracted_text.append(item)
                # Safely get 'text' and ensure it's not None before appending
                elif isinstance(item, dict) and item.get('text') is not None:
                    extracted_text.append(str(item['text']))
            raw = '\n'.join(extracted_text)
            
        # 2. Fallback to string casting just in case it's a null type or unexpected object
        elif not isinstance(raw, str):
            raw = str(raw) if raw is not None else ''

        clean = strip_think(raw)
        
        msgs.append(Message(
            role=m.get('role', ''),
            name=m.get('name', ''),
            content_raw=raw,          # Now guaranteed to be a string
            content_clean=clean,
            word_count=len(clean.split()),
            char_count=len(clean),
        ))
    return msgs


def _parse_arguments(args_data: dict) -> list:
    arguments = []
    
    # Safely get the arguments list, defaulting to empty list if missing or not a dict
    if not isinstance(args_data, dict):
        return arguments
        
    raw_args = args_data.get('arguments', [])
    
    # Ensure raw_args is actually a list before iterating
    if not isinstance(raw_args, list):
        return arguments
        
    for a in raw_args:
        # Skip any malformed entries that aren't dictionaries (e.g., stray strings)
        if not isinstance(a, dict):
            continue
            
        arguments.append(Argument(
            argument=a.get('argument', ''),
            agent=a.get('agent', ''),
            arg_type=a.get('type', ''),
            response=a.get('response', ''),
            justification_type=a.get('justification_for_type', ''),
            justification_response=a.get('justification_for_response', ''),
        ))
    return arguments


def _compute_basic_stats(record: DebateRecord):
    """Populates record.per_agent_stats with turn counts and token proxies."""
    stats = {}
    for m in record.messages:
        if m.name == 'UserProxy':
            continue
        if m.name not in stats:
            stats[m.name] = {
                'turns': 0,
                'total_words': 0,
                'total_chars': 0,
                'messages': [],
            }
        stats[m.name]['turns'] += 1
        stats[m.name]['total_words'] += m.word_count
        stats[m.name]['total_chars'] += m.char_count
        stats[m.name]['messages'].append(m)

    # Enrich with agent_map info
    for agent, s in stats.items():
        info = record.agent_map.get(agent, {})
        s['model'] = info.get('model', 'unknown')
        s['value'] = info.get('value', 'unknown')
        s['action_number'] = info.get('action_number', 'unknown')
        s['avg_words_per_turn'] = (
            s['total_words'] / s['turns'] if s['turns'] > 0 else 0
        )

    record.per_agent_stats = stats


def _extract_temperature_tag(folder_name: str) -> Optional[str]:
    """Extract 'low'/'high' from folder names like gpt-oss-20b_low_vs_gpt-oss-20b_high."""
    if '_low' in folder_name:
        return 'low_high'
    return None


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_all_debates(base_dir: str) -> list:
    """
    Walk base_dir and return a list of DebateRecord for every debate found.
    Skips debates where the main JSON cannot be parsed.
    """
    records = []

    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    for folder_name in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        if not folder_name.startswith('reasoning_'):
            continue

        args_dir = os.path.join(folder_path, 'arguments')

        for fname in sorted(os.listdir(folder_path)):
            if not fname.endswith('.json'):
                continue

            debate_path = os.path.join(folder_path, fname)
            debate_id_stem = fname.replace('.json', '')

            # Load main debate JSON
            try:
                with open(debate_path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
            except Exception as e:
                logger.warning("Could not load %s: %s", debate_path, e)
                continue

            # Load arguments JSON (optional)
            args_data = {}
            args_path = os.path.join(args_dir, f"{debate_id_stem}.arguments.json")
            if os.path.isfile(args_path):
                try:
                    with open(args_path, 'r', encoding='utf-8') as f:
                        args_data = json.load(f)
                except Exception as e:
                    logger.warning("Could not load args %s: %s", args_path, e)

            # Build record
            agent_map = _build_agent_map(raw)
            messages = _parse_messages(raw.get('messages', []))
            arguments = _parse_arguments(args_data)
            final_verdict = args_data.get('final_verdict', None)

            # If no arguments file, try to extract verdict from messages
            if final_verdict is None:
                final_verdict = _infer_verdict_from_messages(messages)

            record = DebateRecord(
                debate_id=raw.get('debate_id', debate_id_stem),
                folder=folder_name,
                file_path=debate_path,
                model_1=raw.get('model_1', ''),
                model_2=raw.get('model_2', ''),
                values1=raw.get('values1', ''),
                values2=raw.get('values2', ''),
                action1=raw.get('action1', ''),
                action2=raw.get('action2', ''),
                dilemma=raw.get('dilemma', ''),
                context=raw.get('context', ''),
                agent_map=agent_map,
                messages=messages,
                arguments=arguments,
                final_verdict=final_verdict,
                temperature_tag=_extract_temperature_tag(folder_name),
            )
            _compute_basic_stats(record)
            records.append(record)

    logger.info("Loaded %d debate records from %s", len(records), base_dir)
    return records


def _infer_verdict_from_messages(messages: list) -> Optional[str]:
    """
    Fallback: scan last agent messages for 'Action 1' / 'Action 2' and
    take the most recent explicit statement.
    """
    action_re = re.compile(r'\bAction\s+([12])\b', re.IGNORECASE)
    for m in reversed(messages):
        if m.name == 'UserProxy':
            continue
        matches = action_re.findall(m.content_clean)
        if matches:
            return f"Action {matches[-1]}"
    return None


# ---------------------------------------------------------------------------
# Convenience: group records
# ---------------------------------------------------------------------------

def group_by(records: list, key_fn) -> dict:
    """Generic grouper."""
    groups = {}
    for r in records:
        k = key_fn(r)
        groups.setdefault(k, []).append(r)
    return groups


def normalize_model_name(model_str: str) -> str:
    """
    Normalize many naming variants seen across runs into a canonical short name.

    IMPORTANT:
      - This function will append ":low"/":high" ONLY when those markers are present
        on the *given string* (e.g., "qwen3-8b_low", "gpt-oss-20b-high").
      - For debates where both sides are the same base model but with different
        reasoning efforts, you must append the mode using reasoning_effort_1/2
        (handled in _build_agent_map below).
    """
    if model_str is None:
        return "unknown"

    s = str(model_str).strip()
    if not s:
        return "unknown"

    lower = s.lower()
    last = s.split("/")[-1].lower()
    canon = re.sub(r"\s+", "-", last).replace("_", "-")

    # Detect explicit low/high markers ON THIS STRING
    mode: Optional[str] = None
    if re.search(r"(^|[^a-z0-9])low([^a-z0-9]|$)", lower) or "-low" in canon:
        mode = "low"
    if re.search(r"(^|[^a-z0-9])high([^a-z0-9]|$)", lower) or "-high" in canon:
        mode = "high" if mode is None else None  # if both, ambiguous -> None

    def with_mode(name: str) -> str:
        return f"{name}:{mode}" if mode in ("low", "high") else name

    # --- DeepSeek ---
    if "deepseek-r1-distill-qwen-14b" in lower:
        return with_mode("deepseek-r1-distill-qwen-14b")
    if "deepseek-r1-distill-qwen-7b" in lower:
        return with_mode("deepseek-r1-distill-qwen-7b")

    # --- GPT OSS 20B ---
    if "gpt-oss-20b" in lower or "gpt_oss-20b" in lower:
        return with_mode("gpt-oss-20b")

    # --- LLaMA 3.1 8B ---
    if "llama" in lower and "3.1" in lower and ("8b" in lower or "8-b" in lower):
        return with_mode("llama3.1-8b")

    # --- Gemma ---
    if "gemma" in lower:
        if "27b" in lower:
            return with_mode("gemma3-27b")
        if "12b" in lower and ("-it" in canon or re.search(r"(^|[^a-z0-9])it([^a-z0-9]|$)", lower)):
            return with_mode("gemma3-12b-it")
        if "12b" in lower:
            return with_mode("gemma3-12b")

    # --- Qwen3 ---
    if "qwen3" in lower:
        if "32b" in lower:
            return with_mode("qwen3-32b")
        if "8b" in lower:
            return with_mode("qwen3-8b")
        return with_mode("qwen3-8b")

    return with_mode(s.split("/")[-1])