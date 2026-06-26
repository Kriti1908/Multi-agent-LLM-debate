import os
import json
import re
import time
import logging
from pathlib import Path
from multiprocessing import Pool, Queue
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from tqdm import tqdm
from openai import OpenAI

# =======================
# CONFIG
# =======================

VLLM_ENDPOINT = "http://localhost:8000/v1"
MODEL_NAME = "Qwen/Qwen3-32B"

NUM_WORKERS = 32
MAX_RETRIES = 3
RETRY_BACKOFF = 2
TEMPERATURE = 0.0
MAX_TOKENS = 8172*2

LOG_DIR = "logs"
MAX_LOG_SIZE = 100 * 1024 * 1024
BACKUP_COUNT = 5

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# =======================
# PROMPTS (UNCHANGED)
# =======================

SYSTEM_PROMPT = """
You are an Argument Analysis Engine.

Your task is to analyze a structured 2-agent debate transcript and extract arguments with extreme precision.

STRICT RULES (VIOLATIONS ARE FATAL):
1. You MUST ignore all text inside <think>...</think> tags completely.
2. You MUST quote arguments using the EXACT STRING from the visible text.
3. You MUST NOT paraphrase, summarize, or rewrite arguments.
4. If a sentence is not a clear argument (claim, justification, conclusion), DO NOT extract it.
5. Each argument must come from a single agent and a single turn.
6. If the other agent does not directly reference or respond to the argument, mark it as Ignored.
7. Output MUST be valid JSON only. No commentary.
8. All the arguments must be minimal and exact quotes.
9. If there are multiple arguments in a single turn, extract them separately.
10. Agreement/Disagreement must be determined ONLY from immediate next turn of the OTHER agent.

FINAL VERDICT RULE:
Return exactly ONE of:
- Action 1
- Action 2
- No Consensus
"""

USER_PROMPT_TEMPLATE = """
You are an Argument Analysis Engine.

Your task is to analyze a structured multi-agent debate transcript and extract arguments with extreme precision.

STRICT RULES (VIOLATIONS ARE FATAL):
1. You MUST ignore all text inside <think>...</think> tags completely.
2. You MUST quote arguments using the EXACT STRING from the visible text.
3. You MUST NOT paraphrase, summarize, or rewrite arguments.
4. If a sentence is not a clear argument (claim, justification, conclusion), DO NOT extract it.
5. Each argument must come from a single agent and a single turn.
6. If the other agent does not directly reference or respond to the argument, mark it as Ignored.
7. Output MUST be valid JSON only. No commentary.
8. All the arguments must be minimal and exact quotes. Do not mention huge text chunks as one argument. 
9. If there are multiple arguments in a single turn, extract them separately.
10. Agreement/Disagreement must be determined ONLY from immediate next turn of the OTHER agent. Not all subsequent turns.

DEFINITIONS:
- Strong argument: Explicitly reasoned, grounded in stated values, principles, or consequences.
- Weak argument: Vague, repetitive, purely assertive, or lacks justification.
- Agreed: Other agent explicitly supports or endorses the argument.
- Disagreed: Other agent explicitly challenges or rejects the argument.
- Ignored: No direct engagement.

FINAL VERDICT RULE:
Return exactly ONE of:
- Action 1
- Action 2
- No Consensus

This verdict is for the ENTIRE RUN, not individual arguments.
"""

USER_PROMPT_TEMPLATE = """\\no_think Analyze the following debate transcript.

Input JSON:
{datapoint_json}

Tasks:
1. Extract all explicit arguments (ignoring <think>).
2. For each argument, produce an object with:
   - argument (exact quoted string)
   - Agent
   - Type (Strong or Weak)
   - Response (Agreed / Disagreed / Ignored by the other agent)
   - Justification for Type
   - Justification for Response (exact quoted string from the other agent in the immediate next turn, Empty string if Ignored)
3. Determine the FINAL RUN VERDICT (Action 1 / Action 2 / No Consensus).

Output JSON format:

{{
  "debate_id": <id>,
  "final_verdict": "<Action 1 or Action 2 or No Consensus>",
  "arguments": [
    {{
      "argument": "...",
      "agent": "...",
      "type": "Strong | Weak",
      "response": "Agreed | Disagreed | Ignored",
      "justification_for_type": "...",
      "justification_for_response": "..."
    }}
  ]
}}
"""

# =======================
# LOGGING
# =======================

def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_queue = Queue(-1)

    formatter = logging.Formatter(
        "%(asctime)s | %(processName)s | %(levelname)s | %(message)s"
    )

    handlers = []
    for name in ["info.log", "errors.log"]:
        h = RotatingFileHandler(
            os.path.join(LOG_DIR, name),
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT,
        )
        h.setFormatter(formatter)
        handlers.append(h)

    listener = QueueListener(log_queue, *handlers)
    return log_queue, listener


def worker_init(log_queue):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(QueueHandler(log_queue))


# =======================
# HELPERS
# =======================

def strip_think(text: str) -> str:
    return THINK_RE.sub("", text).strip()


def preprocess_datapoint(dp: dict) -> dict:
    msgs = []
    for m in dp.get("messages", []):
        content = strip_think(m.get("content", ""))
        if content:
            msgs.append({**m, "content": content})
    dp["messages"] = msgs
    return dp


def extract_json(text: str) -> dict:
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        raise ValueError("No JSON found")
    return json.loads(text[s : e + 1])


# =======================
# WORKER
# =======================

def worker(task):
    input_path, output_path = task
    logger = logging.getLogger()

    with open(input_path) as f:
        dp = json.load(f)

    debate_id = dp["debate_id"]
    dp = preprocess_datapoint(dp)

    prompt = USER_PROMPT_TEMPLATE.format(
        datapoint_json=json.dumps(dp, indent=2)
    )

    client = OpenAI(base_url=VLLM_ENDPOINT, api_key="EMPTY")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
            )

            content = strip_think(resp.choices[0].message.content)
            parsed = extract_json(content)
            parsed["debate_id"] = debate_id

            with open(output_path, "w") as out:
                json.dump(parsed, out, indent=2)

            logger.info(f"Processed {input_path}")
            return

        except Exception as e:
            logger.warning(f"Retry {attempt} {input_path}: {e}")
            time.sleep(RETRY_BACKOFF * attempt)

    logger.error(f"FAILED {input_path}")


# =======================
# MAIN PIPELINE
# =======================

def process_all(input_root: Path):
    log_queue, listener = setup_logging()
    listener.start()

    try:
        tasks = []

        for subdir in input_root.iterdir():
            if not subdir.is_dir():
                continue

            out_dir = subdir / "arguments"
            out_dir.mkdir(exist_ok=True)

            for file in subdir.glob("*.json"):
                out_file = out_dir / f"{file.stem}.arguments.json"
                tasks.append((file, out_file))

        with Pool(
            processes=NUM_WORKERS,
            initializer=worker_init,
            initargs=(log_queue,),
        ) as pool:
            list(tqdm(pool.imap_unordered(worker, tasks), total=len(tasks)))

    finally:
        listener.stop()


# =======================
# ENTRYPOINT
# =======================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=Path, required=True)
    args = parser.parse_args()

    process_all(args.input_root)
