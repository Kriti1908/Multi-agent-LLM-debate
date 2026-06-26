import os
import sys
import logging
import time
import shutil
import random
from typing import Optional, Callable, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('orchestrator')

from data_loader import load_all_debates
from metric_ibc_consensus import compute_per_llm
from metric_unethical_strategy import compute_unethical_strategy
from results_writer import write_all_results

# ============================
# CONFIG
# ============================

BASE_DIRS = [
    "Reasoning vs Non-reasoning",
    "Small vs Large",
]

MODEL_NAME = "deepseek-ai/deepseek-llm-7b-chat"

MAX_RETRIES = 3


# ============================
# LOAD LOCAL MODEL
# ============================

def load_local_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    logger.info("Loading local model: %s", MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        offload_folder="offload",          # ✅ FIX ADDED
        offload_state_dict=True,           # ✅ STABILITY IMPROVEMENT
    )

    return tokenizer, model


# ============================
# LOCAL JUDGE FUNCTION
# ============================

def build_judge_fn(tokenizer, model) -> Callable[[str, str], str]:

    def judge_fn(prompt: str, system: str) -> str:
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                full_prompt = f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>\n"

                inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )

                response = tokenizer.decode(outputs[0], skip_special_tokens=True)

                if "<|assistant|>" in response:
                    response = response.split("<|assistant|>")[-1].strip()

                return response

            except Exception as e:
                last_error = e
                wait_time = min(2 ** attempt, 10) + random.uniform(0, 1)
                logger.warning(
                    "Local model failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1,
                    MAX_RETRIES,
                    str(e),
                    wait_time,
                )
                time.sleep(wait_time)

        raise last_error

    return judge_fn


# ============================
# HELPERS
# ============================

def is_already_processed(folder_name: str) -> bool:
    return os.path.isfile(os.path.join("results", folder_name, "metrics_full.json"))


# ============================
# MAIN
# ============================

def main():
    tokenizer, model = load_local_model()
    judge_fn = build_judge_fn(tokenizer, model)

    all_folders = []

    for base in BASE_DIRS:
        if not os.path.isdir(base):
            logger.warning("Base directory not found, skipping: %s", base)
            continue

        for sub in sorted(os.listdir(base)):
            full_path = os.path.join(base, sub)
            if os.path.isdir(full_path) and sub.startswith("reasoning_"):
                all_folders.append(full_path)

    if not all_folders:
        logger.warning("No matching folders found.")
        print("\nDONE")
        return

    for folder in all_folders:
        folder_name = os.path.basename(folder)

        if is_already_processed(folder_name):
            logger.info("Skipping: %s", folder_name)
            continue

        logger.info("Processing: %s", folder_name)

        temp_base = "__temp_single_run__"
        try:
            if os.path.exists(temp_base):
                shutil.rmtree(temp_base)
            os.makedirs(temp_base, exist_ok=True)

            shutil.copytree(folder, os.path.join(temp_base, folder_name))

            records = load_all_debates(temp_base)

        finally:
            if os.path.exists(temp_base):
                shutil.rmtree(temp_base, ignore_errors=True)

        if not records:
            logger.warning("No records in %s", folder_name)
            continue

        per = compute_per_llm(records, judge_fn=judge_fn)
        usa = compute_unethical_strategy(records, judge_fn=judge_fn)

        out_dir = os.path.join("results", folder_name)
        os.makedirs(out_dir, exist_ok=True)

        all_results = {
            'per': per,
            'usa': usa,
            '_meta': {
                'num_debates': len(records),
                'base_dir': folder,
                'judge_model': MODEL_NAME,
                'llm_judge_used': True,
            },
        }

        write_all_results(all_results, out_dir=out_dir)

        logger.info("Finished: %s", folder_name)

    print("\nDONE")


if __name__ == '__main__':
    main()