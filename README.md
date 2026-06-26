# Multi-Agent LLM Debate Framework for Moral Persuasion

A framework for studying **persuasion, value alignment, and decision stability** in multi-agent Large Language Model (LLM) debates under conflicting human values.

This project investigates whether one LLM can persuade another to change its decision when both agents defend opposing human values in ethical dilemmas. It includes a configurable debate engine, value-swapped evaluations, and an automated LLM-as-a-Judge pipeline for analyzing debate outcomes.

---

## Overview

Recent LLM systems increasingly interact with one another rather than operating in isolation. While collaborative reasoning can improve performance, it also raises important questions:

* Can one AI persuade another to change its decision?
* Does stronger reasoning always lead to more persuasive arguments?
* How do different model families behave under conflicting human values?

To explore these questions, we build a structured debate framework where two LLM agents argue from opposing value systems and attempt to reach consensus over moral dilemmas.

---

## Features

* Multi-agent debate framework supporting heterogeneous and homogeneous model pairings
* Configurable debate protocol with iterative argument and rebuttal rounds
* Value-conditioned prompting using human value definitions
* Value-swapped experiments to separate model persuasion ability from value framing
* Automated LLM-as-a-Judge evaluation pipeline
* Argument extraction and interaction analysis
* Support for large-scale experimentation across hundreds of debate instances

---

## Debate Workflow

1. Assign two agents conflicting human values.
2. Present both agents with the same moral dilemma.
3. Conduct a structured debate with alternating turns.
4. Stop when consensus is reached or the maximum number of rounds is exceeded.
5. Evaluate the complete debate using an LLM-as-a-Judge.
6. Repeat the experiment with swapped value assignments.

---

## Evaluation Pipeline

Each completed debate is automatically analyzed to extract:

* Final debate outcome
* Persuasion success
* Argument quality
* Argument acceptance
* Decision stability
* Outcome consistency
* Persuasion drift

The judge evaluates every argument individually, classifies responses (Agreed, Partially Agreed, Ignored, Disagreed), and produces aggregate metrics for downstream analysis.

---

## Experimental Setup

* **Dataset:** LitmusValues
* **9** human value categories
* **36** unique value pairings
* **25** dilemmas per value pair
* **900+** debate instances

Experiments compare:

* Homogeneous vs heterogeneous model pairings
* Small vs large models
* High-reasoning vs non-reasoning variants
* Original vs value-swapped assignments

---

## Models Evaluated

* Qwen3
* Gemma 3
* Llama 3.1
* GPT-OSS
* DeepSeek-R1 Distill

Multiple model sizes and reasoning variants are supported.

---

## Key Findings

* Persuasion can change model decisions even when arguments are weak.
* Smaller models can successfully persuade larger models.
* Stronger reasoning improves argument quality but does not necessarily increase persuasion.
* Persistence and verbosity often have a greater influence on debate outcomes than argument strength.
* Value assignments significantly affect final decisions, highlighting the importance of value alignment in multi-agent systems.

---

## Repository Structure

```text
.
├── data/                 # Moral dilemmas and value definitions
├── prompts/              # System, value, and judge prompts
├── debate/               # Multi-agent debate engine
├── evaluation/           # LLM-as-a-Judge pipeline
├── metrics/              # Persuasion and stability metrics
├── experiments/          # Experiment configurations
├── results/              # Debate logs and analysis
└── README.md
```

---

## Tech Stack

* Python
* PyTorch
* NumPy
* Hugging Face Transformers
* Ollama / LLM APIs
