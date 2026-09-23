#!/usr/bin/env python3
"""Config-driven extraction of structured clinical variables from free text using Ollama.

This script is a refactored, reproducible version of the original Jupyter notebook.
Prompt definitions are stored in YAML files under ``configs/prompts`` so they can be
reviewed, versioned, and modified without changing Python code.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

LOGGER = logging.getLogger("prompts_pln")


@dataclass(frozen=True)
class OllamaConfig:
    endpoint: str
    model: str
    temperature: float = 0.2
    timeout_seconds: int = 60
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_prompt_definitions(prompt_dir: Path, categories: list[str] | None = None) -> list[dict[str, Any]]:
    files = sorted(prompt_dir.glob("*.yml"))
    if categories:
        wanted = set(categories)
        files = [p for p in files if p.stem in wanted]
        missing = wanted - {p.stem for p in files}
        if missing:
            raise ValueError(f"Unknown prompt categories: {', '.join(sorted(missing))}")

    prompts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in files:
        data = load_yaml(path)
        category = data.get("category", path.stem)
        for spec in data.get("prompts", []):
            name = spec["name"]
            if name in seen:
                raise ValueError(f"Duplicate prompt name: {name}")
            seen.add(name)
            prompts.append({**spec, "category": category})
    return prompts


def build_ollama_config(config: dict[str, Any], endpoint_override: str | None, model_override: str | None) -> OllamaConfig:
    cfg = config["ollama"]
    endpoint = endpoint_override or os.getenv(cfg.get("endpoint_env", "OLLAMA_ENDPOINT"), "")
    model = model_override or os.getenv(cfg.get("model_env", "OLLAMA_MODEL"), cfg.get("default_model", "llama3.2:3b"))
    if not endpoint:
        raise ValueError("Ollama endpoint is empty. Set OLLAMA_ENDPOINT or pass --endpoint.")
    return OllamaConfig(
        endpoint=endpoint,
        model=model,
        temperature=float(cfg.get("temperature", 0.2)),
        timeout_seconds=int(cfg.get("timeout_seconds", 60)),
        max_retries=int(cfg.get("max_retries", 3)),
        retry_backoff_seconds=float(cfg.get("retry_backoff_seconds", 2.0)),
    )


def query_ollama(session: requests.Session, text: str, system_prompt: str, cfg: OllamaConfig) -> str:
    """Query Ollama and return a stripped response, retrying transient failures."""
    payload = {
        "model": cfg.model,
        "prompt": "" if pd.isna(text) else str(text),
        "system": system_prompt,
        "stream": False,
        "options": {"temperature": cfg.temperature},
    }
    last_error: Exception | None = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            response = session.post(cfg.endpoint, json=payload, timeout=cfg.timeout_seconds)
            response.raise_for_status()
            result = response.json().get("response")
            if result is None:
                raise ValueError("Ollama response does not contain a 'response' field")
            return str(result).strip()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            LOGGER.warning("Ollama request failed (attempt %d/%d): %s", attempt, cfg.max_retries, exc)
            if attempt < cfg.max_retries:
                time.sleep(cfg.retry_backoff_seconds * attempt)
    raise RuntimeError(f"Ollama request failed after {cfg.max_retries} attempts") from last_error


def load_sources(config: dict[str, Any]) -> tuple[dict[str, pd.Series], dict[str, str]]:
    """Load configured text columns and record which input dataset each source belongs to."""
    inputs = config["inputs"]
    sources: dict[str, pd.Series] = {}
    source_dataset: dict[str, str] = {}

    for dataset_name, dataset_cfg in inputs.items():
        path = Path(dataset_cfg["path"])
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        frame = pd.read_excel(path)
        for source_name, column in dataset_cfg.get("columns", {}).items():
            if source_name in sources:
                raise ValueError(f"Duplicate source alias: {source_name}")
            if column not in frame.columns:
                raise KeyError(f"Column '{column}' not found in {path}")
            sources[source_name] = frame[column].fillna("").reset_index(drop=True)
            source_dataset[source_name] = dataset_name
    return sources, source_dataset


def extract_variables(
    sources: dict[str, pd.Series],
    source_dataset: dict[str, str],
    prompts: list[dict[str, Any]],
    ollama: OllamaConfig,
) -> dict[str, pd.DataFrame]:
    """Run prompts and return one extraction table per original input dataset.

    Keeping datasets separate avoids assuming that rows in the anamnesis and
    evolution workbooks refer to the same patients in the same order.
    """
    outputs: dict[str, pd.DataFrame] = {}
    with requests.Session() as session:
        for idx, spec in enumerate(prompts, start=1):
            name = spec["name"]
            source_name = spec["source"]
            if source_name not in sources:
                raise KeyError(f"Prompt '{name}' refers to unknown source '{source_name}'")
            dataset = source_dataset[source_name]
            source = sources[source_name]
            output = outputs.setdefault(dataset, pd.DataFrame(index=source.index))
            if len(output) != len(source):
                raise ValueError(f"Sources assigned to dataset '{dataset}' have inconsistent lengths")
            LOGGER.info("[%d/%d] Extracting %s (%s)", idx, len(prompts), name, spec["category"])
            output[name] = source.apply(
                lambda text: query_ollama(session, text, spec["system_prompt"], ollama)
            )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.yml"))
    parser.add_argument("--prompt-dir", type=Path, default=Path("configs/prompts"))
    parser.add_argument("--category", action="append", dest="categories", help="Run only this category; may be repeated.")
    parser.add_argument("--endpoint", help="Override the Ollama endpoint.")
    parser.add_argument("--model", help="Override the Ollama model.")
    parser.add_argument("--output-dir", type=Path, help="Directory for extracted tables.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")

    config = load_yaml(args.config)
    prompts = load_prompt_definitions(args.prompt_dir, args.categories)
    if not prompts:
        raise ValueError("No prompt definitions were found.")

    ollama = build_ollama_config(config, args.endpoint, args.model)
    sources, source_dataset = load_sources(config)
    results = extract_variables(sources, source_dataset, prompts, ollama)

    output_dir = args.output_dir or Path(config.get("output", {}).get("directory", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset, result in results.items():
        output_path = output_dir / f"extracciones_{dataset}.csv"
        result.to_csv(output_path, index=False)
        LOGGER.info(
            "Saved %d rows x %d variables to %s",
            len(result),
            len(result.columns),
            output_path,
        )


if __name__ == "__main__":
    main()
