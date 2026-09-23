# Clinical NLP prompt extraction with Ollama

Refactored from the original `prompts_PLN` Jupyter notebook export. The repeated per-variable request functions have been replaced by a single config-driven extraction engine.

## Structure

```text
prompts_PLN_publication_ready/
├── prompts_pln.py
├── requirements.txt
└── configs/
    ├── pipeline.yml
    └── prompts/
        ├── anamnesis.yml
        ├── sintomas.yml
        ├── signos_vitales.yml
        ├── evolucion.yml
        └── bacteriemia.yml
```

The YAML files contain the original system prompts grouped by clinical category. This keeps prompt wording explicit and version-controllable, while the Python script contains only reusable execution logic.

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Edit `configs/pipeline.yml` if the input filenames or column names differ. Set the Ollama endpoint in the environment rather than hard-coding it:

```bash
export OLLAMA_ENDPOINT="http://HOST:11434/api/generate"
export OLLAMA_MODEL="llama3.2:3b"
```

## Run all prompts

```bash
python prompts_pln.py
```

Run selected categories only:

```bash
python prompts_pln.py --category sintomas --category signos_vitales
```

Choose another output file:

```bash
python prompts_pln.py --output-dir results
```

## Reproducibility notes

- Model, temperature, timeout, and retry behavior are centralized in YAML.
- Every extraction variable has a stable name, source text field, category, and system prompt.
- The script preserves the model response as raw text instead of silently coercing values.
- Anamnesis and clinical-evolution outputs are kept in separate tables, preventing unsafe row-wise merging without an explicit patient identifier.
- For a manuscript, report the exact model/version, inference server, temperature, prompt configuration commit, preprocessing rules, and validation procedure.
- Clinical data should remain within the approved computing environment; avoid sending identifiable text to external services unless explicitly authorized.
