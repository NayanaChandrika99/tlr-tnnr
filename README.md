<p align="center"><strong>Tiny Health Coding Extractor</strong></p>

# Tiny Reasoning Model for Medical Coding

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Model](https://img.shields.io/badge/model-SmolLM2--135M-purple.svg)
![Size](https://img.shields.io/badge/params-135M-orange.svg)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)

**Turnkey repository for building a 135M parameter SmolLM2 model that converts free-form clinical narratives into ICD-10, CPT, and HCPCS billing codes with transparent `<think>` reasoning**

---

## Why This Project?

Medical billing requires translating **free-text clinical notes** (doctor's narratives, procedure descriptions) into **standardized codes** (ICD-10 diagnoses, CPT procedures, HCPCS supplies). This is:

- **Time-consuming**: Human coders spend 5-10 minutes per note
- **Error-prone**: Incorrect codes lead to claim denials and revenue loss
- **Opaque**: Even when automated, coders can't see the model's reasoning
- **Expensive**: Large language models (70B+ params) are costly to run

**THCE solves this with**:
- **Small, efficient model** (135M params) that runs on CPU or single GPU
- **Transparent reasoning** via `<think>` tags showing step-by-step logic
- **Multi-code support** for ICD-10 (diagnoses), CPT (procedures), HCPCS (supplies)
- **Domain-specific data** with curated code databases and medical narratives
- **Production-ready** with validation, quality gates, and evaluation pipelines

---

## Visual Flow

```
┌────────────────────────────────────┐
│  Clinical Narrative (Free-Text)   │
│                                    │
│  "Patient presented with acute     │
│   chest pain radiating to left    │
│   arm. ECG showed ST elevation.   │
│   Performed cardiac catheterization│
│   with stent placement."           │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│   Tokenization & Preprocessing    │
│   (SmolLM2 tokenizer)              │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────┐
│   SmolLM2-135M Model                           │
│   ┌──────────────────────────────────────────┐ │
│   │  Input: Clinical narrative               │ │
│   │  Process: Generate reasoning trace       │ │
│   │  Output: <think> + codes + confidence    │ │
│   └──────────────────────────────────────────┘ │
└────────────┬───────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────┐
│   Reasoning Output                             │
│   <think>                                      │
│   - Chief complaint: chest pain                │
│   - Key finding: ST elevation on ECG           │
│   - Diagnosis: acute myocardial infarction     │
│   - Procedure: cardiac cath + stent            │
│   </think>                                     │
│                                                │
│   ICD-10: I21.09 (ST elevation MI, anterior)  │
│   CPT: 92928 (Coronary angioplasty + stent)   │
│   HCPCS: C1876 (Drug-eluting coronary stent)  │
│   Confidence: 0.92                             │
└────────────┬───────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│   Code Validation                  │
│   • Check against code databases   │
│   • Verify code formats            │
│   • Flag deprecated/invalid codes  │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│   Output to Billing System         │
│   ✓ Validated codes                │
│   ✓ Reasoning trace (audit trail)  │
│   ✓ Confidence scores              │
└────────────────────────────────────┘
```

---

## Tech Stack

- **Base Model**: SmolLM2-135M (HuggingFace Transformers)
- **Training Framework**: PyTorch with Hugging Face `transformers` and `trl`
- **Training Methods**: Supervised Fine-Tuning (SFT) + Direct Preference Optimization (DPO)
- **Package Manager**: `uv` (fast Python dependency resolver)
- **Code Validation**: Custom validators for ICD-10, CPT, HCPCS
- **Evaluation**: Exact match, partial match, code-level F1, reasoning quality
- **Data Quality**: Automated quality checks with thresholds per training stage
- **CI/CD**: GitHub Actions with pre-commit hooks (ruff, black, mypy)
- **Experiment Tracking**: W&B (optional), local JSON reports

---

## Features & ML Components

### Core Capabilities
✓ **Multi-Code Generation** — ICD-10 (diagnoses), CPT (procedures), HCPCS (supplies)  
✓ **Transparent Reasoning** — `<think>` tags expose model logic for audit/trust  
✓ **Small & Efficient** — 135M params run on CPU or single consumer GPU  
✓ **Code Validation** — Real-time checks against curated code databases  
✓ **Stage-Based Training** — Progressive curriculum from simple to complex narratives  
✓ **Quality Gates** — Automated data quality checks per training stage  
✓ **Evaluation Suite** — Exact match, code-level metrics, baseline comparisons  
✓ **Production-Ready** — Pre-commit hooks, CI tests, model deployment scripts  

### Training Stages
- **Stage 1**: Simple narratives → single ICD-10 code (baseline)
- **Stage 2**: Multi-code narratives → ICD-10 + CPT with reasoning
- **Stage 3**: Complex cases → ICD-10 + CPT + HCPCS + DPO preference pairs

### Data Assets
- `data/code_databases/*.json` — ICD-10, CPT, HCPCS code definitions
- `data/examples/sample_stage_*.jsonl` — Stage-specific training datasets
- `tests/fixtures/raw_records.jsonl` — Test fixtures for validation

## Repository Layout

```
tennr-trl/
├── data/
│   ├── code_databases/
│   │   ├── icd10_codes.json
│   │   ├── cpt_codes.json
│   │   └── hcpcs_codes.json
│   ├── examples/
│   │   ├── sample_stage_1.jsonl
│   │   ├── sample_stage_2.jsonl
│   │   └── sample_stage_3.jsonl
│   ├── config/
│   │   ├── stage_1.yaml
│   │   ├── stage_2.yaml
│   │   └── stage_3.yaml
│   ├── data_collection.py
│   ├── preprocess_medical.py
│   └── synthetic_generation.py
├── post_training/
│   ├── config/
│   │   ├── stage_1.yaml
│   │   ├── stage_2.yaml
│   │   └── stage_3.yaml
│   ├── sft.py                  # Supervised Fine-Tuning
│   └── dpo.py                  # Direct Preference Optimization
├── evaluation/
│   ├── evaluate_model.py       # Main evaluation entry point
│   ├── baseline.py             # Baseline strategies
│   ├── metrics.py              # Code-level metrics
│   └── analysis.py             # Report analysis tools
├── utils/
│   ├── code_validator.py       # Validate ICD-10/CPT/HCPCS codes
│   ├── data_quality.py         # Dataset quality checks
│   ├── tokenization.py         # Tokenizer helpers
│   └── medical_chat_templates.py # Prompt templates
├── scripts/
│   ├── generate_synthetic.py   # LLM-based data generation
│   └── generate_predictions.py # Batch inference
├── tests/
│   ├── fixtures/
│   │   └── raw_records.jsonl
│   ├── test_code_validator.py
│   ├── test_data_quality.py
│   ├── test_preprocessing.py
│   └── test_metrics.py
├── docs/
│   ├── DATA_PREPARATION.md     # Dataset schema and guidelines
│   ├── TRAINING.md             # Training workflow
│   └── EVALUATION.md           # Evaluation metrics
├── pyproject.toml              # Project metadata and dependencies
├── uv.lock                     # Locked dependency versions
├── .pre-commit-config.yaml     # Code quality hooks
└── README.md
```

---

## Documentation

- **[docs/DATA_PREPARATION.md](docs/DATA_PREPARATION.md)** — Dataset schema, stage definitions, quality requirements
- **[docs/TRAINING.md](docs/TRAINING.md)** — Training workflow, hyperparameter tuning
- **[docs/EVALUATION.md](docs/EVALUATION.md)** — Metric definitions, baseline comparisons

---

## License & Credits

This repository is provided for research and demonstration purposes.  
Inspired by the **Tiny Reasoning Language Model** work by Shekswess et al.  
Adapted for medical coding domain with ICD-10/CPT/HCPCS datasets.

**Key Technologies**:
- **SmolLM2-135M** by HuggingFace (efficient small language model)
- **TRL (Transformer Reinforcement Learning)** by HuggingFace (SFT/DPO training)
- **uv** by Astral (fast Python package manager)
- **Code Databases** derived from CMS (Centers for Medicare & Medicaid Services)

**Citation**:
```bibtex
@misc{thce2024,
  title={THCE: Tiny Health Coding Extractor},
  author={Tennr Health Engineering},
  year={2024},
  note={Medical coding with transparent reasoning}
}
```
