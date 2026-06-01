# STYLE Baseline — Ambiguity-CHA

Implementation of **STYLE** (*Domain-Invariant Strategy Planning for Conversational Search*) adapted as a baseline for the Ambiguity-CHA benchmark (mental-health and food-safety ambiguity datasets).

This repo implements the paper methodology:

- **DISP** — frozen truncated BERT encoder + dueling DQN strategy planner  
- **MDT** — multi-domain reinforcement learning with experience replay  
- **LLM components** — GPT-3.5 clarification questions and user simulation (not trained)

## Architecture

State at turn \(t\):

\[
s_t = \mathbf{H}_t \oplus \mathbf{D}_t \oplus \text{score}^{1:k}_t
\]

| Component | Description |
|-----------|-------------|
| \(\mathbf{H}_t\) | BERT encoding of full conversation history |
| \(\mathbf{D}_t\) | Concatenated encodings of top-\(k\) retrieved documents |
| \(\text{score}^{1:k}_t\) | Retrieval scores for each of \(k\) documents |

**Actions:** `0` = answer, `1` = ask (binary planner).

**Rewards:** `+1.0` successful retrieval, `-0.5` timeout/failure, `0.0` on ask turns.

## Project structure

```
.
├── README.md
├── requirements.txt
├── setup.py
├── .env.example
├── configs/
│   ├── paper_style.yaml      # Paper hyperparameters (Appendix F.1)
│   └── diagnosis.yaml        # Diagnosis-domain adaptation
├── scripts/
│   ├── train_paper_domains.py       # MDT on ClariQ / OpenDialKG
│   ├── run_three_scenarios.py       # Zero-shot / fine-tune / in-domain
│   ├── train_diagnosis_scratch.py   # Train from scratch on diagnosis CSV
│   ├── finetune_diagnosis.py        # Fine-tune a checkpoint on diagnosis
│   ├── evaluate_new_datasets.py     # Eval on mental-health + food JSON
│   └── evaluate_food_safety.py      # Food-safety binary eval
└── style/
    ├── config.py
    ├── models/
    │   ├── disp.py             # Dueling DQN strategy planner
    │   ├── bert_encoder.py     # Frozen cocodr-base-msmarco (3 layers)
    │   ├── state_builder.py    # [H_t || D_t || scores] construction
    │   └── retriever.py
    ├── training/
    │   ├── mdt.py              # Multi-domain RL training loop
    │   └── train.py
    ├── data/                   # Dataset loaders & adapters
    ├── utils/
    │   └── llm_integration.py  # GPT clarification + user simulation
    ├── simulation/
    ├── evaluation/
    └── examples/
```

## Installation

```bash
git clone <repo-url>
cd STYLE
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Copy environment template and set your OpenAI key (required for MDT training):

```bash
cp .env.example .env
# edit .env → OPENAI_API_KEY=sk-...
```

### Data setup

Training on paper domains requires ClariQ and OpenDialKG JSON splits under `data/`:

```
data/
├── clariq_train.json
├── clariq_dev.json
├── clariq_test.json
├── opendialkg_train.json
├── opendialkg_dev.json
├── opendialkg_test.json
└── document_texts.json
```

For Ambiguity-CHA target benchmarks, place mental-health and food JSON files as documented in the parent [Ambiguity-CHA](https://github.com/Saba-Farahani/Ambiguity-CHA) repository.

## Usage

### 1. Train on paper source domains

```bash
python scripts/train_paper_domains.py \
  --domains clariq opendialkg \
  --episodes 1800 \
  --output-dir saved_models/paper_domains
```

### 2. Three transfer scenarios (target domain)

```bash
python scripts/run_three_scenarios.py \
  --target-train data/mental_health_train.json \
  --target-test  data/mental_health_test.json \
  --output-dir   saved_models/scenarios
```

| Scenario | Training data | Use case |
|----------|---------------|----------|
| 1 — Zero-shot | ClariQ + OpenDialKG | Evaluate on target without target training |
| 2 — Fine-tuned | Scenario 1 → target train | Transfer with light adaptation |
| 3 — In-domain | Target train only | Fully supervised on target |

### 3. Diagnosis benchmark

```bash
# From scratch
python scripts/train_diagnosis_scratch.py --data data/train/diagnosis_train.csv

# Fine-tune a source checkpoint
python scripts/finetune_diagnosis.py --base-model saved_models/paper_domains/final_model.pt
```

### 4. Evaluate on new datasets

```bash
python scripts/evaluate_new_datasets.py \
  --mental-health data/mental_health_test.json \
  --food         data/food_test.json

python scripts/evaluate_food_safety.py
```

## Key hyperparameters (paper defaults)

| Parameter | Value |
|-----------|-------|
| BERT encoder | `OpenMatch/cocodr-base-msmarco` (3 layers, frozen) |
| Top-k documents | 5 |
| Max turns \(T\) | 10 |
| Episodes | 1800 |
| Replay buffer | 10,000 |
| Batch size | 32 |
| Learning rate | \(1 \times 10^{-4}\) |
| Discount \(\gamma\) | 0.99 |
| LLM | `gpt-3.5-turbo` |

## Evaluation metrics

- **SR@k** — success rate when target is in top-\(k\) retrieved docs  
- **Recall@5** — retrieval recall  
- **AvgT** — average clarification turns  
- **Strategy diversity** — action-sequence diversity across episodes  

For Ambiguity-CHA food/mental benchmarks, scripts also report accuracy, abstention rate, and clarification efficiency.

## Citation

If you use this baseline, cite the original STYLE paper and the Ambiguity-CHA benchmark.

## License

See repository license. ClariQ, OpenDialKG, and Synthea-derived data are subject to their respective licenses — obtain and preprocess them separately.
