# C-BERT: Factorized Causal Relation Extraction

A multi-task transformer framework for extracting **fine-grained causal attributions** from German text. C-BERT jointly performs span recognition and relation classification, decomposing causal influence into role, polarity, and salience.

📄 **Paper:** [C-BERT: Factorized Causal Relation Extraction](https://tuprints.ulb.tu-darmstadt.de/handle/tuda/15159)  
🤗 **Models:** [C-BERT v3 (recommended)](https://huggingface.co/pdjohn/C-EBERT-610m-v3) · [C-BERT v2](https://huggingface.co/pdjohn/C-EBERT-610m-v2)  
📊 **Dataset:** [Bundestag Causal Attribution](https://huggingface.co/datasets/pdjohn/bundestag-causal-attribution)  
📚 **Docs:** [Causal Semantics](https://causalsemantics.com/extraction/c-bert.html)
## What C-BERT Does

C-BERT extracts causal relations as **(Cause, Effect, Influence)** triples, where Influence $∈ [−1, +1]$ encodes both **polarity** (promoting vs. inhibiting) and **salience** (monocausal vs. polycausal attribution strength).

**Task 1 — Span Recognition** identifies causal indicators and entities via BIOES tagging:

| Token | Label | Conf. |
| :--- | :--- | :--- |
| **Pesticide bans** | **B-ENTITY** | **0.99** |
| **cause** | **B-INDICATOR** | **1.00** |
| the | O | 1.00 |
| disappearance | O | 0.88 |
| of | O | 1.00 |
| **insect decline** | **B-ENTITY** | **0.71** |

**Task 2 — Relation Classification** determines the causal relation for each indicator–entity pair:

| Indicator | Entity | Role | Polarity | Salience | Label |
| :--- | :--- | :--- | :--- | :--- | :--- |
| verursachen | Pesticide bans | CAUSE | POS | MONO | MONO_POS_CAUSE |
| verursachen | insect decline | EFFECT | NEG | MONO | MONO_NEG_EFFECT |

→ **Tuple:** (Pesticide bans, insect decline, $I = −1.0$) — pesticide bans cause the disappearance of insect decline.

## Two Architectures

| | **v2 (Unified)** | **v3 (Factorized)** ⭐ |
| :--- | :--- | :--- |
| Relation head | Single 14-class softmax | 3 parallel heads (role, polarity, salience) |
| Accuracy | 75.3% | **76.9%** |
| Multi-seed mean | 0.744 ± 0.007 | **0.768 ± 0.009** |
| Entity span F1 | 0.691 | **0.765** |
| Multi-head errors | 22.6% | **16.2%** |
| Best for | Simplicity | Accuracy, interpretability |

The factorized v3 model outperforms v2 on **all five random seeds** tested.

## Installation

```bash
git clone https://github.com/padjohn/cbert.git
cd cbert
pip install .
```

## Quick Start

```python
from causalbert.infer import load_model, sentence_analysis

model, tokenizer, config, device = load_model("pdjohn/C-EBERT-610m-v3")

sentences = [
    "Pestizide verursachen Insektensterben.",
    "Pestizidverbote verursachen die Vernichtung von Insektensterben.",
    "Pestizide sind die Hauptursache von Insektensterben.",
]

results = sentence_analysis(model, tokenizer, config, sentences, device=device, batch_size=8)

for analysis in results:
    print(f"\n{analysis['sentence']}")
    for (indicator, entity), rel in analysis['derived_relations']:
        print(f"  {indicator} → {entity}: {rel['label']} (I={rel['influence']:+.2f})")
```

**Output:**
```
Pestizide verursachen Insektensterben.
  verursachen → Pestizide: MONO_POS_CAUSE (I = +1.00)
  verursachen → Insektensterben: MONO_POS_EFFECT (I = +1.00)

Pestizidverbote verursachen die Vernichtung von Insektensterben.
  verursachen → Pestizidverbote: MONO_POS_CAUSE (I = +1.00)
  verursachen → Insektensterben: MONO_NEG_EFFECT (I = -1.00)

Pestizide sind die Hauptursache von Insektensterben.
  Hauptursache → Pestizide: PRIO_POS_CAUSE (I = +0.75)
  Hauptursache → Insektensterben: MONO_POS_EFFECT (I = +1.00)
```

## Command-Line Inference

```bash
python run_inference.py \
    --model_path "pdjohn/C-EBERT-610m-v3" \
    --sentences "Autoverkehr verursacht Bienensterben." "Lärm ist der Grund für Stress."
```

## Project Structure

```
causalbert/
├── model.py       # CausalBERTMultiTaskModel (v2 + v3 architectures)
├── train.py       # Training with multi-task batching and LoRA
├── infer.py       # Inference: span detection → relation classification
├── dataset.py     # Dataset creation from annotated JSON
└── utils.py       # Shared utilities
```

## Training

### 1. Data Preparation

```bash
python -m causalbert.dataset \
    --input_json_path data/raw_input/all_sentences.json \
    --base_dir ./data \
    --model_name "EuroBERT/EuroBERT-610m" \
    --augment 2 \
    --prop_neg skip
```

Key options:
- `--augment 2`: Include both original and augmented sentences (entity replacement)
- `--prop_neg skip`: Drop propositionally negated relations from training

### 2. Train

```bash
python -m causalbert.train \
    --epochs 7 \
    --model_name "EuroBERT/EuroBERT-610m" \
    --architecture_version 3 \
    --lr 3e-4 \
    --batch_size 32
```

Training uses LoRA (r=16, α=32) for parameter-efficient fine-tuning. Both architectures share identical hyperparameters; see the [paper](https://tuprints.ulb.tu-darmstadt.de/handle/tuda/15159) for full details.

### 3. Checkpoint Merging

Training saves LoRA adapters at each epoch. To merge an intermediate checkpoint into a standalone model:

```python
from causalbert.utils import merge_checkpoint
merge_checkpoint(
    final_checkpoint="data/model/C-EBERT-v3/",     # has config + tokenizer
    lora_checkpoint="data/model/C-EBERT-v3/checkpoint-1424/",  # epoch 4
    output_dir="data/model/C-EBERT-v3-merged/"
)
```

## Model Architecture

C-BERT extends [EuroBERT-610m](https://huggingface.co/EuroBERT/EuroBERT-610m) with task-specific heads:

**Span Recognition:** BIOES token classification (9 labels: B/I/O/E/S × {INDICATOR, ENTITY}) with weighted cross-entropy loss.

**Relation Classification (v3):** Input format `[indicator] <|parallel_sep|> [entity] <|parallel_sep|> [sentence]`, classified by three parallel heads from the CLS representation:
- **Role** (3-class): CAUSE, EFFECT, NO_RELATION
- **Polarity** (2-class, masked for NO_RELATION): POS, NEG
- **Salience** (3-class, masked for NO_RELATION): MONO, PRIO, DIST

Polarity and salience heads are masked during training and inference for NO_RELATION predictions. Influence is reconstructed as $I = \text{sign(polarity)} × \text{salience-value}$.

**Relation Classification (v2):** Same input format, single 14-class softmax over the full combinatorial label space.

## Indicator Taxonomy

The repository includes a taxonomy of **644 German causal indicators** across 42 semantic families, each classified by base polarity and salience:

| Family | Example | Polarity | Salience |
| :--- | :--- | :--- | :--- |
| Cause | *verursachen* | + | MONO |
| Contribution | *beitragen* | + | DIST |
| Stop | *stoppen* | − | MONO |
| Reduce | *reduzieren* | − | DIST |

See `data/indicator_taxonomy.csv` for the complete taxonomy.

## Citation

```bibtex
@article{johnson2026cbert,
  title={C-BERT: Factorized Causal Relation Extraction},
  author={Johnson, Patrick},
  year={2026},
  doi={10.26083/tuda-7797}
}
```

## License

Apache License 2.0. See [LICENSE](./LICENSE).

The fine-tuned model weights inherit the license of the base model ([EuroBERT-610m](https://huggingface.co/EuroBERT/EuroBERT-610m), Apache 2.0).
