# CausalBERT: A Multi-Task Model for German Causal Relation Extraction

CausalBERT is a natural language processing framework designed to extract **causal relationships** from German text. Built upon the `Transformer` architecture, this collection of scripts uses a multi-task fine-tuned encoder-only model to perform two key tasks:

1.  **Token Classification (Span Recognition)**: Identifying and labeling causal indicators and entities within a sentence (e.g., "B-INDICATOR", "I-ENTITY").
2.  **Relation Classification**: Determining the causal relationship (e.g., CAUSE, EFFECT, INTERDEPENDENCY, or NO_RELATION, with polarity/modality) between a given indicator and entity pair within a sentence.

This repository holds the code to create **datasets**, **train**, and run **inference**.

## Features

* **Multi-Task Learning**: Jointly trains on token classification and relation classification tasks for improved performance.
* **German Language Support**: Compatible with [EuroBERT/EuroBERT-610m](https://huggingface.co/EuroBERT/EuroBERT-610m) and other multiliangual BERTs.
* **Custom Token Handling**: Injects a special `<|parallel_sep|>` token for effective handling of multi-part inputs in relation classification.
* **Robust Data Pipeline**: Includes scripts for preparing datasets from raw JSON, handling tokenization, and BIO labeling.
* **Weighted Loss**: Dynamically calculates and applies class weights during training to address potential class imbalance in both token and relation datasets.
* **Command-Line Inference**: Provides a simple command-line interface for running automatic causal extraction on new sentences.

## Example
> *Die Bundesregierung will das Waldsterben stoppen.*

| Token | BIO Label | Confidence |
| :--- | :--- | :--- |
| Die | O | 0.9727 |
| **Bundesregierung** | **B-ENTITY** | **0.6602** |
| will | O | 0.9492 |
| das | O | 0.9453 |
| **Waldsterben** | **B-ENTITY** | **0.6719** |
| **stoppen** | **B-INDICATOR** | **0.4980** |
| . | O | 0.9727 |

| Indicator | Entity | Relation Label | Confidence |
| :--- | :--- | :--- | :--- |
| **stoppen** | **Bundesregierung** | **MONO\_POS\_CAUSE** | **0.9922** |
| **stoppen** | **Waldsterben** | **MONO\_NEG\_EFFECT** | **0.9961** |

## Installation

To set up the environment, clone the repository and install the package using pip:

```bash
git clone [https://github.com/norygami/CausalBERT.git](https://github.com/norygami/CausalBERT.git)
cd CausalBERT
pip install .
````


## Project Structure
```
causalbert/
├── train.py                # Script for training the CausalBERT model
├── model.py                # Defines the CausalBERTMultiTaskModel and its configuration
├── infer.py                # Provides functions for inference and analysis
├── dataset.py              # Script for preparing the token and relation classification datasets
├── data/
│   ├── input/
│   │   └── json/
│   │       └── all_sentences.json # Raw input data for dataset creation (example location)
│   └── dataset/
│       ├── base/           # Processed datasets (without dependency parsing)
│       │   ├── token/
│       │   │   ├── train/
│       │   │   └── test/
│       │   └── relation/
│       │       ├── train/
│       │       └── test/
│   └── model/              # Directory to save trained models
│       └── C-EBERT-610m/ # Example saved model directory
│           ├── config.json
│           ├── model.safetensors
│           └── tokenizer.json
README.md
LICENSE
.gitignore
pyproject.toml
setup.py                # Setup script for package installation
MANIFEST.in
```

## Usage
### 1\. Data Preparation

Your raw data should be in a JSON file. Run `dataset.py` to process your raw data:

```bash
# Example for processing raw data:
python -m causalbert.dataset --input_json_path data/raw_input/all_sentences.json --base_dir ./data --model_name "EuroBERT/EuroBERT-610m"
````

### 2\. Training the Model
The `train.py` script orchestrates the training process using the Hugging Face `Trainer`.

```bash
# Example training command:
python -m causalbert.train --epochs 8 --model_name "EuroBERT/EuroBERT-610m"
````

*Note: Full arguments are detailed in [train.py](https://github.com/padjohn/cbert/blob/main/causalbert/train.py), which supports complex arguments via a training notebook/script.*

### 3\. Inference and Analysis (Command Line)
With `run_inference.py`. You can provide a local path to your model or a Hugging Face Hub ID.

```bash
# Run analysis on two sentences using a local model directory:
python run_inference.py \
    --model_path "pdjohn/C-EBERT-610m" \
    --sentences "Autoverkehr verursacht Bienensterben." "Lärm ist der Grund für Stress."
````

#### Low-Level API Usage

The core functionality remains available via the `causalbert.infer` module for custom scripts.

##### Comprehensive Sentence Analysis (Automatic Extraction)

Automatically extract indicators, entities, and classify all possible relations in a sentence.

```python
from causalbert.infer import load_model, sentence_analysis

# Load model from Hub or local directory
model, tokenizer, config, device = load_model("pdjohn/C-EBERT-610m")

sentence = "Der Krieg in der Ukraine verursachte hohe Verluste und führte zu einem Anstieg der Preise."
analysis = sentence_analysis(model, tokenizer, config, [sentence], device)[0]

print("Derived Relations:")
for (indicator, entity), rel_info in analysis["derived_relations"]:
    print(f"Indicator: '{indicator}', Entity: '{entity}', Relation: '{rel_info['label']}', Confidence: {rel_info['confidence']:.4f}")
```

## Model Architecture (`model.py`)
The `CausalBERTMultiTaskModel` is a custom `PreTrainedModel` built on top of a `Hugging Face AutoModel` (e.g., EuroBERT).

  - **`CausalBERTMultiTaskConfig`**: A custom configuration class inheriting from `PretrainedConfig`, used to store specific parameters for the multi-task model, such as the number of span and relation labels, base model name, and class weights for loss calculation.
  - **Base Model**: The model initializes a `transformers.AutoModel` (e.g., EuroBERT) as its base.
  - **Task-Specific Heads**: `token_classifier` and `relation_classifier` linear layers.
  - **Weighted CrossEntropyLoss**: The model dynamically applies class weights for the `CrossEntropyLoss` function in both token and relation classification tasks.
  - **Forward Pass**: The `forward` method directs the input through the base model and then through the appropriate classification head based on the `task` argument ("token" or "relation").

## Dataset Creation (`dataset.py`)
The `dataset.py` script is responsible for transforming raw sentence data into a format suitable for training.

  - **BIO Labeling**: Robustly assigns BIO tags to tokens for span recognition.
  - **Relation Data Formatting**: Prepares data for relation classification by combining indicator, entity, and sentence into a single input sequence, separated by the `<|parallel_sep|>` token.
  - **Dependency Parsing (Optional)**: Can integrate SpaCy to extract Universal Part-of-Speech (UPOS), morphological features (FEATS), and dependency relations (DEP) for each token, if the `--dep` flag is used.

## Training Process (`train.py`)

The `train.py` script handles the end-to-end training workflow:

  - **Trainer API**: Uses the Hugging Face `Trainer` to manage the training loop, including checkpoints and evaluation.
  - **Multi-Task Data Loaders**: Uses a custom `MultiTaskTrainer` and `GroupByTaskBatchSampler` to create batches that alternate between token and relation tasks.
  - **PEFT/LoRA Integration**: Supports Parameter-Efficient Fine-Tuning (PEFT/LoRA) for efficient training of large base models.
  - **Class Weight Calculation**: Crucially, it calculates and applies normalized inverse-frequency class weights for both token and relation classification tasks to mitigate class imbalance.
  - **Evaluation**: Performs evaluation on a single task (token classification) per epoch to ensure consistency for early stopping, while reporting final metrics for both tasks after training.

## ToDo
  - Comprehensive evaluation metrics (F1-score, precision, recall) for both tasks across the entire test set.
  - ``example_input.json``
  - Infer to ``output.json``

## Contributing
Contributions are welcome\! Please feel free to open issues or submit pull requests.

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](https://www.google.com/search?q=./LICENSE) file for more details.

**Important Note on Base Models:**
This project utilizes and fine-tunes pre-trained Transformer models from the Hugging Face Hub, specifically `EuroBERT/EuroBERT-610m`, which is also licensed under Apache License 2.0. While our code is permissively licensed, **the resulting fine-tuned model (weights and configurations) will inherit the license of the base model it was trained on.** Users are advised to check the specific license of any base model they choose to use with this code to ensure compliance with their intended use cases.
