import os
import json
import torch
import wandb
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, BertConfig
from datasets import load_from_disk
from .model import CausalBERTMultiTaskModel, CausalBERTMultiTaskConfig, MultiTaskCollator

def train(
    base_dir='../data',
    model_name="google-bert/bert-base-german-cased",
    model_save_name="CausalBERT_C",
    epochs=5,
    batch_size=8,
    lr=2e-5,
    device=None,
    use_wandb=True
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_save_path = os.path.join(base_dir, f"model/{model_save_name}")

    span_label_map = {"O": 0, "B-INDICATOR": 1, "I-INDICATOR": 2, "B-ENTITY": 3, "I-ENTITY": 4}
    relation_label_map = {"NO_RELATION": 0, "CAUSE": 1, "EFFECT": 2, "INTERDEPENDENCY": 3}
    id2span_label = {v: k for k, v in span_label_map.items()}
    id2relation_label = {v: k for k, v in relation_label_map.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

    def add_labels_seq(example):
    # Assume 'labels' exists and is the same length as 'input_ids'
        return {
            **example,
            "task": "token",
            "labels_seq": example.get("labels", [-100] * len(example["input_ids"]))
        }

    span_train = load_from_disk(os.path.join(base_dir, 'dataset/base/tokens/train')).map(add_labels_seq)

    def preprocess_relation(example):
        indicator = example.get("indicator", "")
        entity = example.get("entity", "")
        sentence = example["sentence"]

        combined_input = f"{indicator} [SEP] {entity} [SEP] {sentence}"

        tokenized = tokenizer(combined_input, truncation=True)

        tokenized.update({
            "task": "relation",
            "labels_scalar": int(relation_label_map.get(example.get("relation", "NO_RELATION"), 0)),
            "labels_seq": [-100] * len(tokenized["input_ids"])
        })

        return tokenized

    rel_train = load_from_disk(os.path.join(base_dir, 'dataset/base/relations/train')).map(preprocess_relation)

    collator = MultiTaskCollator(tokenizer)

    span_loader = DataLoader(span_train, batch_size=batch_size, shuffle=True, collate_fn=collator)
    rel_loader = DataLoader(rel_train, batch_size=batch_size, shuffle=True, collate_fn=collator)

    base = AutoModel.from_pretrained(model_name)
    config = CausalBERTMultiTaskConfig(
        span_labels=len(span_label_map),
        relation_labels=len(relation_label_map),
        _name_or_path=model_name
    )
    model = CausalBERTMultiTaskModel(config)
    model.bert = base
    model.to(device)

    if use_wandb:
        wandb.init(project="CausalBERT", entity="norygano")

    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch+1}/{epochs}")
        for loader in [span_loader, rel_loader]:
            task_type = loader.dataset[0]["task"]
            loop = tqdm(loader, desc=f"Training {task_type}")
            for batch in loop:
                optimizer.zero_grad()
                task = batch.pop("task")
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch, task=task)
                out["loss"].backward()
                optimizer.step()
                loop.set_postfix(loss=out["loss"].item())
                if use_wandb:
                    wandb.log({f"{task_type}_loss": out["loss"].item(), "epoch": epoch + 1})
        print(f"Finished Epoch {epoch+1}")

    if use_wandb:
        wandb.finish()

    # Save
    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    base_config = BertConfig.from_pretrained(model_name)
    base_config.update({
        "num_span_labels": len(span_label_map),
        "num_relation_labels": len(relation_label_map),
        "id2label_span": {str(k): v for k, v in id2span_label.items()},
        "id2label_relation": {str(k): v for k, v in id2relation_label.items()},
        "architectures": ["CausalBERTMultiTaskModel"],
    })
    base_config.save_pretrained(model_save_path)

    print(f"Model saved to {model_save_path}")
