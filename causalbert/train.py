import os
import torch
import wandb
import logging
import random
from evaluate import load
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.training_args import TrainingArguments
from datasets.load import load_from_disk
from causalbert.model import CausalBERTMultiTaskModel, MultiTaskCollator
from collections import Counter
from math import ceil
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

logger = logging.getLogger(__name__)

class GroupByTaskBatchSampler:
    """Yields homogeneous batches by 'task' to keep model.forward(task=...) simple."""
    def __init__(self, dataset, batch_size: int, seed: int = 42, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last

        def _task_of(idx: int) -> str:
            ex = dataset[idx]
            t = ex.get("task", None)
            if isinstance(t, list):
                t = t[0] if t else None
            if t is None:
                # Fallback inference from fields
                if ("labels_scalar" in ex) or ("relation" in ex) or ("indicator" in ex and "entity" in ex):
                    return "relation"
                return "token"
            return t

        # pre-index by task
        self.token_idx = [i for i in range(len(dataset)) if _task_of(i) == "token"]
        self.relat_idx = [i for i in range(len(dataset)) if _task_of(i) == "relation"]

    def __iter__(self):
        rng = random.Random(self.seed)
        rng.shuffle(self.token_idx)
        rng.shuffle(self.relat_idx)

        def chunks(lst, n):
            for i in range(0, len(lst), n):
                batch = lst[i:i+n]
                if len(batch) == n or not self.drop_last:
                    yield batch

        token_batches = list(chunks(self.token_idx, self.batch_size))
        relat_batches = list(chunks(self.relat_idx, self.batch_size))

        # simple round-robin to balance tasks
        i = j = 0
        while i < len(token_batches) or j < len(relat_batches):
            if i < len(token_batches):
                yield token_batches[i]
                i += 1
            if j < len(relat_batches):
                yield relat_batches[j]
                j += 1

    def __len__(self):
        t = len(self.token_idx)
        r = len(self.relat_idx)
        if self.drop_last:
            return (t // self.batch_size) + (r // self.batch_size)
        return ceil(t / self.batch_size) + ceil(r / self.batch_size)

class PeftSavingCallback(TrainerCallback):
    """
    A custom callback to save the PEFT adapter weights during training
    and prevent the Trainer from saving a full checkpoint to avoid redundancy.
    """
    def on_save(self, args, state, control, **kwargs):
        peft_model_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(peft_model_path)
        return control

class RelationTransform:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        tokenized_batches = []
        for i in range(len(examples["sentence"])):
            indicator = examples.get("indicator", [""])[i]
            entity = examples.get("entity", [""])[i]
            sentence = examples["sentence"][i]
            sep_token_str = "<|parallel_sep|>" 
            
            combined_input = f"{indicator} {sep_token_str} {entity} {sep_token_str} {sentence}"
            tokenized = self.tokenizer(combined_input, truncation=True, max_length=self.tokenizer.model_max_length)

            tokenized.update({
                "task": "relation",
                "labels_scalar": int(examples["relation"][i]),
                "labels_seq": [-100] * len(tokenized["input_ids"])
            })
            tokenized_batches.append(tokenized)

        combined_batch = {key: [d[key] for d in tokenized_batches] for key in tokenized_batches[0]}
        
        return combined_batch

def add_labels_seq(example):
    return {
        **example,
        "task": ["token"] * len(example["input_ids"]),
        "labels_seq": example.get("labels", [-100] * len(example["input_ids"]))
    }

def preprocess_relation(example, tokenizer):
    indicator = example.get("indicator", "")
    entity = example.get("entity", "")
    sentence = example["sentence"]
    sep_token_str = "<|parallel_sep|>" 
    
    combined_input = f"{indicator} {sep_token_str} {entity} {sep_token_str} {sentence}"
    tokenized = tokenizer(combined_input, truncation=True, max_length=tokenizer.model_max_length)

    tokenized.update({
        "task": "relation",
        "labels_scalar": int(example["relation"]),
        "labels_seq": [-100] * len(tokenized["input_ids"])
    })
    return tokenized

def _compute_ce_weights_from_counts(counts: Counter, num_classes: int, smoothing: float = 1.0,
                                    max_ratio: float = 10.0) -> list[float]:
    """
    Inverse-frequency class weights for CE: w_c ∝ 1 / (count_c + smoothing).
    Normalize to mean=1 and clip to avoid extremes.
    """
    totals = [counts.get(i, 0) + smoothing for i in range(num_classes)]
    inv = [1.0 / t for t in totals]
    weights = [w / sum(inv) for w in inv]
    lo, hi = 1.0 / max_ratio, max_ratio
    weights = [min(max(w, lo), hi) for w in weights]
    return weights


# Load the F1 metric with a multiclass averaging method
accuracy_metric = load("accuracy")
f1_metric = load("f1", average="macro")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Flatten the predictions and labels
    flat_predictions = predictions.flatten()
    flat_labels = labels.flatten()
    mask = flat_labels != -100
    filtered_predictions = flat_predictions[mask]
    filtered_labels = flat_labels[mask]

    if len(filtered_predictions) == 0:
      return {"accuracy": 0.0, "f1": 0.0}
      
    accuracy_result = accuracy_metric.compute(predictions=filtered_predictions, references=filtered_labels)
    f1_result = f1_metric.compute(predictions=filtered_predictions, references=filtered_labels, average="macro")
    
    return {**accuracy_result, **f1_result}

class MultiTaskTrainer(Trainer):
    def get_train_dataloader(self):
        if self.train_dataset is None:
            return None
        bs = self.args.per_device_train_batch_size
        sampler = GroupByTaskBatchSampler(self.train_dataset, batch_size=bs, seed=self.args.seed, drop_last=self.args.dataloader_drop_last)
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset or self.eval_dataset
        if eval_dataset is None:
            return None
        bs = self.args.per_device_eval_batch_size
        sampler = GroupByTaskBatchSampler(eval_dataset, batch_size=bs, seed=self.args.seed, drop_last=False)
        return DataLoader(
            eval_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_test_dataloader(self, test_dataset):
        bs = self.args.per_device_eval_batch_size
        sampler = GroupByTaskBatchSampler(test_dataset, batch_size=bs, seed=self.args.seed, drop_last=False)
        return DataLoader(
            test_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

def train(
    train_dir=None,
    test_dir=None,
    out_dir="",
    model_name="EuroBERT/EuroBERT-210m",
    model_save_name="C-EBERT",
    device=None,
    training_args_overrides: dict | None = None,
    peft_config_overrides: dict | None = None,
    callbacks: list | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    compute_dtype = (
        torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported())
        else (torch.float16 if device == "cuda" else torch.float32)
    )

    # tokenizer + special tokens
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True, trust_remote_code=True)
    tokenizer.model_max_length = 512
    if "<|parallel_sep|>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|parallel_sep|>"]})
        logging.info("Added '<|parallel_sep|>' to tokenizer vocabulary.")

    # load datasets (train/test multitask)
    if train_dir == None or not os.path.exists(train_dir):
        print("Please provide a valid train_dir path.")
        return
    mt_train = load_from_disk(train_dir)
    if test_dir and os.path.exists(test_dir):
        mt_test  = load_from_disk(test_dir)
    logging.info(f"Loaded multitask train: total={len(mt_train)}; test={len(mt_test) if mt_test else 0}")

    # class weights
    collator = MultiTaskCollator(tokenizer)
    span_label_map = {"O": 0, "B-INDICATOR": 1, "I-INDICATOR": 2, "B-ENTITY": 3, "I-ENTITY": 4}
    relation_label_map = {
        "NO_RELATION": 0,
        "MONO_POS_CAUSE": 1,
        "DIST_POS_CAUSE": 2,
        "PRIO_POS_CAUSE": 3,
        "MONO_NEG_CAUSE": 4,
        "DIST_NEG_CAUSE": 5,
        "PRIO_NEG_CAUSE": 6,
        "MONO_POS_EFFECT": 7,
        "DIST_POS_EFFECT": 8,
        "PRIO_POS_EFFECT": 9,
        "MONO_NEG_EFFECT": 10,
        "DIST_NEG_EFFECT": 11,
        "PRIO_NEG_EFFECT": 12,
        "INTERDEPENDENCY": 13,
    }
    id2span_label = {v: k for k, v in span_label_map.items()}
    id2relation_label = {v: k for k, v in relation_label_map.items()}

    span_counts = Counter()
    for ex in mt_train.filter(lambda x: x["task"] == "token"):
        enc = tokenizer(ex["sentence"], return_offsets_mapping=True, return_tensors=None)
        labels = collator._labels_from_spans(enc.encodings[0], ex["spans"])
        for lid in labels:
            if lid != -100:
                span_counts[int(lid)] += 1
    relation_counts = Counter(int(ex["relation"]) for ex in mt_train.filter(lambda x: x["task"] == "relation"))

    total_span_tokens = sum(span_counts.values())
    span_majority_class_count = span_counts.most_common(1)[0][1]
    span_accuracy_baseline = span_majority_class_count / total_span_tokens
    logging.info(f"Token Accuracy Baseline (majority class): {span_accuracy_baseline:.4f}")

    # For relations, the logic is the same
    total_relations = sum(relation_counts.values())
    relation_majority_class_count = relation_counts.most_common(1)[0][1]
    relation_accuracy_baseline = relation_majority_class_count / total_relations
    logging.info(f"Relation Accuracy Baseline (majority class): {relation_accuracy_baseline:.4f}")

    span_class_weights_list = _compute_ce_weights_from_counts(span_counts, num_classes=len(span_label_map))
    relation_class_weights_list = _compute_ce_weights_from_counts(relation_counts, num_classes=len(relation_label_map))

    # build config
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.num_span_labels = len(span_label_map)
    config.num_relation_labels = len(relation_label_map)
    config.id2label_span = {str(k): v for k, v in id2span_label.items()}
    config.id2label_relation = {str(k): v for k, v in id2relation_label.items()}
    config.architectures = ["CausalBERTMultiTaskModel"]
    config.torch_dtype = str(compute_dtype).replace("torch.", "")
    config.vocab_size = len(tokenizer)
    config.relation_class_weights = relation_class_weights_list
    config.span_class_weights = span_class_weights_list
    config.model_type = "causalbert_multitask"
    config.base_model_name = model_name

    model = CausalBERTMultiTaskModel(config)

    training_args = TrainingArguments(
        output_dir=os.path.join(out_dir, f"{model_save_name}"),
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=3e-5,
        logging_steps=50,
        logging_strategy="epoch",
        logging_first_step=True,
        save_steps=500,
        eval_strategy="no" if mt_test is None else "epoch",
        report_to=[],
        seed=42,
        remove_unused_columns=False,
        bf16=(device == "cuda" and torch.cuda.is_bf16_supported()),
        fp16=(device == "cuda" and not torch.cuda.is_bf16_supported()),
        group_by_length=False,
        dataloader_drop_last=False,
        dataloader_num_workers=0,
        save_only_model=True
    )

    if training_args_overrides:
        for k, v in training_args_overrides.items():
            if hasattr(training_args, k):
                setattr(training_args, k, v)
            else:
                logging.warning(f"Ignoring unknown TrainingArguments field: {k}")
    default_peft_config = {
        "task_type": TaskType.TOKEN_CLS,
        "inference_mode": False,
        "r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj"],
    }
    if peft_config_overrides:
        logging.info("Overriding default PEFT config with user-provided values.")
        default_peft_config.update(peft_config_overrides)    
    peft_config = LoraConfig(**default_peft_config)

    all_callbacks = []
    if callbacks:
        all_callbacks.extend(callbacks)
    
    model = get_peft_model(model, peft_config)
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=mt_train,
        eval_dataset=mt_test if mt_test is not None else None,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=all_callbacks,
    )

    trainer.train()
    model.to("cpu")
    merged_model = model.merge_and_unload()

    # Save
    save_dir = training_args.output_dir
    merged_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logging.info(f"Model saved to {save_dir}")