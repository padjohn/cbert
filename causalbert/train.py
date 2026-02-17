"""
Training pipeline: multi-task trainer, metrics, and LoRA fine-tuning.
"""

import os
import torch
import logging
import random
import json
from math import ceil
from typing import Tuple, cast, Dict, Any
from collections import Counter
import numpy as np
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.training_args import TrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizer
from datasets.load import load_from_disk
from evaluate import load
from peft import LoraConfig, TaskType, get_peft_model
from causalbert.utils import get_compute_dtype, compute_ce_weights_from_counts
from causalbert.model import (
    CausalBERTMultiTaskModel, 
    MultiTaskCollator,
    ROLE2ID,
    ID2ROLE,
    POLARITY2ID,
    ID2POLARITY,
    SPAN_LABEL_MAP,
    RELATION_MAP_V2,
    V2_TO_V3_MAPPING,
    SALIENCE_VALUES,
    influence_to_components,
    convert_v3_to_v2_label
)

logger = logging.getLogger(__name__)


class GroupByTaskBatchSampler:
    """Yields homogeneous batches by 'task' to keep model.forward(task=...) simple."""
    
    def __init__(self, dataset, batch_size: int, seed: int = 42, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self._iter_count = 0
        

        def _task_of(idx: int) -> str:
            ex = dataset[idx]
            t = ex.get("task", None)
            if isinstance(t, list):
                t = t[0] if t else None
            if t is None:
                if "role" in ex or "relation" in ex or "indicator" in ex:
                    return "relation"
                return "token"
            return t

        self.token_idx = [i for i in range(len(dataset)) if _task_of(i) == "token"]
        self.relation_idx = [i for i in range(len(dataset)) if _task_of(i) == "relation"]

    def __iter__(self):
        rng = random.Random(self.seed + self._iter_count)
        self._iter_count += 1
        rng.shuffle(self.token_idx)
        rng.shuffle(self.relation_idx)

        def chunks(lst, n):
            for i in range(0, len(lst), n):
                batch = lst[i:i+n]
                if len(batch) == n or not self.drop_last:
                    yield batch

        token_batches = list(chunks(self.token_idx, self.batch_size))
        relation_batches = list(chunks(self.relation_idx, self.batch_size))

        i = j = 0
        while i < len(token_batches) or j < len(relation_batches):
            if i < len(token_batches):
                yield token_batches[i]
                i += 1
            if j < len(relation_batches):
                yield relation_batches[j]
                j += 1

    def __len__(self):
        t = len(self.token_idx)
        r = len(self.relation_idx)
        if self.drop_last:
            return (t // self.batch_size) + (r // self.batch_size)
        return ceil(t / self.batch_size) + ceil(r / self.batch_size)


class PeftSavingCallback(TrainerCallback):
    """Save PEFT adapter weights during training."""
    def on_save(self, args, state, control, **kwargs):
        if args.output_dir is None:
            return control
        peft_model_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(peft_model_path)
        return control

# Metrics
accuracy_metric = load("accuracy")
f1_metric = load("f1")


def _compute_flat_metrics(eval_pred, prefix=""):
    """Shared accuracy/f1 computation for any flat classification head."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    flat_predictions = predictions.flatten()
    flat_labels = labels.flatten()
    mask = flat_labels != -100
    filtered_predictions = flat_predictions[mask]
    filtered_labels = flat_labels[mask]

    if len(filtered_predictions) == 0:
        return {f"{prefix}accuracy": 0.0, f"{prefix}f1": 0.0}

    accuracy_result = accuracy_metric.compute(
        predictions=filtered_predictions,
        references=filtered_labels
    )
    f1_result = f1_metric.compute(
        predictions=filtered_predictions,
        references=filtered_labels,
        average="macro"
    )

    return {
        f"{prefix}accuracy": accuracy_result["accuracy"],
        f"{prefix}f1": f1_result["f1"],
    }


def compute_token_metrics(eval_pred):
    """Token classification metrics (no prefix)."""
    return _compute_flat_metrics(eval_pred, prefix="")


def compute_metrics_v2(eval_pred):
    """V2 relation classification metrics + decomposed subtask metrics."""
    # Original 14-class metrics
    base = _compute_flat_metrics(eval_pred, prefix="rel_")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1).flatten()
    flat_labels = labels.flatten()
    mask = flat_labels != -100
    preds_filtered = predictions[mask]
    labels_filtered = flat_labels[mask]
    
    if len(preds_filtered) == 0:
        return base
    
    # Decompose V2 predictions into role, polarity, salience
    id2v2 = {v: k for k, v in RELATION_MAP_V2.items()}
    
    role_preds, role_labels_list = [], []
    pol_preds, pol_labels_list = [], []
    sal_preds, sal_labels_list = [], []
    
    for pred_id, true_id in zip(preds_filtered, labels_filtered):
        pred_name = id2v2[int(pred_id)]
        true_name = id2v2[int(true_id)]
        
        pred_role, pred_inf = V2_TO_V3_MAPPING[pred_name]
        true_role, true_inf = V2_TO_V3_MAPPING[true_name]
        
        role_preds.append(ROLE2ID[pred_role])
        role_labels_list.append(ROLE2ID[true_role])
        
        # Polarity and salience only meaningful for non-NO_RELATION
        if true_role != "NO_RELATION":
            pred_pol, pred_sal = influence_to_components(pred_inf)
            true_pol, true_sal = influence_to_components(true_inf)
            pol_preds.append(POLARITY2ID[pred_pol])
            pol_labels_list.append(POLARITY2ID[true_pol])
            sal_preds.append(pred_sal)
            sal_labels_list.append(true_sal)
    
    # Role accuracy (comparable to V3's role_accuracy)
    role_preds_arr = np.array(role_preds)
    role_labels_arr = np.array(role_labels_list)
    base["role_accuracy"] = float((role_preds_arr == role_labels_arr).mean())
    base["role_f1_macro"] = (f1_metric.compute(
        predictions=role_preds_arr,
        references=role_labels_arr,
        average="macro"
    ) or {})["f1"]
    
    # Polarity accuracy (on non-NO_RELATION samples)
    if pol_preds:
        pol_preds_arr = np.array(pol_preds)
        pol_labels_arr = np.array(pol_labels_list)
        base["polarity_accuracy"] = float((pol_preds_arr == pol_labels_arr).mean())
    
    # Salience accuracy (discretized, on non-NO_RELATION samples)
    if sal_preds:
        sal_preds_arr = np.array(sal_preds)
        sal_labels_arr = np.array(sal_labels_list)
        base["salience_accuracy"] = float((sal_preds_arr == sal_labels_arr).mean())
    
    return base


def compute_metrics_v3(eval_pred):
    """
    Compute metrics for v3 (factorized role + polarity + salience).

    eval_pred contains:
        predictions: tuple of (role_logits, polarity_logits, salience_logits, influence)
        labels: tuple of (role_labels, polarity_labels, salience_labels, influence_labels)

    Reported metrics:
        role_accuracy / role_f1:      Role classification (role_f1 used for early stopping)
        polarity_accuracy:            POS/NEG accuracy on non-NO_RELATION samples
        salience_accuracy:            3-class salience accuracy on non-NO_RELATION samples
        reconstructed_v2_accuracy:    How well role+polarity+salience maps back to v2 label
    """
    predictions, labels = eval_pred
    
    if isinstance(predictions, tuple) and len(predictions) == 4:
        role_logits, polarity_logits, salience_preds, _ = predictions
        role_labels, polarity_labels, salience_labels, influence_labels = labels
    else:
        return {"accuracy": 0.0, "f1": 0.0}
    
    # Role metrics
    role_preds = np.argmax(role_logits, axis=-1).flatten()
    role_labels_flat = role_labels.flatten()
    
    mask = role_labels_flat != -100
    filtered_role_preds = role_preds[mask]
    filtered_role_labels = role_labels_flat[mask]
    
    if len(filtered_role_preds) == 0:
        return {"accuracy": 0.0, "f1": 0.0}
    
    role_accuracy = (accuracy_metric.compute(
        predictions=filtered_role_preds,
        references=filtered_role_labels
    ) or {})["accuracy"]

    role_f1_macro = (f1_metric.compute(
        predictions=filtered_role_preds,
        references=filtered_role_labels,
        average="macro"
    ) or {})["f1"]

    role_f1_weighted = (f1_metric.compute(
        predictions=filtered_role_preds,
        references=filtered_role_labels,
        average="weighted"
    ) or {})["f1"]
    
    metrics = {
        "role_accuracy": role_accuracy,
        "role_f1": role_f1_weighted,
        "role_f1_macro": role_f1_macro,
    }
    
    # All remaining metrics only apply to non-NO_RELATION samples
    inf_mask = filtered_role_labels != ROLE2ID["NO_RELATION"]
    
    if inf_mask.sum() > 0 and polarity_labels is not None:
        pol_preds = np.argmax(polarity_logits, axis=-1).flatten()
        pol_preds_masked = pol_preds[mask][inf_mask]
        pol_labels_masked = polarity_labels.flatten()[mask][inf_mask]
        metrics["polarity_accuracy"] = float((pol_preds_masked == pol_labels_masked).mean())
    
    if inf_mask.sum() > 0 and salience_labels is not None:
        sal_preds = np.argmax(salience_preds, axis=-1).flatten()
        sal_preds_masked = sal_preds[mask][inf_mask]
        sal_labels_masked = salience_labels.flatten()[mask][inf_mask]
        metrics["salience_accuracy"] = float((sal_preds_masked == sal_labels_masked).mean())
    
    if inf_mask.sum() > 0 and influence_labels is not None:
        # Reconstructed v2 accuracy
        reconstructed_v2_preds = []
        reconstructed_v2_labels = []
        
        pol_preds_all = np.argmax(polarity_logits, axis=-1).flatten()[mask]
        sal_preds_all = np.argmax(salience_preds, axis=-1).flatten()[mask]
        
        for i in range(len(filtered_role_preds)):
            if inf_mask[i]:
                role_name = ID2ROLE[filtered_role_preds[i]]
                pol_name = ID2POLARITY[pol_preds_all[i]]
                sal_val = SALIENCE_VALUES[int(sal_preds_all[i])]
                v2_pred = convert_v3_to_v2_label(role_name, pol_name, sal_val)
                reconstructed_v2_preds.append(v2_pred)
                
                role_name_true = ID2ROLE[filtered_role_labels[i]]
                influence = float(influence_labels.flatten()[mask][i])
                polarity, salience = influence_to_components(influence)
                v2_true = convert_v3_to_v2_label(role_name_true, polarity, salience)
                reconstructed_v2_labels.append(v2_true)
        
        if reconstructed_v2_preds:
            v2_match = sum(p == t for p, t in zip(reconstructed_v2_preds, reconstructed_v2_labels))
            metrics["reconstructed_v2_accuracy"] = v2_match / len(reconstructed_v2_preds)
    
    return metrics

class MultiTaskMetricsSaverCallback(TrainerCallback):
    """Saves a multitask_metrics.json summary to each checkpoint directory.

    Captures both relation and token evaluation metrics from the most recent
    evaluation step and writes them alongside the checkpoint. Works for both
    v2 (14-class) and v3 (factorized) architectures, auto-detecting which
    metrics are available.
    """
    def __init__(self):
        self.last_metrics = {}

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self.last_metrics = metrics

    def on_save(self, args, state, control, **kwargs):
        if not self.last_metrics:
            return

        checkpoint_folder = f"checkpoint-{state.global_step}"
        output_dir = os.path.join(args.output_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine if we are in V2 or V3 based on available keys
        rel_acc = self.last_metrics.get("eval_rel_accuracy") or self.last_metrics.get("eval_role_accuracy")
        v2_final_acc = self.last_metrics.get("eval_rel_accuracy") or self.last_metrics.get("eval_reconstructed_v2_accuracy")

        summary = {
            "epoch": state.epoch,
            "step": state.global_step,
            "relation_loss": self.last_metrics.get("eval_loss"),
            "token_loss": self.last_metrics.get("eval_tok_loss"),
            "primary_rel_accuracy": rel_acc,
            "token_accuracy": self.last_metrics.get("eval_tok_accuracy"),
            "final_v2_accuracy": v2_final_acc
        }

        for key in ["eval_role_accuracy", "eval_role_f1_macro", 
                     "eval_polarity_accuracy", "eval_salience_accuracy"]:
            if key in self.last_metrics:
                summary[key.replace("eval_", "")] = self.last_metrics[key]
        
        with open(os.path.join(output_dir, "multitask_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

class MultiTaskTrainer(Trainer):
    """Trainer with task-grouped batch sampling and combined token+relation metrics."""
    
    def __init__(self, *args, token_eval_dataset=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_eval_dataset = token_eval_dataset
        
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Run evaluation on both relation and token classification tasks.

        Extends the standard Trainer.evaluate() by appending a separate token
        classification evaluation pass. Relation metrics are computed by the
        parent class via compute_metrics; token metrics (accuracy, F1) are
        computed manually and added under the ``{prefix}_tok_`` namespace.

        This two-pass approach is necessary because the relation and token
        tasks have different label structures and cannot share a single
        compute_metrics function.
        """
        # Relation evaluation (logs to table normally)
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        
        # Token evaluation — run manually to avoid any Trainer side effects
        if self.token_eval_dataset is not None:
            tok_dataloader = self.get_eval_dataloader(self.token_eval_dataset)
            all_preds = []
            all_labels = []
            total_loss = 0.0
            total_samples = 0
            
            self.model.eval()
            for batch in tok_dataloader:
                batch = self._prepare_inputs(batch)
                with torch.no_grad():
                    outputs = self.model(**batch)
                
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]
                labels = batch["labels"]
                
                total_loss += loss.item() * labels.shape[0]
                total_samples += labels.shape[0]
                all_preds.append(logits.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())
            
            # Pad to common length before concatenating (sequences vary in length)
            max_len = max(p.shape[1] for p in all_preds)
            padded_preds = [
                np.pad(p, ((0,0), (0, max_len - p.shape[1]), (0,0)), constant_values=0)
                for p in all_preds
            ]
            padded_labels = [
                np.pad(l, ((0,0), (0, max_len - l.shape[1])), constant_values=-100)
                for l in all_labels
            ]
            
            all_preds = np.concatenate(padded_preds)
            all_labels = np.concatenate(padded_labels)
            tok_metrics = compute_token_metrics((all_preds, all_labels))
            
            prefix = f"{metric_key_prefix}_tok"
            metrics[f"{prefix}_loss"] = total_loss / max(total_samples, 1)
            for k, v in tok_metrics.items():
                metrics[f"{prefix}_{k}"] = v
        
        return metrics
    
    def get_train_dataloader(self):
        if self.train_dataset is None:
            return None
        bs = self.args.per_device_train_batch_size
        sampler = GroupByTaskBatchSampler(
            self.train_dataset, 
            batch_size=bs, 
            seed=self.args.seed, 
            drop_last=self.args.dataloader_drop_last
        )
        return DataLoader(
            cast(TorchDataset[Any], self.train_dataset),
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
        sampler = GroupByTaskBatchSampler(
            eval_dataset, 
            batch_size=bs, 
            seed=self.args.seed, 
            drop_last=False
        )
        return DataLoader(
            cast(TorchDataset[Any], eval_dataset),
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_test_dataloader(self, test_dataset):
        bs = self.args.per_device_eval_batch_size
        sampler = GroupByTaskBatchSampler(
            test_dataset, 
            batch_size=bs, 
            seed=self.args.seed, 
            drop_last=False
        )
        return DataLoader(
            test_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Custom prediction step that returns role, polarity, salience, and influence."""
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            with self.compute_loss_context_manager():
                outputs = model(**inputs)
        
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        if prediction_loss_only:
            return (loss, None, None)
        
        if isinstance(outputs, dict) and "polarity_logits" in outputs:
            # v3: return all four outputs
            logits = (
                outputs["role_logits"].detach(),
                outputs["polarity_logits"].detach(),
                outputs["salience"].detach(),
                outputs["influence"].detach(),
            )
        elif isinstance(outputs, dict) and "role_logits" in outputs:
            # v3 fallback (shouldn't happen with current model)
            logits = (outputs["role_logits"].detach(), outputs["influence"].detach())
        else:
            # v2 or token classification
            logits = outputs["logits"].detach() if "logits" in outputs else outputs[1].detach()
        
        # Get labels
        if "polarity_labels" in inputs:
            labels = (
                inputs["role_labels"].detach(),
                inputs["polarity_labels"].detach(),
                inputs["salience_labels"].detach(),
                inputs["influence_labels"].detach(),
            )
        elif "role_labels" in inputs:
            labels = (inputs["role_labels"].detach(),
                      inputs.get("influence_labels", torch.zeros(1)).detach())
        elif "labels" in inputs:
            labels = inputs["labels"].detach()
        else:
            labels = None
        
        return (loss, logits, labels)


def train(
    train_dir: str | None = None,
    test_dir: str | None = None,
    out_dir: str = "",
    model_name: str = "EuroBERT/EuroBERT-210m",
    model_save_name: str = "C-EBERT",
    device: str | None = None,
    architecture_version: int = 2,
    training_args_overrides: dict | None = None,
    peft_config_overrides: dict | None = None,
    span_class_mod: list = [1, 1, 1, 1, 1, 1, 1, 1, 1],
    role_class_mod: list = [1, 1, 1],
    polarity_loss_weight: float = 1.0,
    salience_loss_weight: float = 1.0,
    relation_loss_weight: float = 1.0,
    callbacks: list | None = None,
) -> Tuple[CausalBERTMultiTaskModel, PreTrainedTokenizer, dict]:
    """
    Train C-BERT model.
    
    Args:
        train_dir: Path to training dataset
        test_dir: Path to test dataset
        out_dir: Output directory for model
        model_name: Base transformer model
        model_save_name: Name for saved model
        device: Training device (cuda/cpu)
        architecture_version: 2 for 14-class, 3 for factorized role+polarity+salience
        training_args_overrides: Override default TrainingArguments
        peft_config_overrides: Override default LoRA config
        span_class_mod: Multipliers for span class weights, ordered by SPAN_LABEL_MAP:
            [O, B-INDICATOR, I-INDICATOR, B-ENTITY, I-ENTITY].
            Applied on top of inverse-frequency weights. Use to boost B-tags, e.g.
            [1, 5, 1, 5, 1] to 5x-boost B-INDICATOR and B-ENTITY.
            Default None = no modification (all 1s).
        role_class_mod: Multipliers for [CAUSE, EFFECT, NO_RELATION] class weights
        polarity_loss_weight: Weight for polarity classification loss (v3)
        salience_loss_weight: Weight for salience classification loss (v3)
        relation_loss_weight: Scalar multiplier for V2 relation loss (default 1.0).
            Set to ~3.0 to match V3's gradient budget for ablation studies.
        callbacks: Additional trainer callbacks
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Architecture version: {architecture_version}")
    
    compute_dtype, device = get_compute_dtype(device)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True, trust_remote_code=True)
    tokenizer.model_max_length = 512
    if "<|parallel_sep|>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|parallel_sep|>"]})
        logging.info("Added '<|parallel_sep|>' to tokenizer.")

    # Load datasets
    if train_dir is None or not os.path.exists(train_dir):
        raise FileNotFoundError(f"Invalid train_dir path: {train_dir}. Cannot proceed with loading.")

        
    mt_train = load_from_disk(train_dir)
    mt_test = load_from_disk(test_dir) if test_dir and os.path.exists(test_dir) else None
    
    logging.info(f"Loaded train: {len(mt_train)}, test: {len(mt_test) if mt_test else 0}")

    # Split by task
    mt_test_token = mt_test.filter(lambda x: x["task"] == "token") if mt_test else None
    mt_test_relation = mt_test.filter(lambda x: x["task"] == "relation") if mt_test else None

    # Collator
    collator = MultiTaskCollator(tokenizer, architecture_version=architecture_version)
    
    # Label maps
    span_label_map = SPAN_LABEL_MAP
    id2span_label = {v: k for k, v in span_label_map.items()}
    id2relation_label = {v: k for k, v in RELATION_MAP_V2.items()}

    # Compute class weights
    span_counts = Counter()
    for ex_raw in mt_train.filter(lambda x: x["task"] == "token"):
        ex = cast(Dict[str, Any], ex_raw)
        enc = tokenizer(ex["sentence"], return_offsets_mapping=True, return_tensors=None)
        labels = collator._labels_from_spans(enc.encodings[0], ex["spans"])
        
        for lid in labels:
            if lid != -100:
                span_counts[int(lid)] += 1
    
    span_class_weights = compute_ce_weights_from_counts(span_counts, num_classes=len(span_label_map))
    
    # Apply span class weight modifiers (e.g. to boost B-tags)
    if span_class_mod is not None:
        assert len(span_class_mod) == len(span_label_map), \
            f"span_class_mod must have {len(span_label_map)} elements " \
            f"(O, B-INDICATOR, I-INDICATOR, B-ENTITY, I-ENTITY), got {len(span_class_mod)}"
        span_class_weights = [w * m for w, m in zip(span_class_weights, span_class_mod)]
        logging.info(f"Span class weights after modification: {span_class_weights}")
        logging.info(f"  Modifier applied: {span_class_mod}")
    
    logging.info(f"Span class weights: " + ", ".join(
        f"{id2span_label[i]}={span_class_weights[i]:.3f}" for i in range(len(span_class_weights))
    ))

    # Log baselines
    total_span_tokens = sum(span_counts.values())
    if total_span_tokens > 0:
        span_majority = span_counts.most_common(1)[0][1]
        logging.info(f"Token accuracy baseline: {span_majority / total_span_tokens:.4f}")

    # Build config based on architecture version
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.architecture_version = architecture_version
    config.num_span_labels = len(span_label_map)
    config.id2label_span = {str(k): v for k, v in id2span_label.items()}
    config.span_class_weights = span_class_weights
    config.architectures = ["CausalBERTMultiTaskModel"]
    config.torch_dtype = str(compute_dtype).replace("torch.", "")
    config.vocab_size = len(tokenizer)
    config.model_type = "causalbert_multitask"
    config.base_model_name = model_name

    if architecture_version == 2:
        # v2: 14-class relation classification
        relation_counts = Counter(
            int(cast(dict[str, Any], ex)["relation"]) 
            for ex in mt_train.filter(lambda x: x["task"] == "relation")
        )
        relation_class_weights = compute_ce_weights_from_counts(
            relation_counts, num_classes=len(RELATION_MAP_V2)
        )
        
        config.num_relation_labels = len(RELATION_MAP_V2)
        config.id2label_relation = {str(k): v for k, v in id2relation_label.items()}
        config.relation_class_weights = relation_class_weights
        config.relation_loss_weight = relation_loss_weight

        if relation_loss_weight != 1.0:
            logging.info(f"V2 relation loss weight: {relation_loss_weight:.2f}")
        
        total_relations = sum(relation_counts.values())
        if total_relations > 0:
            rel_majority = relation_counts.most_common(1)[0][1]
            logging.info(f"Relation accuracy baseline (v2): {rel_majority / total_relations:.4f}")
        
        compute_metrics = compute_metrics_v2
        
    else:
        # v3: Factorized role + polarity + salience
        role_counts = Counter(
            int(cast(dict[str, Any], ex)["role"]) 
            for ex in mt_train.filter(lambda x: x["task"] == "relation")
        )
        effect_count = role_counts.get(ROLE2ID["EFFECT"], 0)
        cause_count = role_counts.get(ROLE2ID["CAUSE"], 0)
        no_rel_count = role_counts.get(ROLE2ID["NO_RELATION"], 0)

        # Set EFFECT as baseline (1.0), scale others relative to it
        role_class_weights = [
            role_class_mod[0] * effect_count / cause_count,
            role_class_mod[1] * 1.0,
            role_class_mod[2] * (effect_count / max(no_rel_count, 1))
        ]

        # Optionally cap NO_RELATION
        role_class_weights[2] = min(role_class_weights[2], 50.0)

        config.role_class_weights = role_class_weights
        logging.info(f"Role class weights: CAUSE={role_class_weights[0]:.3f}, "
                f"EFFECT={role_class_weights[1]:.3f}")
        logging.info(f"Role class distribution: CAUSE={cause_count}, EFFECT={effect_count}")
        
        config.num_role_labels = 3
        config.id2label_role = ID2ROLE
        
        total_roles = sum(role_counts.values())
        if total_roles > 0:
            role_majority = role_counts.most_common(1)[0][1]
            logging.info(f"Role accuracy baseline (v3): {role_majority / total_roles:.4f}")
        
        # Log influence distribution
        influences = [
            float(cast(Dict[str, Any], ex)["influence"]) 
            for ex in mt_train.filter(lambda x: x["task"] == "relation")
        ]
        logging.info(f"Influence stats: mean={np.mean(influences):.3f}, std={np.std(influences):.3f}")
        

        # Polarity class weights
        rel_data = mt_train.filter(lambda x: x["task"] == "relation")
        polarity_counts = Counter()
        for ex_raw in rel_data:
            ex = cast(Dict[str, Any], ex_raw)
            pol = ex.get("polarity")
            if pol is not None:
                polarity_counts[int(pol)] += 1
        pos_count = polarity_counts.get(POLARITY2ID["POS"], 0)
        neg_count = polarity_counts.get(POLARITY2ID["NEG"], 0)
        
        if pos_count > 0 and neg_count > 0:
            # Inverse frequency weighting
            polarity_class_weights = [
                neg_count / (pos_count + neg_count) * 2,  # POS weight
                pos_count / (pos_count + neg_count) * 2,  # NEG weight
            ]
        else:
            polarity_class_weights = [1.0, 1.0]
        
        config.polarity_class_weights = polarity_class_weights
        config.num_polarity_labels = 2
        config.id2label_polarity = ID2POLARITY
        config.polarity_loss_weight = polarity_loss_weight
        config.salience_loss_weight = salience_loss_weight
        
        logging.info(f"Polarity distribution: POS={pos_count}, NEG={neg_count}")
        logging.info(f"Polarity class weights: POS={polarity_class_weights[0]:.3f}, "
                    f"NEG={polarity_class_weights[1]:.3f}")
        
        # Salience distribution
        salience_counts = Counter()
        for ex_raw in rel_data:
            ex = cast(Dict[str, Any], ex_raw)
            sal = ex.get("salience")
            if sal is not None:
                salience_counts[int(sal)] += 1

        salience_class_weights = compute_ce_weights_from_counts(
            salience_counts, num_classes=3
        )
        config.salience_class_weights = salience_class_weights
        config.num_salience_labels = 3
        
        compute_metrics = compute_metrics_v3

    # Create model
    model = CausalBERTMultiTaskModel(config)

    best_model_metric = "eval_rel_f1" if architecture_version == 2 else "eval_role_f1"

    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(out_dir, model_save_name),
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=3e-5,
        logging_steps=50,
        logging_strategy="epoch",
        logging_first_step=True,
        save_strategy="epoch",
        eval_strategy="no" if mt_test is None else "epoch",
        report_to=[],
        seed=42,
        remove_unused_columns=False,
        bf16=(device == "cuda" and torch.cuda.is_bf16_supported()),
        fp16=(device == "cuda" and not torch.cuda.is_bf16_supported()),
        group_by_length=False,
        dataloader_drop_last=False,
        dataloader_num_workers=0,
        metric_for_best_model=best_model_metric,
        save_only_model=True,
        greater_is_better=True,
    )

    if training_args_overrides:
        for k, v in training_args_overrides.items():
            if hasattr(training_args, k):
                setattr(training_args, k, v)
            else:
                logging.warning(f"Ignoring unknown TrainingArguments field: {k}")

    # PEFT config
    default_peft_config = {
        "task_type": TaskType.TOKEN_CLS,
        "inference_mode": False,
        "r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj"],
    }
    if peft_config_overrides:
        default_peft_config.update(peft_config_overrides)
    peft_config = LoraConfig(**default_peft_config)

    # Apply PEFT
    model = get_peft_model(model, peft_config)
    
    # Trainer
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=mt_train,
        eval_dataset=mt_test_relation,
        token_eval_dataset=mt_test_token,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks or [],
    )

    # Train
    trainer.train()
    
    final_metrics = {}
    if mt_test_relation:
        final_metrics = trainer.evaluate(eval_dataset=cast(Any, mt_test_relation))
    
    # Save
    save_dir = training_args.output_dir
    if save_dir is None:
        raise ValueError("No output directory specified in training_args.")

    if final_metrics:
        metrics_path = os.path.join(save_dir, "eval_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, indent=2, ensure_ascii=False)
        logging.info(f"Final metrics saved to: {metrics_path}")

    if hasattr(model, "merge_and_unload"):
        merged_model = cast(Any, model).merge_and_unload()
    else:
        merged_model = cast(CausalBERTMultiTaskModel, model)
    
    merged_model.save_pretrained(save_dir) 
    tokenizer.save_pretrained(save_dir)
    config.save_pretrained(save_dir)
    
    logging.info(f"Model saved to: {save_dir}")
    
    return cast(CausalBERTMultiTaskModel, merged_model), tokenizer, final_metrics