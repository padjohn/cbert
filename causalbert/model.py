"""
Model architecture, configuration, loss functions, and label constants for C-BERT.
"""

import torch.nn as nn
import torch
import logging
from causalbert.utils import get_compute_dtype
from transformers import AutoModel
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

SPAN_LABEL_MAP = {
    "O": 0,
    "B-INDICATOR": 1, "I-INDICATOR": 2, "E-INDICATOR": 3, "S-INDICATOR": 4,
    "B-ENTITY": 5, "I-ENTITY": 6, "E-ENTITY": 7, "S-ENTITY": 8,
}

# Role labels (v3)
ROLE_LABELS = ["CAUSE", "EFFECT", "NO_RELATION"]
ROLE2ID = {label: i for i, label in enumerate(ROLE_LABELS)}
ID2ROLE = {i: label for i, label in enumerate(ROLE_LABELS)}

# Polarity labels (v3)
POLARITY_LABELS = ["POS", "NEG"]
POLARITY2ID = {label: i for i, label in enumerate(POLARITY_LABELS)}
ID2POLARITY = {i: label for i, label in enumerate(POLARITY_LABELS)}

# v2: 14-class labels
RELATION_MAP_V2 = {
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

V2_TO_V3_MAPPING = {
    "NO_RELATION":      ("NO_RELATION", 0.0),
    "MONO_POS_CAUSE":   ("CAUSE", 1.0),
    "DIST_POS_CAUSE":   ("CAUSE", 0.5),
    "PRIO_POS_CAUSE":   ("CAUSE", 0.75),
    "MONO_NEG_CAUSE":   ("CAUSE", -1.0),
    "DIST_NEG_CAUSE":   ("CAUSE", -0.5),
    "PRIO_NEG_CAUSE":   ("CAUSE", -0.75),
    "MONO_POS_EFFECT":  ("EFFECT", 1.0),
    "DIST_POS_EFFECT":  ("EFFECT", 0.5),
    "PRIO_POS_EFFECT":  ("EFFECT", 0.75),
    "MONO_NEG_EFFECT":  ("EFFECT", -1.0),
    "DIST_NEG_EFFECT":  ("EFFECT", -0.5),
    "PRIO_NEG_EFFECT":  ("EFFECT", -0.75),
    "INTERDEPENDENCY":  ("CAUSE", 1.0),
}

# Salience labels (v3)
SALIENCE_LABELS = ["DIST", "PRIO", "MONO"]
SALIENCE_VALUES = [0.5, 0.75, 1.0]
SALIENCE2ID = {label: i for i, label in enumerate(SALIENCE_LABELS)}
ID2SALIENCE = {i: label for i, label in enumerate(SALIENCE_LABELS)}

PARALLEL_SEP_TOKEN = "<|parallel_sep|>"

# =============================================================================
# Conversion helpers
# =============================================================================

def influence_to_components(influence: float) -> tuple[str, float]:
    """Split signed influence into polarity and absolute salience."""
    polarity = "POS" if influence >= 0 else "NEG"
    salience = abs(influence)
    return polarity, salience

def convert_v3_to_v2_label(role: str, polarity: str, salience: float) -> str:
    """Convert v3 (role, polarity, salience) components to v2 label string."""
    if role == "NO_RELATION":
        return "NO_RELATION"
    
    if salience < 0.625:
        sal_str = "DIST"
    elif salience < 0.875:
        sal_str = "PRIO"
    else:
        sal_str = "MONO"
    
    return f"{sal_str}_{polarity}_{role}"

def reconstruct_influence(polarity: str, salience: float) -> float:
    """Reconstruct scalar influence from polarity + salience.
    
    I = ±(Polarität) × |Salienz| — mirrors thesis formalization.
    """
    sign = 1.0 if polarity == "POS" else -1.0
    return sign * abs(salience)


# =============================================================================
# Config
# =============================================================================

class CausalBERTMultiTaskConfig(PretrainedConfig):
    """Configuration for CausalBERTMultiTaskModel.

    Supports two architecture versions:
        - v2: Unified 14-class relation classification
              (3 salience × 2 polarity × 2 role + NO_RELATION + INTERDEPENDENCY)
        - v3: Factorized relation heads (role + polarity + salience classification)

    Common attributes:
        architecture_version: 2 or 3.
        num_span_labels: Number of BIOES span tags (default 9).
        base_model_name: HuggingFace model ID for the transformer backbone.
        vocab_size: Tokenizer vocabulary size (after adding special tokens).
        torch_dtype: Compute dtype string ("bfloat16", "float16", or "float32").

    v2-specific attributes:
        num_relation_labels: Number of relation classes (default 14).
        relation_class_weights: Inverse-frequency weights for relation CE loss.
        relation_loss_weight: Scalar multiplier for the relation loss.

    v3-specific attributes:
        num_role_labels: Number of role classes (default 3: CAUSE, EFFECT, NO_RELATION).
        num_polarity_labels: Number of polarity classes (default 2: POS, NEG).
        num_salience_labels: Number of salience classes (default 3: DIST, PRIO, MONO).
        polarity_loss_weight: Scalar multiplier for the masked polarity loss.
        salience_loss_weight: Scalar multiplier for the masked salience loss.
    """
    model_type = "causalbert_multitask"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Architecture version: 2 = original 14-class, 3 = factorized role + polarity + salience
        self.architecture_version = kwargs.get("architecture_version", 2)
        
        # Span (token) classification config (unchanged across versions)
        self.num_span_labels = kwargs.get("num_span_labels", 5)
        self.id2label_span = kwargs.get("id2label_span", {})
        self.span_class_weights = kwargs.get("span_class_weights", None)
        
        # v2 relation config (14-class)
        self.num_relation_labels = kwargs.get("num_relation_labels", 14)
        self.id2label_relation = kwargs.get("id2label_relation", {})
        self.relation_class_weights = kwargs.get("relation_class_weights", None)
        self.relation_loss_weight = kwargs.get("relation_loss_weight", 1.0)
        
        # v3 relation config (factorized: role + polarity + salience)
        self.num_role_labels = kwargs.get("num_role_labels", 3)
        self.id2label_role = kwargs.get("id2label_role", ID2ROLE)
        self.role_class_weights = kwargs.get("role_class_weights", None)
        
        self.num_polarity_labels = kwargs.get("num_polarity_labels", 2)
        self.id2label_polarity = kwargs.get("id2label_polarity", ID2POLARITY)
        self.polarity_class_weights = kwargs.get("polarity_class_weights", None)
        self.polarity_loss_weight = kwargs.get("polarity_loss_weight", 1.0)
        self.salience_loss_weight = kwargs.get("salience_loss_weight", 1.0)

        self.num_salience_labels = kwargs.get("num_salience_labels", 3)
        self.salience_class_weights = kwargs.get("salience_class_weights", None)    
        
        # Common config
        self.torch_dtype = kwargs.get("torch_dtype", "float32")
        self.vocab_size = kwargs.get("vocab_size", 0)
        self.base_model_name = kwargs.get("base_model_name", None)


# =============================================================================
# Loss functions
# =============================================================================

class MaskedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss masked to ignore NO_RELATION samples.

    Used for polarity and salience heads, where predictions are only
    meaningful when the role head predicts CAUSE or EFFECT.

    Args:
        weight: Optional class weight tensor for the CE loss.

    Forward args:
        pred_logits: [B, num_classes] predicted logits.
        target: [B] integer class targets.
        target_role: [B] role label IDs; samples where
            target_role == ROLE2ID["NO_RELATION"] are masked out.
    """

    def __init__(self, weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, pred_logits, target, target_role):
        mask = (target_role != ROLE2ID["NO_RELATION"]).float()
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred_logits.device, dtype=pred_logits.dtype)
        ce_loss = self.ce(pred_logits, target)
        return (ce_loss * mask).sum() / (mask.sum() + 1e-8)


# =============================================================================
# Model
# =============================================================================

class CausalBERTMultiTaskModel(PreTrainedModel):
    """Multi-task transformer for causal span recognition and relation classification.

    Extends a pretrained transformer (e.g. EuroBERT) with two task heads:

    Task 1 — Span Recognition (task="token"):
        Linear token classifier producing BIOES tags for INDICATOR and ENTITY spans.

    Task 2 — Relation Classification (task="relation"):
        v2: Single linear head over [CLS] → 14-class relation label.
        v3: Three parallel heads over [CLS]:
            - Role (3-class: CAUSE, EFFECT, NO_RELATION)
            - Polarity (2-class: POS, NEG), masked for NO_RELATION
            - Salience (3-class: DIST, PRIO, MONO), masked for NO_RELATION

    Input for relation classification uses the format:
        [indicator] <|parallel_sep|> [entity] <|parallel_sep|> [sentence]

    See: Johnson (2026), "C-BERT: Factorized Causal Relation Extraction."
    """
    config_class = CausalBERTMultiTaskConfig

    def __init__(self, config):
        super().__init__(config)
        
        arch_version = getattr(config, "architecture_version", 2)
        logging.info(f"Initializing CausalBERTMultiTaskModel (architecture v{arch_version})")
        logging.info(f"  - Base Model Name: {config.base_model_name}")
        logging.info(f"  - Vocab Size: {config.vocab_size}")
        logging.info(f"  - Num Span Labels: {config.num_span_labels}")
        
        if arch_version == 2:
            logging.info(f"  - Num Relation Labels (v2): {config.num_relation_labels}")
        else:
            logging.info(f"  - Num Role Labels (v3): {config.num_role_labels}")
            logging.info(f"  - Polarity head: POS/NEG classification")
            logging.info(f"  - Salience head: MONO/DIST/PRIO classification")
        
        # Prefer bf16 > f16 > f32; respect explicit user override
        dtype_str = getattr(config, "torch_dtype", None)
        if dtype_str in ("bfloat16", "float16"):
            torch_dtype_val = getattr(torch, dtype_str)
        else:
            torch_dtype_val, _ = get_compute_dtype() 

        self.bert = AutoModel.from_pretrained(
            config.base_model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype_val
        )
        config.torch_dtype = str(torch_dtype_val).replace("torch.", "")
        logging.info(f"Loaded base model with dtype: {config.torch_dtype}")
        
        if len(self.bert.get_input_embeddings().weight) != config.vocab_size:
            self.bert.resize_token_embeddings(config.vocab_size)
            logging.info(f"Resized token embeddings to {config.vocab_size}.")

        hidden = self.bert.config.hidden_size
        
        # Token classifier (unchanged across versions)
        self.token_classifier = nn.Linear(hidden, config.num_span_labels)
        
        # Version-specific relation heads
        if arch_version == 2:
            self.relation_classifier = nn.Linear(hidden, config.num_relation_labels)
        else:  # v3: role + polarity + salience
            self.role_classifier = nn.Linear(hidden, config.num_role_labels)
            self.polarity_classifier = nn.Linear(hidden, config.num_polarity_labels)
            self.salience_classifier = nn.Linear(hidden, config.num_salience_labels)
        
        self.post_init()
        logging.info("Model heads initialized.")

        # Loss functions
        self.span_loss = nn.CrossEntropyLoss(
            weight=(torch.tensor(config.span_class_weights, dtype=torch.float)
                    if getattr(config, "span_class_weights", None) else None),
            ignore_index=-100
        )
        
        if arch_version == 2:
            self.relation_loss = nn.CrossEntropyLoss(
                weight=(torch.tensor(config.relation_class_weights, dtype=torch.float)
                        if getattr(config, "relation_class_weights", None) else None)
            )
        else:  # v3
            self.role_loss = nn.CrossEntropyLoss(
                weight=(torch.tensor(config.role_class_weights, dtype=torch.float)
                        if getattr(config, "role_class_weights", None) else None)
            )
            self.polarity_loss = MaskedCrossEntropyLoss(
                weight=(torch.tensor(config.polarity_class_weights, dtype=torch.float)
                        if getattr(config, "polarity_class_weights", None) else None)
            )
            self.salience_loss = MaskedCrossEntropyLoss(
                weight=(torch.tensor(config.salience_class_weights, dtype=torch.float)
                        if getattr(config, "salience_class_weights", None) else None)
            )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = getattr(self.config, "initializer_range", 0.02)
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids=None, 
                labels=None, role_labels=None,
                polarity_labels=None, salience_labels=None,
                influence_labels=None, # passed through to prediction_step for metric computation
                task=None, **kwargs):
        """
        Forward pass supporting v2 and v3 architectures.
        
        For v2 (task="relation"):
            - labels: [B] integer class labels (0-13)
            - Returns: {"loss": ..., "logits": [B, 1, 14]}
            
        For v3 (task="relation"):
            - role_labels: [B] integer role labels (0=CAUSE, 1=EFFECT, 2=NO_RELATION)
            - polarity_labels: [B] integer polarity labels (0=POS, 1=NEG)
            - salience_labels: [B] integer salience class IDs (0=DIST, 1=PRIO, 2=MONO)
            - Returns: {"loss": ..., "role_logits": [B, 3], "polarity_logits": [B, 2],
                        "salience": [B, 3] (logits), "influence": [B]}
        """
        bert_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
        }
        if token_type_ids is not None and getattr(self.bert.config, "type_vocab_size", 0) > 1:
            bert_inputs["token_type_ids"] = token_type_ids
            
        out = self.bert(**bert_inputs)
        
        if task == "token":
            return self._forward_token(out, labels)
        elif task == "relation":
            arch_version = getattr(self.config, "architecture_version", 2)
            if arch_version == 2:
                return self._forward_relation_v2(out, labels)
            else:
                return self._forward_relation_v3(out, role_labels, polarity_labels, salience_labels)
        else:
            raise ValueError(f"Task must be 'token' or 'relation', got {task}")

    def _forward_token(self, bert_out, labels):
        """Token classification forward pass."""
        token_logits = self.token_classifier(bert_out.last_hidden_state)
        loss = None
        if labels is not None:
            loss = self.span_loss(
                token_logits.view(-1, token_logits.size(-1)),
                labels.view(-1)
            )
        return {"loss": loss, "logits": token_logits}

    def _forward_relation_v2(self, bert_out, labels):
        """v2: 14-class relation classification."""
        cls_hidden = bert_out.last_hidden_state[:, 0]
        relation_logits = self.relation_classifier(cls_hidden)
        relation_logits = relation_logits.unsqueeze(1)
        
        loss = None
        if labels is not None:
            if labels.dim() < 2:
                labels = labels.unsqueeze(-1)
            loss = self.relation_loss(relation_logits.squeeze(1), labels.squeeze(-1))
            loss = loss * getattr(self.config, "relation_loss_weight", 1.0)
        return {"loss": loss, "logits": relation_logits}

    def _forward_relation_v3(self, bert_out, role_labels, polarity_labels, salience_labels):
        """v3: Factorized role + polarity + salience classification.

        Runs three parallel linear heads on the [CLS] representation and
        reconstructs a scalar influence value for backward compatibility.

        Returns:
            dict with keys:
                loss: Combined weighted loss (None if no labels provided).
                role_logits: [B, 3] logits over CAUSE / EFFECT / NO_RELATION.
                polarity_logits: [B, 2] logits over POS / NEG.
                salience: [B, 3] logits over DIST / PRIO / MONO.
                influence: [B] reconstructed scalar I = sign(polarity) × salience_value,
                    where salience_value ∈ {0.5, 0.75, 1.0}.
        """
        cls_hidden = bert_out.last_hidden_state[:, 0]
        
        role_logits = self.role_classifier(cls_hidden)
        polarity_logits = self.polarity_classifier(cls_hidden)
        salience_logits = self.salience_classifier(cls_hidden)
        
        # Reconstruct influence for backwards compatibility
        polarity_preds = torch.argmax(polarity_logits, dim=-1)
        polarity_sign = torch.where(polarity_preds == 0,
                                    torch.ones(polarity_preds.shape, device=polarity_preds.device),
                                    -torch.ones(polarity_preds.shape, device=polarity_preds.device))
        salience_preds = torch.argmax(salience_logits, dim=-1)
        # Map class IDs to values: DIST=0.5, PRIO=0.75, MONO=1.0
        sal_values = torch.tensor(SALIENCE_VALUES, device=salience_preds.device)
        salience_vals = sal_values[salience_preds]
        influence = polarity_sign * salience_vals
        
        loss = None
        if role_labels is not None:
            total_loss = self.role_loss(role_logits, role_labels)
            if polarity_labels is not None:
                pol_loss = self.polarity_loss(polarity_logits, polarity_labels, role_labels)
                total_loss = total_loss + self.config.polarity_loss_weight * pol_loss
            if salience_labels is not None:
                sal_loss = self.salience_loss(salience_logits, salience_labels, role_labels)
                total_loss = total_loss + self.config.salience_loss_weight * sal_loss
            loss = total_loss
        
        return {
            "loss": loss,
            "role_logits": role_logits,
            "polarity_logits": polarity_logits,
            "salience": salience_logits,
            "influence": influence,
        }

AutoModel.register(CausalBERTMultiTaskConfig, CausalBERTMultiTaskModel)


# =============================================================================
# Collator
# =============================================================================

class MultiTaskCollator:
    """Data collator supporting both v2 and v3 relation formats."""
    
    def __init__(self, tokenizer, architecture_version: int = 2):
        self.tokenizer = tokenizer
        self.architecture_version = architecture_version
        self.label2id = SPAN_LABEL_MAP

    def _labels_from_spans(self, enc, spans):
        """Convert character spans to BIOES token labels (all subwords)."""
        tokens = enc.tokens
        offsets = enc.offsets
        word_ids = enc.word_ids

        labels = [-100 if wid is None else self.label2id["O"] for wid in word_ids]

        spans_sorted = sorted(spans, key=lambda x: (x["start"], x["end"]))
        for span in spans_sorted:
            s, e = span["start"], span["end"]
            t = span["type"]

            # Collect ALL token indices that overlap this span
            hit_indices = []
            for i, wid in enumerate(word_ids):
                if wid is None:
                    continue
                ts, te = offsets[i]
                tok = tokens[i]
                if (tok.startswith("\u0120") or tok.startswith("\u2581")) and ts < te:
                    ts += 1
                if max(s, ts) < min(e, te):
                    if labels[i] != self.label2id["O"]:
                        continue
                    hit_indices.append(i)

            # Assign BIOES tags
            if len(hit_indices) == 1:
                labels[hit_indices[0]] = self.label2id[f"S-{t}"]
            elif len(hit_indices) > 1:
                labels[hit_indices[0]] = self.label2id[f"B-{t}"]
                for idx in hit_indices[1:-1]:
                    labels[idx] = self.label2id[f"I-{t}"]
                labels[hit_indices[-1]] = self.label2id[f"E-{t}"]

        return labels

    def __call__(self, features):
        f0 = features[0]
        task = f0.get("task")
        if task is None:
            task = "relation" if ("relation" in f0 or "role" in f0) else "token"
            
        if not all((f.get("task", task) == task) for f in features):
            raise ValueError("Mixed 'task' values in batch.")

        if task == "relation":
            return self._collate_relation(features)
        else:
            return self._collate_token(features)

    def _collate_relation(self, features):
        """Collate relation examples for v2 or v3."""
        sep = PARALLEL_SEP_TOKEN
        indicators = [f.get("indicator", "") for f in features]
        entities = [f.get("entity", "") for f in features]
        sentences = [f["sentence"] for f in features]
        combined = [f"{i} {sep} {e} {sep} {s}" for i, e, s in zip(indicators, entities, sentences)]
        
        toks = self.tokenizer(
            combined, padding=True, truncation=True,
            max_length=self.tokenizer.model_max_length, return_tensors="pt",
        )
        
        batch = {**toks, "task": "relation"}
        
        has_polarity = (features[0].get("polarity") is not None)
        has_role = (features[0].get("role") is not None)
        
        if has_polarity and self.architecture_version >= 3:
            batch["role_labels"] = torch.tensor([f["role"] for f in features], dtype=torch.long)
            batch["polarity_labels"] = torch.tensor([f["polarity"] for f in features], dtype=torch.long)
            batch["salience_labels"] = torch.tensor([f["salience"] for f in features], dtype=torch.long)
            batch["influence_labels"] = torch.tensor([f["influence"] for f in features], dtype=torch.float)
        elif has_role and self.architecture_version >= 3:
            # Fallback: dataset with role+influence but no polarity columns
            batch["role_labels"] = torch.tensor([f["role"] for f in features], dtype=torch.long)
            batch["influence_labels"] = torch.tensor([f["influence"] for f in features], dtype=torch.float)
        else:
            labels = torch.tensor([int(f["relation"]) for f in features], dtype=torch.long)
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
            batch["labels"] = labels
            
        return batch

    def _collate_token(self, features):
        """Collate token classification examples."""
        sentences = [f["sentence"] for f in features]
        
        encodings = self.tokenizer(
            sentences, padding=True, truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_offsets_mapping=True, return_tensors="pt",
        )
        
        max_length = encodings["input_ids"].size(1)

        per_labels = []
        for i, f in enumerate(features):
            labels_i = self._labels_from_spans(encodings.encodings[i], f.get("spans", []))
            padded_labels_i = labels_i + [-100] * (max_length - len(labels_i))
            per_labels.append(padded_labels_i)

        encodings["labels"] = torch.tensor(per_labels, dtype=torch.long)
        encodings["task"] = "token"
        
        if "offset_mapping" in encodings:
            del encodings["offset_mapping"]
            
        return encodings