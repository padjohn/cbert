import torch.nn as nn
import torch.nn.functional as F
import torch
import logging
from transformers import AutoModel
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)

class CausalBERTMultiTaskConfig(PretrainedConfig):
    model_type = "causalbert_multitask"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_span_labels = kwargs.get("num_span_labels", 5)
        self.num_relation_labels = kwargs.get("num_relation_labels", 4)
        self.id2label_span = kwargs.get("id2label_span", {})
        self.id2label_relation = kwargs.get("id2label_relation", {})
        self.torch_dtype = kwargs.get("torch_dtype", "float32")
        self.vocab_size = kwargs.get("vocab_size", 0)
        self.relation_class_weights = kwargs.get("relation_class_weights", None)
        self.span_class_weights = kwargs.get("span_class_weights", None)
        self.base_model_name = kwargs.get("base_model_name", None)

class CausalBERTMultiTaskModel(PreTrainedModel):
    config_class = CausalBERTMultiTaskConfig

    def __init__(self, config):
        super().__init__(config)

        logging.info("Initializing CausalBERTMultiTaskModel with the following config:")
        logging.info(f"  - Base Model Name: {config.base_model_name}")
        logging.info(f"  - Vocab Size: {config.vocab_size}")
        logging.info(f"  - Num Span Labels: {config.num_span_labels}")
        logging.info(f"  - Num Relation Labels: {config.num_relation_labels}")
        
        # Prefer bf16 > f16 > f32; respect explicit user override for bf16/f16
        dtype_str = getattr(config, "torch_dtype", None)
        if dtype_str in ("bfloat16", "float16"):
            torch_dtype_val = getattr(torch, dtype_str)
        else:
            # auto-detect based on runtime
            torch_dtype_val = (
                torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
                else (torch.float16 if torch.cuda.is_available() else torch.float32)
            )

        self.bert = AutoModel.from_pretrained(
            config.base_model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype_val
        )
        # persist the actual dtype used
        config.torch_dtype = str(torch_dtype_val).replace("torch.", "")
        logging.info(f"Loaded base model with dtype: {config.torch_dtype}")
        
        if len(self.bert.get_input_embeddings().weight) != config.vocab_size:
            self.bert.resize_token_embeddings(config.vocab_size)
            logging.info(f"Resized base BERT model's embeddings to {config.vocab_size}.")

        hidden = self.bert.config.hidden_size
        self.token_classifier    = nn.Linear(hidden, config.num_span_labels)
        self.relation_classifier = nn.Linear(hidden, config.num_relation_labels)
        self.post_init()
        logging.info("Model classifiers for token and relation tasks initialized.")

        self.span_loss = nn.CrossEntropyLoss(
            weight=(torch.tensor(config.span_class_weights, dtype=torch.float)
                    if getattr(config, "span_class_weights", None) else None),
            ignore_index=-100
        )
        self.relation_loss = nn.CrossEntropyLoss(
            weight=(torch.tensor(config.relation_class_weights, dtype=torch.float)
                    if getattr(config, "relation_class_weights", None) else None)
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = getattr(self.config, "initializer_range", 0.02)
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, task=None, **kwargs):
        bert_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
        }
        if token_type_ids is not None and getattr(self.bert.config, "type_vocab_size", 0) > 1:
            bert_inputs["token_type_ids"] = token_type_ids
        out = self.bert(**bert_inputs)
        
        if task == "token":
            token_logits = self.token_classifier(out.last_hidden_state)
            loss = None
            if labels is not None:
                # [B, L, C] -> [B*L, C], labels -> [B*L]
                loss = self.span_loss(
                    token_logits.view(-1, token_logits.size(-1)),
                    labels.view(-1)
                )
            return {"loss": loss, "logits": token_logits}

        elif task == "relation":
            relation_logits = self.relation_classifier(out.last_hidden_state[:, 0])
            relation_logits = relation_logits.unsqueeze(1)
            
            loss = None
            if labels is not None:
                if labels.dim() < 2:
                    labels = labels.unsqueeze(-1)
                loss = self.relation_loss(relation_logits.squeeze(1), labels.squeeze(-1))            
            return {"loss": loss, "logits": relation_logits}

        else:
            logging.error(f"Invalid task specified: {task}. Must be 'token' or 'relation'.")
            raise ValueError(f"Task must be 'token' or 'relation', but got {task}")

AutoModel.register(CausalBERTMultiTaskConfig, CausalBERTMultiTaskModel)

class MultiTaskCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.label2id = {
            "O": 0,
            "B-INDICATOR": 1,
            "I-INDICATOR": 2,
            "B-ENTITY": 3,
            "I-ENTITY": 4,
        }

    def _labels_from_spans(self, enc, spans):
        # enc: tokenizer Encoding (per-sample)
        tokens = enc.tokens
        offsets = enc.offsets
        word_ids = enc.word_ids

        # -100 for special tokens, else "O"
        labels = [-100 if wid is None else self.label2id["O"] for wid in word_ids]

        # sort spans 
        spans_sorted = sorted(spans, key=lambda x: (x["start"], x["end"]))
        for span in spans_sorted:
            s, e = span["start"], span["end"]
            t = span["type"]  # "INDICATOR" or "ENTITY"
            started = False

            for i, wid in enumerate(word_ids):
                if wid is None:
                    continue
                ts, te = offsets[i]

                # Adjust for leading markers
                tok = tokens[i]
                if (tok.startswith("Ġ") or tok.startswith("▁")) and ts < te:
                    ts += 1

                if max(s, ts) < min(e, te):
                    if labels[i] != self.label2id["O"] and labels[i] != -100:
                        continue
                    tag = f"B-{t}" if not started else f"I-{t}"
                    labels[i] = self.label2id[tag]
                    started = True

        return labels

    def __call__(self, features):
        f0 = features[0]
        task = f0.get("task")
        if task is None:
            task = "relation" if ("relation" in f0 and "indicator" in f0) else "token"
        if not all((f.get("task", task) == task) for f in features):
            raise ValueError("Mixed 'task' values in a batch. Ensure batches are homogeneous.")

        if task == "relation":
            sep = "<|parallel_sep|>"
            indicators = [f.get("indicator", "") for f in features]
            entities   = [f.get("entity", "") for f in features]
            sentences  = [f["sentence"] for f in features]
            combined   = [f"{i} {sep} {e} {sep} {s}" for i, e, s in zip(indicators, entities, sentences)]
            toks = self.tokenizer(
                combined,
                padding=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            labels = torch.tensor([int(f["relation"]) for f in features], dtype=torch.long)

            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
            batch = {**toks, "labels": labels, "task": "relation"}
            return batch

        if "sentence" in f0 and "spans" in f0:
            sentences = [f["sentence"] for f in features]
            
            # Correctly tokenize the batch with padding enabled
            encodings = self.tokenizer(
                sentences,
                padding=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            
            # Get the padded length from the tokenizer output
            max_length = encodings["input_ids"].size(1)

            per_labels = []
            for i, f in enumerate(features):
                # Pass the tokenized encoding to your label function
                labels_i = self._labels_from_spans(encodings.encodings[i], f.get("spans", []))
                
                # Pad the labels to the max length
                padded_labels_i = labels_i + [-100] * (max_length - len(labels_i))
                per_labels.append(padded_labels_i)

            labels = torch.tensor(per_labels, dtype=torch.long)
            # Ensure labels tensor is always at least 2D
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)

            # Create the final labels tensor
            encodings["labels"] = torch.tensor(per_labels, dtype=torch.long)
            encodings["task"] = "token"
            
            if "offset_mapping" in encodings:
                del encodings["offset_mapping"]
                
            return encodings

        else:
            raise ValueError("Task must be 'token' or 'relation'")