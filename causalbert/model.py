import torch.nn as nn
import torch
import logging
from transformers import AutoModel
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from torch.nn import CrossEntropyLoss

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
        
        torch_dtype_val = None
        if getattr(config, "torch_dtype", None) == "float16":
            torch_dtype_val = torch.float16
        elif getattr(config, "torch_dtype", None) == "bfloat16":
            torch_dtype_val = torch.bfloat16
        elif getattr(config, "torch_dtype", None) == "float32":
            torch_dtype_val = torch.float32

        self.bert = AutoModel.from_pretrained(
            config.base_model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype_val
        )
        
        if len(self.bert.get_input_embeddings().weight) != config.vocab_size:
            self.bert.resize_token_embeddings(config.vocab_size)
            logging.info(f"Resized base BERT model's embeddings to {config.vocab_size}.")

        hidden = self.bert.config.hidden_size
        self.token_classifier    = nn.Linear(hidden, config.num_span_labels)
        self.relation_classifier = nn.Linear(hidden, config.num_relation_labels)
        self.post_init()
        logging.info("Model classifiers for token and relation tasks initialized.")

        self.relation_loss_weights = None
        if config.relation_class_weights is not None:
            self.relation_loss_weights = torch.tensor(config.relation_class_weights, dtype=torch.float32)
            logging.info(f"Loaded relation loss weights: {self.relation_loss_weights}")
        self.token_loss_weights = None
        if config.span_class_weights is not None:
            self.token_loss_weights = torch.tensor(config.span_class_weights, dtype=torch.float32)
            logging.info(f"Loaded token loss weights: {self.token_loss_weights}")

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
            logits = self.token_classifier(out.last_hidden_state)
        elif task == "relation":
            logits = self.relation_classifier(out.last_hidden_state[:,0])
        else:
            logging.error(f"Invalid task specified: {task}. Must be 'token' or 'relation'.")
            raise ValueError(f"Task must be 'token' or 'relation', but got {task}")
        
        loss = None
        if labels is not None:
            if task == "token":
                if self.token_loss_weights is not None:
                    loss_fct = CrossEntropyLoss(ignore_index=-100, weight=self.token_loss_weights.to(labels.device))
                else:
                    loss_fct = CrossEntropyLoss(ignore_index=-100)
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)
                loss = loss_fct(logits_flat, labels_flat)
            elif task == "relation":
                if self.relation_loss_weights is not None:
                    loss_fct = CrossEntropyLoss(weight=self.relation_loss_weights.to(labels.device))
                else:
                    loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            else:
                pass

        return {"loss": loss, "logits": logits}

AutoModel.register(CausalBERTMultiTaskConfig, CausalBERTMultiTaskModel)

class MultiTaskCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        task = features[0]["task"]

        if task == "relation":
            indicators = [f.get("indicator", "") for f in features]
            entities = [f.get("entity", "") for f in features]
            sentences = [f["sentence"] for f in features]
            sep_token_str = "<|parallel_sep|>" 
            
            combined_inputs = [f"{i} {sep_token_str} {e} {sep_token_str} {s}" for i, e, s in zip(indicators, entities, sentences)]
            
            batch = self.tokenizer(
                combined_inputs,
                padding=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt"
            )
            
            labels_scalar = [f["relation"] for f in features]
            batch["labels"] = torch.tensor(labels_scalar, dtype=torch.long)
            batch["task"] = task

        elif task == "token":
            input_keys = ["input_ids", "attention_mask", "token_type_ids"]
            max_len = self.tokenizer.model_max_length

            clean_inputs = []
            for f in features:
                ci = {}
                for k in input_keys:
                    if k not in f:
                        continue
                    v = f[k]
                    if isinstance(v, list) and len(v) > max_len:
                        v = v[:max_len]
                    ci[k] = v
                clean_inputs.append(ci)

            batch = self.tokenizer.pad(
                clean_inputs,
                padding=True,
                return_tensors="pt"
            )
            seq_len = batch["input_ids"].size(1)

            labels_seq = [f["labels_seq"] for f in features]
            padded_labels = [
                l[:seq_len] + [-100] * max(0, seq_len - len(l))
                for l in labels_seq
            ]
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
            batch["task"] = "token"

        else:
            raise ValueError("Task must be 'token' or 'relation'")
        return batch