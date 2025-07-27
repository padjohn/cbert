import torch.nn as nn
import torch
from transformers import AutoModel, AutoConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoTokenizer
from torch.nn import CrossEntropyLoss

class CausalBERTMultiTaskConfig(PretrainedConfig):
    model_type = "causalbert_multitask"
    def __init__(self, span_labels=5, relation_labels=4, base_model_name=None, vocab_size_with_special_tokens=None, relation_class_weights=None, span_class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.num_span_labels = span_labels
        self.num_relation_labels = relation_labels
        self.base_model_name = base_model_name
        self.vocab_size_with_special_tokens = vocab_size_with_special_tokens
        self.relation_class_weights = relation_class_weights
        self.span_class_weights = span_class_weights

class CausalBERTMultiTaskModel(PreTrainedModel):
    config_class = CausalBERTMultiTaskConfig

    def __init__(self, config):
        super().__init__(config)
        
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
        
        if config.vocab_size_with_special_tokens is not None and len(self.bert.get_input_embeddings().weight) != config.vocab_size_with_special_tokens:
            self.bert.resize_token_embeddings(config.vocab_size_with_special_tokens)
            print(f"Resized base BERT model's embeddings to {config.vocab_size_with_special_tokens}.")

        bert_model_type = getattr(self.bert.config, "model_type", "unknown")
        hidden = config.hidden_size
        self.token_classifier    = nn.Linear(hidden, config.num_span_labels)
        self.relation_classifier = nn.Linear(hidden, config.num_relation_labels)
        self.post_init()
        self.relation_loss_weights = None
        if config.relation_class_weights is not None:
            self.relation_loss_weights = torch.tensor(config.relation_class_weights, dtype=torch.float32)
        self.token_loss_weights = None
        if config.span_class_weights is not None:
            self.token_loss_weights = torch.tensor(config.span_class_weights, dtype=torch.float32)


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
            input_keys = ["input_ids", "attention_mask"]
        else:
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

        if task == "token":
            labels_seq = [f["labels_seq"] for f in features]
            padded_labels = [
                l[:seq_len] + [-100] * max(0, seq_len - len(l))
                for l in labels_seq
            ]
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
            batch["task"]   = "token"

        elif task == "relation":
            labels_scalar = [int(f["labels_scalar"]) for f in features]
            batch["labels"] = torch.tensor(labels_scalar, dtype=torch.long)
            batch["task"]   = "relation"

        else:
            raise ValueError("Task must be 'token' or 'relation'")
        return batch