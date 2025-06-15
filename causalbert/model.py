import torch
from torch import nn
from transformers import AutoModel, PreTrainedModel, PretrainedConfig, BertConfig

class CausalBERTMultiTaskConfig(PretrainedConfig):
    def __init__(self, span_labels=5, relation_labels=4, **kwargs):
        super().__init__(**kwargs)
        self.num_span_labels = span_labels
        self.num_relation_labels = relation_labels

class CausalBERTMultiTaskModel(PreTrainedModel):
    config_class = CausalBERTMultiTaskConfig

    def __init__(self, config):
        super().__init__(config)
        # explicitly load a BertModel using the BertConfig part of your custom config
        self.bert = AutoModel.from_pretrained(config._name_or_path)
        hidden_size = self.bert.config.hidden_size
        self.token_classifier = nn.Linear(hidden_size, config.num_span_labels)
        self.relation_classifier = nn.Linear(hidden_size, config.num_relation_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, task=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, return_dict=True)

        if task == "token":
            logits = self.token_classifier(outputs.last_hidden_state)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1)) if labels is not None else None
        elif task == "relation":
            pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
            logits = self.relation_classifier(pooled_output)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels) if labels is not None else None
        else:
            raise ValueError("Task must be 'token' or 'relation'")

        return {"loss": loss, "logits": logits}

class MultiTaskCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        task = features[0]["task"]
        input_keys = ["input_ids", "attention_mask", "token_type_ids"]
        clean_inputs = [{k: f[k] for k in input_keys if k in f} for f in features]

        if task == "token":
            labels = [f["labels_seq"] for f in features]
            max_len = max(len(l) for l in labels)
            padded_labels = [l + [-100] * (max_len - len(l)) for l in labels]
            batch = self.tokenizer.pad(clean_inputs, return_tensors="pt")
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
            batch["task"] = "token"
        elif task == "relation":
            labels = [int(f["labels_scalar"]) for f in features]
            batch = self.tokenizer.pad(clean_inputs, return_tensors="pt")
            batch["labels"] = torch.tensor(labels, dtype=torch.long)
            batch["task"] = "relation"
        else:
            raise ValueError("Task must be 'token' or 'relation'")
        return batch
