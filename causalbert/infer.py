import torch
import json
import numpy as np
from transformers import AutoTokenizer
from .model import CausalBERTMultiTaskModel, CausalBERTMultiTaskConfig

def load_model(model_dir, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    config = CausalBERTMultiTaskConfig.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = CausalBERTMultiTaskModel.from_pretrained(model_dir, config=config).eval().to(device)
    return model, tokenizer, config, device

def predict_token_labels(model, tokenizer, config, sentence, device="cuda"):
    tokens = tokenizer(sentence.strip(), return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        out = model(**tokens, task="token")
    preds = out["logits"].argmax(-1).squeeze().tolist()
    input_ids = tokens["input_ids"].squeeze()
    tokens_decoded = tokenizer.convert_ids_to_tokens(input_ids)
    labels = [config.id2label_span.get(str(p), "O") for p in preds]
    return list(zip(tokens_decoded, labels))

def predict_relation_label(model, tokenizer, config, sentence, indicator, entity, device="cuda"):
    input_text = f"{indicator.strip()} [SEP] {entity.strip()} [SEP] {sentence.strip()}"
    tokens = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        out = model(**tokens, task="relation")
    pred = out["logits"].argmax(-1).item()
    return config.id2label_relation.get(str(pred), "None")

def predict_relations_batch(model, tokenizer, config, test_cases, device="cuda"):
    """
    test_cases: list of (indicator, entity, sentence)
    returns: list of relation labels
    """
    inputs = [
        f"{indicator} [SEP] {entity} [SEP] {sentence}"
        for (indicator, entity, sentence) in test_cases
    ]
    encodings = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        out = model(**encodings, task="relation")

    preds = out["logits"].argmax(-1).cpu().tolist()
    return [config.id2label_relation[str(p)] for p in preds]

def analyze_sentence(model, tokenizer, config, sentence, indicators_entities=[], device="cuda"):
    """
    - Runs token classification
    - Runs relation classification for given (indicator, entity) pairs
    """
    result = {}

    # Token classification
    token_preds = predict_token_labels(model, tokenizer, config, sentence, device)
    result["tokens"] = token_preds

    # Relation classification
    if indicators_entities:
        test_cases = [(i, e, sentence) for i, e in indicators_entities]
        relation_preds = predict_relations_batch(model, tokenizer, config, test_cases, device)
        result["relations"] = list(zip(indicators_entities, relation_preds))
    else:
        result["relations"] = []

    return result


def classify_relation(model, tokenizer, config, sentence, indicator, entity):
    model.eval()
    inputs = tokenizer(
        sentence,
        text_pair=f"{indicator} [SEP] {entity}",
        return_tensors="pt",
        truncation=True
    ).to(model.device)

    with torch.no_grad():
        output = model(**inputs, task="relation")
        logits = output["logits"].squeeze()
        probs = torch.softmax(logits, dim=-1)
        pred_id = torch.argmax(probs).item()

    return {
        "label": config.id2label_relation[str(pred_id)],
        "confidence": probs[pred_id].item()
    }

def analyze_sentence_with_confidence(model, tokenizer, config, sentence, indicator_entity_pairs, device="cuda"):
    result = {"tokens": [], "relations": []}

    # Token classification with confidence
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        out = model(**inputs, task="token")
    logits = out["logits"]
    probs = torch.nn.functional.softmax(logits, dim=-1)
    preds = logits.argmax(-1).squeeze()
    confs = probs.max(dim=-1).values.squeeze()

    tokens_decoded = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
    labels = [config.id2label_span[str(p.item())] for p in preds]
    confidences = [round(c.item(), 4) for c in confs]
    result["tokens"] = list(zip(tokens_decoded, labels, confidences))

    # Relation classification with confidence
    for indicator, entity in indicator_entity_pairs:
        input_text = f"{indicator} [SEP] {entity} [SEP] {sentence}"
        rel_inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            out = model(**rel_inputs, task="relation")
        logits = out["logits"]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred = logits.argmax(-1).item()
        confidence = round(probs[0][pred].item(), 4)
        label = config.id2label_relation[str(pred)]
        result["relations"].append(((indicator, entity), {"label": label, "confidence": confidence}))

    return result

def analyze_token_trajectories(model, tokenizer, sentence, target_tokens=None, output_json="embedding_trajectories.json"):
    inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    with torch.no_grad():
        outputs = model.bert(**inputs, output_hidden_states=True, return_dict=True)

    hidden_states = outputs.hidden_states
    layer_embeddings = torch.stack(hidden_states)
    layer_embeddings = layer_embeddings.squeeze(1)

    # Choose which tokens to analyze
    token_indices = (
        [tokens.index(tok) for tok in target_tokens if tok in tokens]
        if target_tokens else list(range(len(tokens)))
    )

    # Collect vectors per token across layers
    data = []
    for tok_idx in token_indices:
        token_label = tokens[tok_idx]
        trajectory = layer_embeddings[:, tok_idx, :].cpu().numpy()
        for layer_idx, vec in enumerate(trajectory):
            data.append({
                "token": token_label,
                "layer": layer_idx,
                "embedding": vec.tolist()
            })

    # Save JSON
    with open(output_json, "w") as f:
        json.dump(data, f, indent=2)

    return data