import torch
import json
import numpy as np
import torch.nn.functional as F
import os 
import logging
import safetensors.torch
from causalbert.model import CausalBERTMultiTaskModel
from transformers import AutoTokenizer, AutoConfig 
import string

logger = logging.getLogger(__name__)

def load_model(model_dir, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = (
        torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported())
        else (torch.float16 if device == "cuda" else torch.float32)
    )
    logging.info(f"Using device: {device} with compute dtype: {compute_dtype}")

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    logging.info("Configuration loaded.")
    # force the actual runtime dtype into config before instantiation
    config.torch_dtype = str(compute_dtype).replace("torch.", "")

    # Check for the base_model_name.
    if not hasattr(config, "base_model_name") or config.base_model_name is None:
        raise ValueError("'base_model_name' missing in saved config. Please ensure training saved it.")

    model = CausalBERTMultiTaskModel(config)
    # load weights to CPU
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    pytorch_path = os.path.join(model_dir, "pytorch_model.bin")

    if os.path.isfile(safetensors_path):
        logging.info(f"Loading model weights from: {safetensors_path}")
        loaded_state_dict = safetensors.torch.load_file(safetensors_path, device="cpu")
    elif os.path.isfile(pytorch_path):
        logging.info(f"Loading model weights from: {pytorch_path}")
        loaded_state_dict = torch.load(pytorch_path, map_location="cpu")
    else:
        logging.error(f"Missing model weights file (pytorch_model.bin or model.safetensors) in {model_dir}.")
        raise FileNotFoundError(f"Missing model weights file (pytorch_model.bin or model.safetensors) in {model_dir}. Please ensure your training saved this file.")

    model.load_state_dict(loaded_state_dict)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    tokenizer.model_max_length = 512
    logging.info("Tokenizer loaded successfully.")

    model.to(device=device, dtype=compute_dtype)
    logging.info(f"Model loaded to {device} with dtype {compute_dtype}")
    model.eval()
    return model, tokenizer, config, device

def _get_token_predictions(model, tokenizer, config, sentence, device, aggregate_words=True):
    """
    Performs token classification and returns tokens, labels, and confidences.
    If aggregate_words=True, subword predictions are aggregated to one label per word.
    """
    # main forward (for logits)
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).to(device)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=model.dtype, enabled=(device == "cuda")):
            out = model(**inputs, task="token")
    logits = out["logits"]  # [B, T, C]
    probs = F.softmax(logits, dim=-1)[0]  # [T, C]
    pred_ids = probs.argmax(-1)           # [T]

    # token-level decode
    tok_ids = inputs["input_ids"][0]
    toks = tokenizer.convert_ids_to_tokens(tok_ids)
    id2label = {int(k): v for k, v in config.id2label_span.items()} if isinstance(next(iter(config.id2label_span.keys())), str) else config.id2label_span
    token_labels = [id2label[int(i.item())] for i in pred_ids]
    token_confs  = [float(probs[i_idx, int(pred_ids[i_idx])].item()) for i_idx in range(len(tok_ids))]

    if not aggregate_words:
        return [{"token": t, "label": l, "confidence": round(c, 4)} for t, l, c in zip(toks, token_labels, token_confs)]

    # word-level aggregation (use fast tokenizer word_ids)
    enc_fast = tokenizer(sentence, return_offsets_mapping=True, truncation=True)
    enc = enc_fast.encodings[0]
    word_ids = enc.word_ids  # list of len T
    # group indices by word_id, skipping specials (None)
    word_groups = []
    current = []
    last_wid = None
    for i, wid in enumerate(word_ids):
        if wid is None:
            continue
        if last_wid is None or wid == last_wid:
            current.append(i)
        else:
            word_groups.append(current)
            current = [i]
        last_wid = wid
    if current:
        word_groups.append(current)

    # aggregate: sum probs across subwords, pick best class per word
    words = []
    word_types = []  # "INDICATOR"/"ENTITY"/"O"
    word_confs = []
    for grp in word_groups:
        grp_probs = probs[grp, :]              # [k, C]
        agg_probs = grp_probs.mean(0)          # average over subwords
        word_pred = int(agg_probs.argmax().item())
        label = id2label[word_pred]
        # normalize to type
        if label == "O":
            wtype = "O"
        else:
            wtype = label.split("-", 1)[-1]    # B-ENTITY -> ENTITY
        # reconstruct surface word from subwords
        surf = "".join(clean_tok(toks[i]) for i in grp).replace("##", "")
        words.append(surf)
        word_types.append(wtype)
        word_confs.append(float(agg_probs[word_pred].item()))

    # convert types to BIO across words
    bio_labels = []
    prev_t = "O"
    for t in word_types:
        if t == "O":
            bio_labels.append("O")
            prev_t = "O"
        else:
            bio_labels.append(("I-" if prev_t == t else "B-") + t)
            prev_t = t

    return [{"token": w, "label": l, "confidence": round(c, 4)} for w, l, c in zip(words, bio_labels, word_confs)]

def _get_relation_predictions(model, tokenizer, config, sentence, iep, device):
    """
    Classifies relations for a given list of indicator-entity pairs.
    """
    if not iep:
        return []

    sep_token_str = "<|parallel_sep|>"
    combined_texts = [f"{indicator} {sep_token_str} {entity} {sep_token_str} {sentence}" for indicator, entity in iep]
    
    # Tokenize all pairs at once
    rel_inputs = tokenizer(
        combined_texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=tokenizer.model_max_length
    ).to(device)

    # Perform inference on the whole batch
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=model.dtype, enabled=(device == "cuda")):
            out = model(**rel_inputs, task="relation")
    
    # The output logits tensor should be (batch_size, num_labels)
    # The unsqueeze(1) in the model's forward pass might make it (batch_size, 1, num_labels), so we squeeze it here.
    logits = out["logits"].squeeze(1)
    
    probs = F.softmax(logits, dim=-1)
    predictions = torch.argmax(logits, dim=-1)
    
    relation_results = []
    
    for i in range(len(iep)):
        pred = predictions[i].item()
        confidence = round(probs[i][pred].item(), 4)
        label = config.id2label_relation[str(pred)]
        indicator, entity = iep[i]
        relation_results.append(((indicator, entity), {"label": label, "confidence": confidence}))
        
    return relation_results

def analyze_sentence(model, tokenizer, config, sentence, iep, device="cuda", confidence=True):
    """
    Analyzes a sentence for token classification and specified relation pairs.

    Args:
        model (CausalBERTMultiTaskModel): The loaded CausalBERT model.
        tokenizer (AutoTokenizer): The loaded tokenizer.
        config (AutoConfig): The model configuration.
        sentence (str): The sentence to analyze.
        iep (list): A list of (indicator, entity) pairs for relation classification.
        device (str, optional): The device to use for inference. Defaults to "cuda".
        confidence (bool, optional): If True, the output will include confidence scores. Defaults to True.

    Returns:
        dict: A dictionary containing the token predictions and relation analysis results.
              The structure of the output depends on the `confidence` flag.
    """
    result = {"tokens": [], "relations": []}
    
    # Token predictions
    token_preds = _get_token_predictions(model, tokenizer, config, sentence, device)
    if confidence:
        result["tokens"] = token_preds
    else:
        result["tokens"] = [(token, label) for token, label, _ in token_preds]

    # Relation predictions
    if iep:
        relation_preds = _get_relation_predictions(model, tokenizer, config, sentence, iep, device)
        if confidence:
            result["relations"] = relation_preds
        else:
            result["relations"] = [((indicator, entity), {'label': info['label']}) for (indicator, entity), info in relation_preds]
            
    return result

# Merge BIO tags into entity spans (Revised and placed in infer.py)
def merge_bio_entities(tokens, labels, confidences):
    merged = []
    current_tokens = []
    current_label = None
    current_confs = []

    for token, label, conf in zip(tokens, labels, confidences):
        if '-' in label:
            prefix, actual_label = label.split('-', 1)
        else:
            prefix = label
            actual_label = label

        if prefix == "B": 
            if current_tokens:
                merged.append(("".join(current_tokens).replace("##", ""), current_label, np.mean(current_confs)))
            current_tokens = [token]
            current_label = actual_label
            current_confs = [conf]
        elif prefix == "I" and current_label == actual_label:
            current_tokens.append(token)
            current_confs.append(conf)
        elif prefix == "I" and current_label != actual_label:
            if current_tokens:
                merged.append(("".join(current_tokens).replace("##", ""), current_label, np.mean(current_confs)))
            current_tokens = [token]
            current_label = actual_label
            current_confs = [conf]
        else: 
            if current_tokens:
                merged.append(("".join(current_tokens).replace("##", ""), current_label, np.mean(current_confs)))
                current_tokens = []
                current_confs = []
            current_label = None

    if current_tokens:
        merged.append(("".join(current_tokens).replace("##", ""), current_label, np.mean(current_confs)))

    return merged

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

def clean_tok(tok: str) -> str:
    """Clean Special Tokens and Umlauts."""
    tok = tok.lstrip("Ġ ").strip("▁ ")
    try:
        tok = tok.encode("latin-1").decode("utf-8")
    except UnicodeError:
        pass
    return tok


def sentence_analysis(model, tokenizer, config, sentence, device="cuda"):
    """
    Performs a complete analysis of a sentence, including token and relation classification.
    """
    # 1. Token predictions
    classified_tokens = _get_token_predictions(model, tokenizer, config, sentence, device)

    # 2. Merge BIO spans
    tokens = [t["token"] for t in classified_tokens]
    labels = [t["label"] for t in classified_tokens]
    confs = [t["confidence"] for t in classified_tokens]
    merged_spans = merge_bio_entities(tokens, labels, confs)

    # 3. Derive indicators and entities
    indicators = []
    entities = []
    
    for token, label, _ in merged_spans:
        if all(c in string.punctuation for c in token) or token in tokenizer.all_special_tokens:
            continue
        cleaned_token = token.lstrip('Ġ').strip(string.punctuation)
        
        if label == "INDICATOR":
            indicators.append(cleaned_token)
        elif label == "ENTITY":
            entities.append(cleaned_token)
            
    # 4. Classify relations
    derived_pairs = [(i, e) for i in indicators for e in entities if i != e]
    relation_results = analyze_sentence(model, tokenizer, config, sentence, derived_pairs, device)

    return {
        "sentence": sentence,
        "token_predictions": classified_tokens,
        "merged_spans": merged_spans,
        "derived_relations": relation_results["relations"] 
    }
