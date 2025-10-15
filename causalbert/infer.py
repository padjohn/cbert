import torch
import json
import numpy as np
import torch.nn.functional as F
import os 
import logging
import tqdm
import safetensors.torch
from causalbert.model import CausalBERTMultiTaskModel
from transformers import AutoTokenizer, AutoConfig 
import string

def load_model(model_dir, device=None):
    # LOGGING -------------------------------
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "infer.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w')
        ],
        force=True
    )
    # ---------------------------------------------

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
    elif device == "cuda":
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    logging.info(f"Using device: {device} with compute dtype: {compute_dtype}")

    try:
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        logging.info("Configuration loaded.")
        logging.debug(f"Full loaded config dict: {config.to_dict()}")
    except Exception as e:
        logging.error(f"Error loading config from {model_dir}: {e}")
        raise RuntimeError(f"Error loading config from {model_dir}: {e}")

    if not hasattr(config, "base_model_name") or config.base_model_name is None:
        logging.error(f"'base_model_name' key is missing or None in the config.json file located at {model_dir}.")
        raise ValueError(
            f"'base_model_name' key is missing or None in the config.json file located at {model_dir}. "
            "Please ensure that your training script correctly saves this value."
        )
    
    try:
        model = CausalBERTMultiTaskModel(config)
        logging.info("CausalBERTMultiTaskModel instantiated from config.")
    except Exception as e:
        logging.error(f"Error instantiating CausalBERTMultiTaskModel: {e}")
        raise

    model = CausalBERTMultiTaskModel(config)
    
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    pytorch_path = os.path.join(model_dir, "pytorch_model.bin")

    loaded_state_dict = None
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

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True
    )
    tokenizer.model_max_length = 512
    logging.info("Tokenizer loaded successfully.")

    model.eval() 
    model.to(device)
    if model.dtype != compute_dtype:
        model.to(compute_dtype)
        logging.warning(f"Model dtype was not {compute_dtype}, converting now.")

    logging.info(f"Model loaded successfully to {device} with dtype {model.dtype}")

    return model, tokenizer, config, device

def _get_token_predictions(model, tokenizer, config, sentence, device):
    """
    Performs token classification and returns tokens, labels, and confidences.
    """
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).to(device)
    with torch.no_grad():
        out = model(**inputs, task="token")
    logits = out["logits"]
    probs = F.softmax(logits, dim=-1)
    preds = logits.argmax(-1).squeeze()
    confs = probs.max(dim=-1).values.squeeze()

    tokens_decoded = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
    labels = [config.id2label_span[str(p.item())] for p in preds]
    confidences = [round(c.item(), 4) for c in confs]
    
    result = []
    for token, label, confidence in zip(tokens_decoded, labels, confidences):
        result.append({
            "token": token,
            "label": label,
            "confidence": confidence
        })
    return result

def _get_relation_predictions(model, tokenizer, config, sentence, iep, device):
    """
    Classifies relations for a given list of indicator-entity pairs.
    """
    relation_results = []
    sep_token_str = "<|parallel_sep|>"
    for indicator, entity in iep:
        input_text = f"{indicator} {sep_token_str} {entity} {sep_token_str} {sentence}"
        rel_inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).to(device)
        with torch.no_grad():
            out = model(**rel_inputs, task="relation")
        logits = out["logits"]
        probs = F.softmax(logits, dim=-1)
        pred = logits.argmax(-1).item()
        confidence = round(probs[0][pred].item(), 4)
        label = config.id2label_relation[str(pred)]
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

def merge_bio_entities(tokens, labels, confidences):
    """
    Merges tokens with BIO (Begin, Inside, Outside) tags into a single, cohesive entity.

    The function iterates through a list of tokens, labels, and confidences,
    and groups adjacent tokens that form a single named entity (e.g., "B-PERSON",
    "I-PERSON"). The confidence for the merged entity is the average of the
    confidences of its constituent tokens.

    Args:
        tokens (list): A list of token strings from the tokenizer.
        labels (list): A list of BIO labels corresponding to each token.
        confidences (list): A list of confidence scores for each token's label.

    Returns:
        list: A list of tuples, where each tuple represents a merged entity
              or a single token that is not part of a multi-token entity.
              Each tuple contains the merged token string, the label, and
              the average confidence score.

    Example:
        >>> tokens = ['[CLS]', 'The', 'ĠUnited', 'ĠStates', 'of', 'ĠAmerica', '[SEP]']
        >>> labels = ['O', 'B-COUNTRY', 'I-COUNTRY', 'I-COUNTRY', 'I-COUNTRY', 'I-COUNTRY', 'O']
        >>> confidences = [0.9, 0.95, 0.92, 0.91, 0.93, 0.90, 0.99]
        >>> merge_bio_entities(tokens, labels, confidences)
        [('The United States of America', 'COUNTRY', 0.922)]
    """
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

    hidden_states = outputs.hidden_hidden_states
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

    with open(output_json, "w") as f:
        json.dump(data, f, indent=2)

    return data

def clean_tok(tok: str) -> str:
    """Clean Special Tokens and Umlauts."""
    tok = tok.lstrip("Ġ ")
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
