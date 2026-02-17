"""
Inference pipeline: model loading, token/relation prediction, and tuple extraction.
"""

import torch
import numpy as np
from typing import cast
import os 
import logging
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch
from transformers import AutoConfig, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.tokenization_utils import PreTrainedTokenizer
from causalbert.utils import get_compute_dtype, clean_tok
from causalbert.model import (
    CausalBERTMultiTaskModel,
    ID2ROLE,
    ID2POLARITY,
    V2_TO_V3_MAPPING,
    convert_v3_to_v2_label,
    reconstruct_influence,
    PARALLEL_SEP_TOKEN,
    SALIENCE_VALUES
)

logger = logging.getLogger(__name__)

def load_model(model_dir: str, device: str | None = None) -> tuple[CausalBERTMultiTaskModel, PreTrainedTokenizer, PretrainedConfig, str]:
    """
    Load a trained C-BERT model (v2 or v3).
    
    Returns:
        model, tokenizer, config, device
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    compute_dtype, device = get_compute_dtype(device)
    logging.info(f"Using device: {device} with dtype: {compute_dtype}")

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    config.torch_dtype = str(compute_dtype).replace("torch.", "")
    
    arch_version = getattr(config, "architecture_version", 2)
    logging.info(f"Loading C-BERT v{arch_version} model")

    if not hasattr(config, "base_model_name") or config.base_model_name is None:
        raise ValueError("'base_model_name' missing in config.")

    model = CausalBERTMultiTaskModel(config)
    
    # Load weights
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    pytorch_path = os.path.join(model_dir, "pytorch_model.bin")

    if os.path.isfile(safetensors_path):
        logging.info(f"Loading weights from: {safetensors_path}")
        state_dict = safetensors.torch.load_file(safetensors_path, device="cpu")
    elif os.path.isfile(pytorch_path):
        logging.info(f"Loading weights from: {pytorch_path}")
        state_dict = torch.load(pytorch_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model weights found in {model_dir}")

    model.load_state_dict(state_dict)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=False, trust_remote_code=True)
    tokenizer.model_max_length = 512
    logging.info("Tokenizer loaded.")

    cast(nn.Module, model).to(device=device, dtype=compute_dtype)
    model.eval()
    logging.info(f"Model loaded to {device}")
    
    return model, tokenizer, config, device


def _get_token_predictions(model, tokenizer, config, sentences, device, aggregate_words=True):
    """
    Token classification for a batch of sentences.
    Returns list of token predictions per sentence.
    """
    inputs = tokenizer(
        sentences, 
        padding=True, 
        truncation=True, 
        return_tensors="pt", 
        max_length=tokenizer.model_max_length
    ).to(device)
    
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=model.dtype, enabled=(device == "cuda")):
            out = model(**inputs, task="token")
            
    logits = out["logits"]
    probs = F.softmax(logits, dim=-1)
    pred_ids = probs.argmax(-1)
    
    id2label = {int(k): v for k, v in config.id2label_span.items()}

    all_predictions = []
    
    for i in range(len(sentences)):
        tok_ids = inputs["input_ids"][i]
        toks = tokenizer.convert_ids_to_tokens(tok_ids)
        
        token_labels = [id2label[int(pid)] for pid in pred_ids[i].tolist()]
        token_confs = [float(probs[i, j, pred_ids[i, j]].item()) for j in range(len(tok_ids))]
        
        if not aggregate_words:
            enc = tokenizer(sentences[i], return_offsets_mapping=True, truncation=True).encodings[0]
            word_ids_raw = enc.word_ids
            prev_wid = None
            preds = []
            for j, (t, l, c) in enumerate(zip(toks, token_labels, token_confs)):
                wid = word_ids_raw[j] if j < len(word_ids_raw) else None
                if wid is not None and wid != prev_wid:
                    preds.append({"token": t, "label": l, "confidence": round(c, 4)})
                else:
                    preds.append({"token": t, "label": "O", "confidence": round(c, 4)})
                prev_wid = wid
            all_predictions.append(preds)
            continue
            
        # Word-level aggregation (first-subword only → BIOES)
        enc = tokenizer(sentences[i], return_offsets_mapping=True, truncation=True).encodings[0]
        word_ids = enc.word_ids
        
        # Group subword indices by word_id
        word_groups = []
        current = []
        last_wid = None
        for j, wid in enumerate(word_ids):
            if wid is None:
                continue
            if last_wid is None or wid == last_wid:
                current.append(j)
            else:
                word_groups.append(current)
                current = [j]
            last_wid = wid
        if current:
            word_groups.append(current)

        # For each word, take prediction from first subword only
        words, word_labels, word_confs = [], [], []
        for grp in word_groups:
            first_idx = grp[0]
            word_pred = int(pred_ids[i, first_idx].item())
            label = id2label[word_pred]
            conf = float(probs[i, first_idx, word_pred].item())
            surf = "".join(clean_tok(toks[j]) for j in grp).replace("##", "")
            words.append(surf)
            word_labels.append(label)  # already a BIOES label
            word_confs.append(conf)

        all_predictions.append([
            {"token": w, "label": l, "confidence": round(c, 4)} 
            for w, l, c in zip(words, word_labels, word_confs)
        ])

    return all_predictions


def _get_relation_predictions_v2(model, tokenizer, config, sentences_and_pairs, device, rel_per_sentence=50):
    """v2: 14-class relation classification."""
    combined_texts = []
    original_pairs = []
    sep = PARALLEL_SEP_TOKEN
    
    for sentence, pairs in sentences_and_pairs:
        if len(pairs) > rel_per_sentence:
            logging.warning(f"Limiting {len(pairs)} pairs to {rel_per_sentence}")
            pairs = pairs[:rel_per_sentence]
        for indicator, entity in pairs:
            combined_texts.append(f"{indicator} {sep} {entity} {sep} {sentence}")
            original_pairs.append((indicator, entity))

    if not combined_texts:
        return []

    inputs = tokenizer(
        combined_texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=tokenizer.model_max_length
    ).to(device)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=model.dtype, enabled=(device == "cuda")):
            out = model(**inputs, task="relation")
    
    logits = out["logits"].squeeze(1)
    probs = F.softmax(logits, dim=-1)
    predictions = torch.argmax(logits, dim=-1)
    
    results = []
    for i in range(len(original_pairs)):
        pred = int(predictions[i].item())
        conf = round(probs[i][pred].item(), 4)
        label = config.id2label_relation[str(pred)]
        results.append((original_pairs[i], {
            "label": label, 
            "confidence": conf,
            # v3-style fields for compatibility
            "role": None,
            "influence": None,
        }))
        
    return results


def _get_relation_predictions(model, tokenizer, config, sentences_and_pairs, device, rel_per_sentence=50):
    """Route to v2 or v3 relation prediction based on model architecture."""
    arch_version = getattr(config, "architecture_version", 2)
    
    if arch_version == 2:
        return _get_relation_predictions_v2(
            model, tokenizer, config, sentences_and_pairs, device, rel_per_sentence
        )
    else:
        return _get_relation_predictions_v3(
            model, tokenizer, config, sentences_and_pairs, device, rel_per_sentence
        )


def _get_relation_predictions_v3(model, tokenizer, config, sentences_and_pairs, device, rel_per_sentence=50):
    """v3: Factorized role + polarity + salience prediction."""
    combined_texts = []
    original_pairs = []
    sep = PARALLEL_SEP_TOKEN
    
    for sentence, pairs in sentences_and_pairs:
        if len(pairs) > rel_per_sentence:
            logging.warning(f"Limiting {len(pairs)} pairs to {rel_per_sentence}")
            pairs = pairs[:rel_per_sentence]
        for indicator, entity in pairs:
            combined_texts.append(f"{indicator} {sep} {entity} {sep} {sentence}")
            original_pairs.append((indicator, entity))

    if not combined_texts:
        return []

    inputs = tokenizer(
        combined_texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=tokenizer.model_max_length
    ).to(device)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=model.dtype, enabled=(device == "cuda")):
            out = model(**inputs, task="relation")
    
    role_logits = out["role_logits"]
    polarity_logits = out["polarity_logits"]
    salience = out["salience"]
    
    role_probs = F.softmax(role_logits, dim=-1)
    role_preds = torch.argmax(role_logits, dim=-1)
    
    polarity_probs = F.softmax(polarity_logits, dim=-1)
    polarity_preds = torch.argmax(polarity_logits, dim=-1)
    
    results = []
    for i in range(len(original_pairs)):
        role_id = int(role_preds[i].item())
        role_name = ID2ROLE[role_id]
        role_conf = round(role_probs[i][role_id].item(), 4)
        
        pol_id = int(polarity_preds[i].item())
        pol_name = ID2POLARITY[pol_id]
        pol_conf = round(polarity_probs[i][pol_id].item(), 4)
        
        sal_id = int(torch.argmax(salience[i]).item())
        sal_val = SALIENCE_VALUES[sal_id]
        
        # Reconstruct influence: I = ±(polarity) × salience
        inf_val = round(reconstruct_influence(pol_name, sal_val), 4)
        
        # Reconstruct v2 label
        v2_label = convert_v3_to_v2_label(role_name, pol_name, sal_val)
        
        results.append((original_pairs[i], {
            "label": v2_label,
            "confidence": role_conf,
            "role": role_name,
            "role_confidence": role_conf,
            "polarity": pol_name,
            "polarity_confidence": pol_conf,
            "salience": sal_val,
            "influence": inf_val,
        }))
        
    return results


def merge_bio_entities(classified_tokens):
    """
    Merge BIOES tags into entity spans.
    
    Returns:
        List of (text, label, confidence) tuples
    """
    merged = []
    current_tokens = []
    current_label = None
    current_confs = []

    for pred in classified_tokens:
        token = pred["token"]
        label = pred["label"]
        conf = pred["confidence"]

        if '-' in label:
            prefix, actual_label = label.split('-', 1)
        else:
            prefix = label
            actual_label = label

        if prefix == "S":
            # Close any open span, then emit single-token span
            if current_tokens:
                merged.append((
                    "".join(current_tokens).replace("##", ""),
                    current_label,
                    np.mean(current_confs)
                ))
                current_tokens = []
                current_label = None
                current_confs = []
            merged.append((token.replace("##", ""), actual_label, conf))

        elif prefix == "B":
            if current_tokens:
                merged.append((
                    "".join(current_tokens).replace("##", ""),
                    current_label,
                    np.mean(current_confs)
                ))
            current_tokens = [token]
            current_label = actual_label
            current_confs = [conf]

        elif prefix == "I" and current_label == actual_label:
            current_tokens.append(token)
            current_confs.append(conf)

        elif prefix == "E":
            if current_label == actual_label and current_tokens:
                current_tokens.append(token)
                current_confs.append(conf)
                merged.append((
                    "".join(current_tokens).replace("##", ""),
                    current_label,
                    np.mean(current_confs)
                ))
            else:
                # Orphan E: close previous, emit as single
                if current_tokens:
                    merged.append((
                        "".join(current_tokens).replace("##", ""),
                        current_label,
                        np.mean(current_confs)
                    ))
                merged.append((token.replace("##", ""), actual_label, conf))
            current_tokens = []
            current_label = None
            current_confs = []

        else:  # O
            if current_tokens:
                merged.append((
                    "".join(current_tokens).replace("##", ""),
                    current_label,
                    np.mean(current_confs)
                ))
            current_tokens = []
            current_label = None
            current_confs = []

    if current_tokens:
        merged.append((
            "".join(current_tokens).replace("##", ""),
            current_label,
            np.mean(current_confs)
        ))

    return merged

def sentence_analysis(
    model, 
    tokenizer, 
    config, 
    sentences, 
    device="cuda", 
    batch_size=32, 
    rel_per_sentence=50
):
    """
    Full pipeline: token classification -> relation extraction.
    
    Returns list of analysis results per sentence, each containing:
        - sentence: original text
        - token_predictions: per-token BIO predictions
        - merged_spans: merged entity spans
        - derived_relations: relation predictions for all (indicator, entity) pairs
    """
    arch_version = getattr(config, "architecture_version", 2)
    logging.info(f"Running inference with C-BERT v{arch_version}")
    
    all_results = []
    
    for i in tqdm.tqdm(range(0, len(sentences), batch_size), desc="Batches"):
        batch_sentences = sentences[i:i + batch_size]
    
        # Token classification
        batch_token_preds = _get_token_predictions(
            model, tokenizer, config, batch_sentences, device
        )
        
        # Build relation pairs
        relation_pairs_in_batch = []
        sentence_info_in_batch = []
        
        for j, sentence in enumerate(batch_sentences):
            merged_spans = merge_bio_entities(batch_token_preds[j])
            
            indicators = [(s[0], s[2]) for s in merged_spans if s[1] == "INDICATOR"]
            entities = [(s[0], s[2]) for s in merged_spans if s[1] == "ENTITY"]
            
            derived_pairs = [
                (ind_tok, ent_tok) 
                for (ind_tok, _) in indicators 
                for (ent_tok, _) in entities 
                if ind_tok != ent_tok
            ]
            relation_pairs_in_batch.append((sentence, derived_pairs))
            
            sentence_info_in_batch.append({
                "sentence": sentence,
                "token_predictions": batch_token_preds[j],
                "merged_spans": merged_spans,
                "derived_relations": []
            })

        # Relation classification
        if relation_pairs_in_batch:
            relation_results = _get_relation_predictions(
                model, tokenizer, config, relation_pairs_in_batch, device, rel_per_sentence
            )

            current_idx = 0
            for j in range(len(batch_sentences)):
                num_pairs = len(relation_pairs_in_batch[j][1])
                if num_pairs > 0:
                    sentence_info_in_batch[j]["derived_relations"] = \
                        relation_results[current_idx:current_idx + num_pairs]
                    current_idx += num_pairs

        all_results.extend(sentence_info_in_batch)

    return all_results


def extract_tuples(analysis_results, min_confidence=0.5):
    """
    Convert analysis results to (C, E, I) tuples for graph construction.
    
    Args:
        analysis_results: Output from sentence_analysis()
        min_confidence: Minimum confidence threshold
        
    Returns:
        List of dicts with keys: cause, effect, influence, sentence, confidence
    """
    tuples = []
    
    for result in analysis_results:
        sentence = result["sentence"]
        
        for (indicator, entity), rel_info in result.get("derived_relations", []):
            if rel_info["confidence"] < min_confidence:
                continue
                
            label = rel_info["label"]
            
            # Skip NO_RELATION
            if label == "NO_RELATION":
                continue
            
            # Parse role from label
            if "CAUSE" in label:
                cause = entity
                effect = indicator  # The indicator represents what's being caused
            elif "EFFECT" in label:
                cause = indicator
                effect = entity
            else:
                continue  # INTERDEPENDENCY handled separately
            
            # Get influence value
            if rel_info.get("influence") is not None:
                # v3 model: use direct influence value
                influence = rel_info["influence"]
            else:
                # v2 model: derive from label
                influence = _label_to_influence(label)
            
            tuples.append({
                "cause": cause,
                "effect": effect,
                "influence": influence,
                "sentence": sentence,
                "confidence": rel_info["confidence"],
                "label": label,
            })
    
    return tuples


def _label_to_influence(label: str) -> float:
    """Convert v2 label to influence value."""
    return V2_TO_V3_MAPPING.get(label, ("NO_RELATION", 0.0))[1]