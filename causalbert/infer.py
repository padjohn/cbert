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

def _get_token_predictions(model, tokenizer, config, sentences, device, aggregate_words=True):
    """
    Performs token classification for a batch of sentences.
    """
    # Tokenize the entire batch of sentences
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=tokenizer.model_max_length).to(device)
    
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=model.dtype, enabled=(device == "cuda")):
            out = model(**inputs, task="token")
            
    logits = out["logits"]  # [B, T, C]
    probs = F.softmax(logits, dim=-1) # [B, T, C]
    pred_ids = probs.argmax(-1)       # [B, T]
    
    id2label = {int(k): v for k, v in config.id2label_span.items()} if isinstance(next(iter(config.id2label_span.keys())), str) else config.id2label_span

    all_token_predictions = []
    
    # Process each sentence in the batch
    for i in range(len(sentences)):
        tok_ids = inputs["input_ids"][i]
        toks = tokenizer.convert_ids_to_tokens(tok_ids)
        
        token_labels = [id2label[int(id)] for id in pred_ids[i].tolist()]
        token_confs = [float(probs[i, j, pred_ids[i, j]].item()) for j in range(len(tok_ids))]
        
        if not aggregate_words:
            all_token_predictions.append([{"token": t, "label": l, "confidence": round(c, 4)} for t, l, c in zip(toks, token_labels, token_confs)])
            continue
            
        # Word-level aggregation
        enc_fast = tokenizer(sentences[i], return_offsets_mapping=True, truncation=True)
        enc = enc_fast.encodings[0]
        word_ids = enc.word_ids
        word_groups = []
        current = []
        last_wid = None
        for j, wid in enumerate(word_ids):
            if wid is None: continue
            if last_wid is None or wid == last_wid: current.append(j)
            else: word_groups.append(current); current = [j]
            last_wid = wid
        if current: word_groups.append(current)

        # Aggregate and store
        words = []
        word_types = []
        word_confs = []
        for grp in word_groups:
            grp_probs = probs[i, grp, :]
            agg_probs = grp_probs.mean(0)
            word_pred = int(agg_probs.argmax().item())
            label = id2label[word_pred]
            wtype = label.split("-", 1)[-1] if label != "O" else "O"
            surf = "".join(clean_tok(toks[j]) for j in grp).replace("##", "")
            words.append(surf)
            word_types.append(wtype)
            word_confs.append(float(agg_probs[word_pred].item()))

        bio_labels = []
        prev_t = "O"
        for t in word_types:
            if t == "O": bio_labels.append("O"); prev_t = "O"
            else: bio_labels.append(("I-" if prev_t == t else "B-") + t); prev_t = t
        
        all_token_predictions.append([{"token": w, "label": l, "confidence": round(c, 4)} for w, l, c in zip(words, bio_labels, word_confs)])

    return all_token_predictions

def _get_relation_predictions(model, tokenizer, config, sentences_and_pairs, device):
    """
    Classifies relations for a list of (sentence, iep) pairs in a single batch.
    `sentences_and_pairs` should be a list of tuples: [(sentence, iep), ...].
    """
    combined_texts = []
    original_pairs = []
    sep_token_str = "<|parallel_sep|>"
    
    for sentence, iep in sentences_and_pairs:
        for indicator, entity in iep:
            combined_texts.append(f"{indicator} {sep_token_str} {entity} {sep_token_str} {sentence}")
            original_pairs.append((indicator, entity))

    if not combined_texts:
        return []

    rel_inputs = tokenizer(
        combined_texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=tokenizer.model_max_length
    ).to(device)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=model.dtype, enabled=(device == "cuda")):
            out = model(**rel_inputs, task="relation")
    
    logits = out["logits"].squeeze(1)
    probs = F.softmax(logits, dim=-1)
    predictions = torch.argmax(logits, dim=-1)
    
    relation_results = []
    for i in range(len(original_pairs)):
        pred = predictions[i].item()
        confidence = round(probs[i][pred].item(), 4)
        label = config.id2label_relation[str(pred)]
        relation_results.append((original_pairs[i], {"label": label, "confidence": confidence}))
        
    return relation_results

def merge_bio_entities(classified_tokens):
    """
    Merges BIO tags from a list of classified tokens into entity spans.
    
    Args:
        classified_tokens (list): A list of dictionaries, where each dictionary
                                  contains 'token', 'label', and 'confidence'.

    Returns:
        list: A list of merged spans, each with a single label and average confidence.
              Example: [('effect', 'EFFECT', 0.95), ('causal factors', 'INDICATOR', 0.88)]
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

        if prefix == "B": 
            if current_tokens:
                # Store the previous merged entity
                merged.append(("".join(current_tokens).replace("##", ""), current_label, np.mean(current_confs)))
            current_tokens = [token]
            current_label = actual_label
            current_confs = [conf]
        elif prefix == "I" and current_label == actual_label:
            current_tokens.append(token)
            current_confs.append(conf)
        else: # "O" or a new "B" that wasn't a "B"
            if current_tokens:
                merged.append(("".join(current_tokens).replace("##", ""), current_label, np.mean(current_confs)))
            if prefix != "O": # Start a new entity if it's not "O"
                current_tokens = [token]
                current_label = actual_label
                current_confs = [conf]
            else: # Reset for "O"
                current_tokens = []
                current_label = None
                current_confs = []

    if current_tokens:
        merged.append(("".join(current_tokens).replace("##", ""), current_label, np.mean(current_confs)))

    return merged

def sentence_analysis(model, tokenizer, config, sentences, device="cuda", batch_size=16):
    all_results = []
    
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]
    
        batch_token_preds = _get_token_predictions(model, tokenizer, config, batch_sentences, device)
        
        all_relation_pairs_in_batch = []
        sentence_info_in_batch = []
        for j, sentence in enumerate(batch_sentences):
            merged_spans = merge_bio_entities(batch_token_preds[j])
            
            # Derive indicators and entities from the merged spans
            indicators = [(span[0], span[2]) for span in merged_spans if span[1] == "INDICATOR"]
            entities = [(span[0], span[2]) for span in merged_spans if span[1] == "ENTITY"]
            
            # Create pairs and a unique ID for each pair
            derived_pairs = [(ind_tok, ent_tok) for (ind_tok, ind_conf) in indicators for (ent_tok, ent_conf) in entities if ind_tok != ent_tok]
            all_relation_pairs_in_batch.append((sentence, derived_pairs))
            
            sentence_info_in_batch.append({
                "sentence": sentence,
                "token_predictions": batch_token_preds[j],
                "merged_spans": merged_spans,
                "derived_relations": [] 
            })

        if all_relation_pairs_in_batch:
            relation_results = _get_relation_predictions(model, tokenizer, config, all_relation_pairs_in_batch, device)
            
            current_pair_idx = 0
            for j in range(len(batch_sentences)):
                sentence_pairs = all_relation_pairs_in_batch[j][1]
                num_pairs = len(sentence_pairs)
                if num_pairs > 0:
                    sentence_info_in_batch[j]["derived_relations"] = relation_results[current_pair_idx : current_pair_idx + num_pairs]
                    current_pair_idx += num_pairs

        all_results.extend(sentence_info_in_batch)

    return all_results