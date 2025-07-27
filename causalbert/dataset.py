import json
import os
import re
import logging
import spacy
from transformers import AutoTokenizer
from datasets import Dataset
import numpy as np

logger = logging.getLogger(__name__)

def create_datasets(
    input_json_path: str,
    base_dir='../data/',
    model_name="EuroBERT/EuroBERT-2.1B",
    include_empty=False,
    debug=False,
    dep=False
):
    def clean_sentence(text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\s([,.;:!?])', r'\1', text)
        return text.strip()
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True, trust_remote_code=True)
    
    if "<|parallel_sep|>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|parallel_sep|>"]})

    if dep:
        try:
            nlp = spacy.load("de_core_news_trf")
        except OSError:
            logger.error("SpaCy model 'de_core_news_trf' not found. Please run 'python -m spacy download de_core_news_trf'")
            raise

    input_json = input_json_path

    output_dir_tokens = os.path.join(base_dir, f"dataset/{'dep' if dep else 'base'}/token")
    output_dir_relations = os.path.join(base_dir, f"dataset/{'dep' if dep else 'base'}/relation")

    os.makedirs(output_dir_tokens, exist_ok=True)
    os.makedirs(output_dir_relations, exist_ok=True)

    label_map = {
        "O": 0,
        "B-INDICATOR": 1,
        "I-INDICATOR": 2,
        "B-ENTITY": 3,
        "I-ENTITY": 4,
    }

    relation_map = {
        "NO_RELATION": 0,
        "CAUSE": 1,
        "EFFECT": 2,
        "INTERDEPENDENCY": 3,
    }

    def normalize_relation_label(relation):
        label = relation.upper()
        if label in relation_map:
            return label
        logger.warning(f"Unknown relation '{relation}', defaulting to NO_RELATION")
        return "NO_RELATION"

    def get_char_spans_from_relations(relations_list, cleaned_sentence_text):
        """Extracts all unique indicator and entity character spans from the relations list."""
        char_spans = []
        for rel in relations_list:
            indicator_text_cleaned = clean_sentence(rel["indicator"])
            char_start_ind = cleaned_sentence_text.find(indicator_text_cleaned)
            if char_start_ind != -1:
                char_end_ind = char_start_ind + len(indicator_text_cleaned)
                char_spans.append((char_start_ind, char_end_ind, "INDICATOR"))
            else:
                logger.warning(f"Indicator '{indicator_text_cleaned}' not found in cleaned sentence for char span extraction: '{cleaned_sentence_text}'")

            for ent_data in rel["entities"]:
                entity_text_cleaned = clean_sentence(ent_data["entity"])
                char_start_ent = cleaned_sentence_text.find(entity_text_cleaned)
                if char_start_ent != -1:
                    char_end_ent = char_start_ent + len(entity_text_cleaned)
                    char_spans.append((char_start_ent, char_end_ent, "ENTITY"))
                else:
                    logger.warning(f"Entity '{entity_text_cleaned}' not found in cleaned sentence for char span extraction: '{cleaned_sentence_text}'")
        return char_spans

    def label_tokens(tokenized_input, char_spans_with_type):
        """
        Labels tokens with BIO tags based on character spans using word_ids.
        Args:
            tokenized_input: The output from tokenizer(text, return_offsets_mapping=True)
            char_spans_with_type: List of (char_start, char_end, type) tuples for spans (e.g., "ENTITY", "INDICATOR")
        Returns:
            A list of BIO labels for each token.
        """
        word_ids = tokenized_input.word_ids()
        offset_mapping = tokenized_input.offset_mapping
        
        token_labels_str = ["O"] * len(word_ids)
        char_spans_with_type.sort(key=lambda x: (x[0], x[1]))

        for char_start, char_end, span_type in char_spans_with_type:
            previous_word_idx = None
            
            for token_idx, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue

                token_char_start, token_char_end = offset_mapping[token_idx]

                token_text = tokenized_input.tokens()[token_idx]
                if token_text.startswith('Ġ') and token_char_start < token_char_end:
                    token_char_start += 1
                overlap_start = max(char_start, token_char_start)
                overlap_end = min(char_end, token_char_end)

                if overlap_start < overlap_end:
                    current_word_idx = word_ids[token_idx]

                    if current_word_idx != previous_word_idx:
                        token_labels_str[token_idx] = f"B-{span_type}"
                    else:
                        token_labels_str[token_idx] = f"I-{span_type}"
                    previous_word_idx = current_word_idx
        return token_labels_str

    data_tokens = []
    data_relations = []

    with open(input_json, 'r', encoding='utf-8') as f:
        sentences_data = json.load(f)

    for entry in sentences_data:
        original_sentence = entry["sentence"]
        cleaned_sentence_text = clean_sentence(original_sentence)

        relations = entry.get("relations", [])

        if not relations and not include_empty:
            continue

        tokenized_output = tokenizer(
            cleaned_sentence_text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=True,
            max_length=512
        )
        input_ids = tokenized_output['input_ids']

        all_char_spans = get_char_spans_from_relations(relations, cleaned_sentence_text)

        current_token_labels_str = label_tokens(tokenized_output, all_char_spans)
        upos, feats, deps = [], [], []
        if dep:
            doc = nlp(cleaned_sentence_text)
            for token in doc:
                upos.append(token.pos_)
                morph = token.morph.to_json()
                feat_strings = [f"{k}={','.join(sorted(v))}" for k, v in morph.items()]
                feats.append("|".join(sorted(feat_strings)) if feat_strings else "_")
                deps.append(token.dep_)
                
        token_entry = {
            "sentence": cleaned_sentence_text,
            "input_ids": input_ids,
            "labels": [label_map[label] for label in current_token_labels_str]
        }
        if dep:
            token_entry.update({
                "upos": upos,
                "feats": feats,
                "dep": deps
            })
        data_tokens.append(token_entry)
        if debug and len(data_tokens) < 5:
            logger.debug(f"[Token] Sample #{len(data_tokens)}: {data_tokens[-1]}")

        for relation in relations:
            indicator_text_cleaned = clean_sentence(relation["indicator"])
            if cleaned_sentence_text.find(indicator_text_cleaned) == -1:
                logger.warning(f"Skipping relation. Indicator '{indicator_text_cleaned}' not found in cleaned sentence: '{original_sentence}' -> '{cleaned_sentence_text}'")
                continue

            for entity_data in relation["entities"]:
                entity_text_cleaned = clean_sentence(entity_data["entity"])
                
                if cleaned_sentence_text.find(entity_text_cleaned) == -1:
                    logger.warning(f"Skipping relation. Entity '{entity_text_cleaned}' not found in cleaned sentence for indicator '{indicator_text_cleaned}': '{original_sentence}' -> '{cleaned_sentence_text}'")
                    continue

                original_relation_type_str = entity_data["relation"].upper()
                relation_type = normalize_relation_label(original_relation_type_str)
                data_relations.append({
                    "sentence": cleaned_sentence_text,
                    "indicator": indicator_text_cleaned,
                    "entity": entity_text_cleaned,
                    "relation": relation_type
                })
                if debug and len(data_relations) < 5:
                    logger.debug(f"[Relation] Sample #{len(data_relations)}: {data_relations[-1]}")

    dataset_tokens = Dataset.from_list(data_tokens)
    
    if len(data_relations) > 0:
        dataset_relations = Dataset.from_list(data_relations)

        def tokenize_relations(examples):
            sep_token_str = "<|parallel_sep|>"
            combined_inputs = [
                f"{ind} {sep_token_str} {ent} {sep_token_str} {sent}"
                for ind, ent, sent in zip(examples["indicator"], examples["entity"], examples["sentence"])
            ]
            tokenized = tokenizer(
                combined_inputs, truncation=True, padding="max_length", max_length=512, return_tensors="pt"
            )
            tokenized.update({
                "indicator": examples["indicator"],
                "entity": examples["entity"],
                "relation": [relation_map[r] for r in examples["relation"]]
            })
            return tokenized

        tokenized_relations = dataset_relations.map(tokenize_relations, batched=True)
        if len(tokenized_relations) > 1:
            train_test_split_relations = tokenized_relations.train_test_split(test_size=0.2)
            train_test_split_relations["train"].save_to_disk(os.path.join(output_dir_relations, "train"))
            train_test_split_relations["test"].save_to_disk(os.path.join(output_dir_relations, "test"))
        else:
            logger.warning("Relation dataset has too few samples for train-test split. Saving all to train.")
            tokenized_relations.save_to_disk(os.path.join(output_dir_relations, "train"))
            Dataset.from_dict({"input_ids": [], "attention_mask": [], "indicator": [], "entity": [], "relation": []}).save_to_disk(os.path.join(output_dir_relations, "test"))

    else:
        logger.warning("No relation examples to process. Skipping relation dataset creation.")
        os.makedirs(os.path.join(output_dir_relations, "train"), exist_ok=True)
        os.makedirs(os.path.join(output_dir_relations, "test"), exist_ok=True)
        Dataset.from_dict({"input_ids": [], "attention_mask": [], "indicator": [], "entity": [], "relation": []}).save_to_disk(os.path.join(output_dir_relations, "train"))
        Dataset.from_dict({"input_ids": [], "attention_mask": [], "indicator": [], "entity": [], "relation": []}).save_to_disk(os.path.join(output_dir_relations, "test"))

    if len(dataset_tokens) > 1:
        train_test_split_tokens = dataset_tokens.train_test_split(test_size=0.2)
        train_test_split_tokens["train"].save_to_disk(os.path.join(output_dir_tokens, "train"))
        train_test_split_tokens["test"].save_to_disk(os.path.join(output_dir_tokens, "test"))
    else:
        logger.warning("Token dataset has too few samples for train-test split. Saving all to train.")
        dataset_tokens.save_to_disk(os.path.join(output_dir_tokens, "train"))
        Dataset.from_dict({"sentence": [], "input_ids": [], "labels": []}).save_to_disk(os.path.join(output_dir_tokens, "test"))


    print("✅ Token and Relation Classification datasets created and saved.")