import json
import os
import re
import logging
import spacy
from transformers import AutoTokenizer
from datasets import Dataset

logger = logging.getLogger(__name__)

def create_datasets(
    base_dir='../data/',
    model_name="google-bert/bert-base-german-cased",
    include_empty=False,
    debug=False,
    dep=False
):
    def clean_sentence(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s([,.;:!?])', r'\1', text)
        return text.strip()

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

    if dep:
        nlp = spacy.load("de_core_news_trf")

    input_json = os.path.join(base_dir, 'output/inception/json', 'all_sentences.json')
    output_dir_tokens = os.path.join(base_dir, f"dataset/{'base' if debug else 'dep'}/token")
    output_dir_relations = os.path.join(base_dir, f"dataset/{'base' if debug else 'dep'}/relation")

    os.makedirs(output_dir_tokens, exist_ok=True)
    os.makedirs(output_dir_relations, exist_ok=True)

    label_map = {
        "O": 0,
        "B-INDICATOR": 1,
        "I-INDICATOR": 2,
        "B-ENTITY": 3,
        "I-ENTITY": 4
    }

    relation_map = {
        "NO_RELATION": 0,
        "CAUSE": 1,
        "EFFECT": 2,
        "INTERDEPENDENCY": 3
    }

    data_tokens = []
    data_relations = []

    def label_span(tokens, char_span, label_type, existing_labels, sentence_text):
        first_token = True
        found_token = False
        for i, (start, end) in enumerate(tokens['offset_mapping']):
            if start < char_span[1] and end > char_span[0]:
                existing_labels[i] = f"B-{label_type}" if first_token else f"I-{label_type}"
                first_token = False
                found_token = True
        if not found_token:
            logger.warning(f"No tokens labeled for span {char_span} ({label_type}) in: {sentence_text}")
        return existing_labels

    def extract_entity_span(sentence, entity):
        start_index = sentence.find(entity)
        if start_index == -1:
            logger.warning(f"Entity '{entity}' not found in: {sentence}")
            return None
        end_index = start_index + len(entity)
        tokenized = tokenizer(sentence, return_offsets_mapping=True, add_special_tokens=True)
        adjusted_start, adjusted_end = None, None
        for start, end in tokenized['offset_mapping']:
            if start == start_index:
                adjusted_start = start
            if end == end_index:
                adjusted_end = end
        if adjusted_start is not None and adjusted_end is not None:
            return (adjusted_start, adjusted_end)
        logger.warning(f"Span misalignment for '{entity}' in: {sentence}")
        return (start_index, end_index)

    def normalize_relation_label(relation):
        label = relation.upper()
        if label in relation_map:
            return label
        logger.warning(f"Unknown relation '{relation}', defaulting to NO_RELATION")
        return "NO_RELATION"

    with open(input_json, 'r', encoding='utf-8') as f:
        sentences = json.load(f)

    for entry in sentences:
        sentence_text = clean_sentence(entry["sentence"])
        relations = entry.get("relations", [])

        if not relations and not include_empty:
            continue

        tokens = tokenizer(sentence_text, return_offsets_mapping=True, add_special_tokens=True)
        token_labels = ['O'] * len(tokens['input_ids'])

        upos, feats, deps = [], [], []
        if dep:
            doc = nlp(sentence_text)
            for token in doc:
                upos.append(token.pos_)
                morph = token.morph.to_json()
                feat_strings = [f"{k}={','.join(sorted(v))}" for k, v in morph.items()]
                feats.append("|".join(sorted(feat_strings)) if feat_strings else "_")
                deps.append(token.dep_)

        for relation in relations:
            indicator_span = extract_entity_span(sentence_text, relation["indicator"])
            if indicator_span:
                token_labels = label_span(tokens, indicator_span, "INDICATOR", token_labels, sentence_text)
            for entity_data in relation["entities"]:
                entity_text = entity_data["entity"]
                entity_span = extract_entity_span(sentence_text, entity_text)
                if entity_span:
                    token_labels = label_span(tokens, entity_span, "ENTITY", token_labels, sentence_text)

        token_entry = {
            "sentence": sentence_text,
            "input_ids": tokens["input_ids"],
            "labels": [label_map[label] for label in token_labels]
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
            indicator_text = relation["indicator"]
            indicator_span = extract_entity_span(sentence_text, indicator_text)
            if not indicator_span:
                continue
            for entity_data in relation["entities"]:
                entity_text = entity_data["entity"]
                relation_type = normalize_relation_label(entity_data["relation"])
                entity_span = extract_entity_span(sentence_text, entity_text)
                if not entity_span:
                    continue
                data_relations.append({
                    "sentence": sentence_text,
                    "indicator": indicator_text,
                    "entity": entity_text,
                    "relation": relation_type
                })
                if debug and len(data_relations) < 5:
                    logger.debug(f"[Relation] Sample #{len(data_relations)}: {data_relations[-1]}")

    dataset_tokens = Dataset.from_list(data_tokens)
    dataset_relations = Dataset.from_list(data_relations)

    def tokenize_relations(examples):
        tokenized = tokenizer(
            examples["sentence"], truncation=True, padding="max_length", max_length=512, return_tensors="pt"
        )
        tokenized.update({
            "indicator": examples["indicator"],
            "entity": examples["entity"],
            "relation": examples["relation"]
        })
        return tokenized

    tokenized_relations = dataset_relations.map(tokenize_relations, batched=True)

    train_test_split_tokens = dataset_tokens.train_test_split(test_size=0.2)
    train_test_split_relations = tokenized_relations.train_test_split(test_size=0.2)

    train_test_split_tokens["train"].save_to_disk(os.path.join(output_dir_tokens, "train"))
    train_test_split_tokens["test"].save_to_disk(os.path.join(output_dir_tokens, "test"))
    train_test_split_relations["train"].save_to_disk(os.path.join(output_dir_relations, "train"))
    train_test_split_relations["test"].save_to_disk(os.path.join(output_dir_relations, "test"))

    print("✅ Token and Relation Classification datasets created and saved.")

if __name__ == "__main__":
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if __name__ == "__main__" else logging.INFO,
        format="%(levelname)s: %(message)s",
        filename="causalbert/log/dataset.txt",
        filemode="w",
    )
    create_datasets(debug=True, dep=True)
