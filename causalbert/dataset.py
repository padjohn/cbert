import json
import os
import re
import yaml
import logging
from datasets.arrow_dataset import Dataset
from datasets.combine import concatenate_datasets
import random

logger = logging.getLogger(__name__)

def _load_replacement_entities(yaml_path: str):
    """Loads a list of entities from a YAML file."""
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            entities = yaml.safe_load(f)
            if not isinstance(entities, list) or not all(isinstance(e, str) for e in entities):
                raise TypeError("YAML content must be a list of strings.")
            return entities
    except FileNotFoundError:
        logger.error(f"YAML file not found: {yaml_path}")
        return []
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {yaml_path}. Error: {e}")
        return []
    except TypeError as e:
        logger.error(f"Invalid YAML format in {yaml_path}. Error: {e}")
        return []

def _augment_data(data, replacements):
    """
    Augments the dataset by replacing entities in existing relations with new ones.
    Only entities are replaced to maintain grammatical correctness.
    """
    augmented_data = []
    
    # Create a copy of the original
    original_data = data[:]
    
    for entry in original_data:
        if not entry.get("relations"):
            continue

        original_sentence = entry["sentence"]
        new_sentence = original_sentence
        new_relations = []
        term_mapping = {}

        for rel in entry["relations"]:
            # Copy original indicator
            new_rel = {"indicator": rel["indicator"], "entities": []}
            
            for entity_data in rel["entities"]:
                original_entity_text = entity_data["entity"]
                
                # Check if this entity has already been replaced in this sentence
                if original_entity_text in term_mapping:
                    new_entity_text = term_mapping[original_entity_text]
                else:
                    # Find a new, unique replacement entity
                    candidate_replacements = [e for e in replacements if e not in new_sentence]
                    if not candidate_replacements:
                        logger.warning(f"Could not find a unique replacement for '{original_entity_text}'. Skipping augmentation for this sentence.")
                        new_entity_text = original_entity_text # Fallback to original
                    else:
                        new_entity_text = random.choice(candidate_replacements)
                        
                    # Replace whole words
                    new_sentence = re.sub(r'\b' + re.escape(original_entity_text) + r'\b', new_entity_text, new_sentence, 1)
                    term_mapping[original_entity_text] = new_entity_text
                
                new_entity_data = {
                    "entity": new_entity_text,
                    "relation": entity_data["relation"]
                }
                new_rel["entities"].append(new_entity_data)

            new_relations.append(new_rel)
            
        new_entry = {
            "sentence": new_sentence,
            "relations": new_relations,
            "year": entry.get("year", None)
        }
        augmented_data.append(new_entry)
        
    return original_data + augmented_data

def create_datasets(
    input_json: str,
    base_dir='',
    include_empty=False,
    debug=False,
    augment=0,
    entities_yml='yml/entities.yml',
):
    def clean_sentence(text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\s([,.;:!?])', r'\1', text)
        return text.strip()
    
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
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

    data_tokens = []
    data_relations = []

    with open(input_json, 'r', encoding='utf-8') as f:
        sentences_data = json.load(f)

    if augment > 0:
        logger.info("Starting data augmentation...")
        entity_replacements = _load_replacement_entities(entities_yml)
        if not entity_replacements:
            logger.error("No entities in YAML. Skipping augmentation.")
        else:
            logger.info(f"Loaded {len(entity_replacements)} entities from {entities_yml}")
            augmented_data = _augment_data(sentences_data, entity_replacements)
            if augment == 1:
                sentences_data = augmented_data[len(sentences_data):]
            elif augment == 2:
                sentences_data = augmented_data
        logger.info(f"Data augmentation complete. Total samples: {len(sentences_data)}")
    
    for entry in sentences_data:
        original_sentence = entry["sentence"]
        cleaned_sentence_text = clean_sentence(original_sentence)

        relations = entry.get("relations", [])

        if not relations and not include_empty:
            continue

        all_char_spans = get_char_spans_from_relations(relations, cleaned_sentence_text)
        token_entry = {
            "sentence": cleaned_sentence_text,
            "spans": [{"start": s, "end": e, "type": t} for (s, e, t) in all_char_spans]
        }
        data_tokens.append(token_entry)


        if debug and len(data_tokens) < 5:
            logger.debug(f"Sample #{len(data_tokens)}: {data_tokens[-1]}")

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
    dataset_relations = Dataset.from_list(data_relations) if len(data_relations) > 0 else None

    # Train-test split
    if len(dataset_tokens) > 1:
        tokens_split = dataset_tokens.train_test_split(test_size=0.2)
        tokens_train = tokens_split["train"]
        tokens_test  = tokens_split["test"]
    else:
        tokens_train = dataset_tokens
        tokens_test  = Dataset.from_dict({"sentence": [], "spans": [], "task": []})

    if dataset_relations and len(dataset_relations) > 1:
        def to_relation_id(batch):
            return {"relation": [relation_map[r] for r in batch["relation"]]}
        dataset_relations = dataset_relations.map(to_relation_id, batched=True)
        rel_split = dataset_relations.train_test_split(test_size=0.2)
        rel_train = rel_split["train"]
        rel_test  = rel_split["test"]
    elif dataset_relations:
        rel_train = dataset_relations
        rel_test  = Dataset.from_dict({"sentence": [], "indicator": [], "entity": [], "relation": [], "task": []})
    else:
        rel_train = Dataset.from_dict({"sentence": [], "indicator": [], "entity": [], "relation": [], "task": []})
        rel_test  = Dataset.from_dict({"sentence": [], "indicator": [], "entity": [], "relation": [], "task": []})

    # Tag tasks
    if len(tokens_train) > 0:
        tokens_train = tokens_train.add_column("task", ["token"] * len(tokens_train))
    if len(tokens_test) > 0:
        tokens_test  = tokens_test.add_column("task",  ["token"] * len(tokens_test))
    if len(rel_train) > 0:
        rel_train    = rel_train.add_column("task",   ["relation"] * len(rel_train))
    if len(rel_test) > 0:
        rel_test     = rel_test.add_column("task",    ["relation"] * len(rel_test))

    # Concatenate
    multitask_train = concatenate_datasets([ds for ds in [tokens_train, rel_train] if len(ds) > 0])
    multitask_test  = concatenate_datasets([ds for ds in [tokens_test,  rel_test]  if len(ds) > 0])

    output_dir_multitask = os.path.join(base_dir, "dataset/multitask")
    os.makedirs(output_dir_multitask, exist_ok=True)
    multitask_train.save_to_disk(os.path.join(output_dir_multitask, "train"))
    multitask_test.save_to_disk(os.path.join(output_dir_multitask, "test"))
    logger.info(f"Saved multitask datasets to {output_dir_multitask}")

    print("Token and Relation Classification datasets created and saved.")