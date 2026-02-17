"""
Dataset construction from annotated JSON with negation-aware label computation.
"""

import json
import os
import re
import csv
import yaml
import random
import logging
from collections import Counter
from datasets.arrow_dataset import Dataset
from datasets.combine import concatenate_datasets
from causalbert.model import ROLE2ID, RELATION_MAP_V2, SALIENCE2ID

logger = logging.getLogger(__name__)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# =============================================================================
# Label mappings
# =============================================================================

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


def _augment_data(data, replacements, seed=42):
    """
    Augments the dataset by replacing entities in existing relations with new ones.
    Only entities are replaced to maintain grammatical correctness.
    """
    rng = random.Random(seed)
    augmented_data = []
    
    for entry in data:
        if not entry.get("relations"):
            continue

        original_sentence = entry["sentence"]
        new_sentence = original_sentence
        new_relations = []
        term_mapping = {}

        # Collect entities from relations
        entities_to_replace = set()
        for rel in entry["relations"]:
            for entity_data in rel["entities"]:
                entities_to_replace.add(entity_data["entity"])
        
        # Sort by length (longest first)
        sorted_entities = sorted(entities_to_replace, key=lambda e: (-len(e), e))
        
        # Create replacements
        for original_entity_text in sorted_entities:
            if original_entity_text in term_mapping:
                continue
                
            candidate_replacements = [
                e for e in replacements 
                if e not in term_mapping.values()
                and e not in original_sentence
            ]
            
            if not candidate_replacements:
                # No unique replacement available - skip this entity
                term_mapping[original_entity_text] = original_entity_text
            else:
                new_entity_text = rng.choice(candidate_replacements)
                term_mapping[original_entity_text] = new_entity_text
        
        # Perform replacements using word boundaries
        new_sentence = original_sentence
        for original_text, replacement_text in term_mapping.items():
            if original_text == replacement_text:
                continue  # Skip if no replacement
            pattern = r'\b' + re.escape(original_text) + r'\b'
            new_sentence = re.sub(pattern, replacement_text, new_sentence)
        
        # Build new relations
        for rel in entry["relations"]:
            new_rel = {
                "indicator": rel["indicator"],
                "entities": [],
                **{k: v for k, v in rel.items() if k not in ["indicator", "entities"]}
            }
            
            for entity_data in rel["entities"]:
                original_entity_text = entity_data["entity"]
                new_entity_text = term_mapping.get(original_entity_text, original_entity_text)
                
                new_entity_data = {
                    "entity": new_entity_text,
                    **{k: v for k, v in entity_data.items() if k != "entity"}
                }
                new_rel["entities"].append(new_entity_data)
            
            new_relations.append(new_rel)
        
        # RELAXED VALIDATION: Use substring check instead of word boundary
        # This allows for inflected forms and compound words
        validation_failed = False
        for rel in new_relations:
            for entity_data in rel["entities"]:
                entity = entity_data["entity"]
                
                # Skip entities that weren't actually replaced
                if entity == term_mapping.get(entity, entity):
                    if entity in original_sentence:
                        continue
                
                # Check if entity appears anywhere in the sentence
                # (allows for inflections like "Treibhausgasen" vs "Treibhausgas")
                if entity not in new_sentence:
                    logger.debug(
                        f"Entity '{entity}' not found in: '{new_sentence[:80]}...' "
                        f"(might be inflected form)"
                    )
                    validation_failed = True
                    break
            if validation_failed:
                break
        
        if validation_failed:
            logger.debug("Skipping augmented example due to validation failure")
            continue
        
        new_entry = {
            "sentence": new_sentence,
            "relations": new_relations,
            "year": entry.get("year", None)
        }
        augmented_data.append(new_entry)
    
    return data + augmented_data

def _check_coefficient(data: dict, coeff_name: str) -> bool:
    """Checks if a coefficient is present in the main 'coefficient' field or 'dependent_coefficients'."""
    if data.get("coefficient") == coeff_name:
        return True
    
    indicator_coeffs = [dc.get("coefficient") for dc in data.get("dependent_coefficients", [])]
    if coeff_name in indicator_coeffs:
        return True
        
    return False



def _check_propositional_negation(relation: dict) -> bool:
    """
    Check if a relation carries PROPOSITIONAL negation (e.g. 'nicht verursachen').
    
    Propositional negation neutralizes the entire causal relation (I â†’ 0).
    It lives ONLY in the relation's dependent_coefficients, NOT in the
    indicator's own 'coefficient' field (which encodes inherent polarity).
    
    This is distinct from:
    - Indicator polarity (stoppt â†’ inherently negative, stored in CSV polarity=1)
    - Object negation (Verlust von X â†’ entity-level, stored in entity coefficient)
    """
    
    dep_coeffs = [dc.get("coefficient") for dc in relation.get("dependent_coefficients", [])]
    return "Negation" in dep_coeffs


def _check_object_negation(entity_data: dict) -> bool:
    """
    Check if an entity carries OBJECT negation (e.g. 'Verlust von LebensrÃ¤umen').
    
    Object negation inverts the polarity of the tuple when odd-counted.
    It is stored in the entity's own coefficient or dependent_coefficients.
    """
    return _check_coefficient(entity_data, "Negation")


def _get_indicator_base_polarity(relation: dict, indicator_text: str, 
                                  indicator_map: dict, use_indicator_mapping: bool) -> float:
    """
    Get the inherent polarity of an indicator.
    
    'verursachen' +1.0 (promoting)
    'stoppen'     -1.0 (inhibiting) 
    
    Sources: CSV polarity=1 means inhibiting, or JSON coefficient='Negation' 
    on the indicator itself (NOT dependent_coefficients, which is propositional).
    """
    if use_indicator_mapping and indicator_text in indicator_map:
        props = indicator_map[indicator_text]
        return -1.0 if props['polarity'] == 1 else 1.0
    else:
        # Fallback: indicator's own coefficient field
        if relation.get("coefficient") == "Negation":
            return -1.0
        return 1.0


def _get_indicator_base_salience_flags(relation: dict, indicator_text: str,
                                        indicator_map: dict, use_indicator_mapping: bool) -> tuple:
    """
    Get distribution and priority flags from indicator.
    
    Returns: (D_Relation, P_Relation)
    """
    if use_indicator_mapping and indicator_text in indicator_map:
        props = indicator_map[indicator_text]
        D_Relation = props['distribution'] == 1
        P_Relation = props['priority'] == 1
    else:
        D_Relation = _check_coefficient(relation, "Division")
        P_Relation = _check_coefficient(relation, "Priority")
    
    # Also check dependent_coefficients for Division/Priority 
    # (these are legitimate contextual markers, unlike Negation which is ambiguous there)
    dep_coeffs = [dc.get("coefficient") for dc in relation.get("dependent_coefficients", [])]
    D_Relation = D_Relation or ("Division" in dep_coeffs)
    P_Relation = P_Relation or ("Priority" in dep_coeffs)
    
    return D_Relation, P_Relation

def _load_indicator_csv(csv_path: str) -> dict:
    """Load indicator properties from CSV."""
    indicator_map = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            form = row['form'].lower().strip()
            indicator_map[form] = {
                'distribution': int(row['distribution']),
                'priority': int(row['priority']),
                'polarity': int(row['polarity']),
                'family': row['family'],
            }
    
    return indicator_map

def _process_sentences(sentences_data, indicator_map, use_indicator_mapping, 
                       include_empty, debug, clean_sentence_fn, prop_neg='include'):
    """
    Process raw sentence entries into token classification and relation classification data.
    
    Extracted as a helper so it can be called separately on original and augmented data.
    
    Returns:
        data_tokens: list of token classification entries
        data_relations: list of relation classification entries
    """

    def normalize_relation_label(relation):
        label = relation.upper()
        if label in RELATION_MAP_V2:
            return label
        logger.warning(f"Unknown relation '{relation}', defaulting to NO_RELATION")
        return "NO_RELATION"

    def get_char_spans_from_relations(relations_list, cleaned_sentence_text):
        """Extracts all unique indicator and entity character spans."""
        char_spans = []
        for rel in relations_list:
            indicator_text_cleaned = clean_sentence_fn(rel["indicator"])
            char_start_ind = cleaned_sentence_text.find(indicator_text_cleaned)
            if char_start_ind != -1:
                char_end_ind = char_start_ind + len(indicator_text_cleaned)
                char_spans.append((char_start_ind, char_end_ind, "INDICATOR"))
            else:
                logger.warning(f"Indicator \'{indicator_text_cleaned}\' not found: \'{cleaned_sentence_text}\'")

            for ent_data in rel["entities"]:
                entity_text_cleaned = clean_sentence_fn(ent_data["entity"])
                char_start_ent = cleaned_sentence_text.find(entity_text_cleaned)
                if char_start_ent != -1:
                    char_end_ent = char_start_ent + len(entity_text_cleaned)
                    char_spans.append((char_start_ent, char_end_ent, "ENTITY"))
                else:
                    logger.warning(f"Entity \'{entity_text_cleaned}\' not found: \'{cleaned_sentence_text}\'")
        return char_spans

    data_tokens = []
    data_relations = []

    for entry in sentences_data:
        original_sentence = entry["sentence"]
        cleaned_sentence_text = clean_sentence_fn(original_sentence)
        relations = entry.get("relations", [])

        if not relations and not include_empty:
            continue

        # Token classification data
        all_char_spans = get_char_spans_from_relations(relations, cleaned_sentence_text)
        token_entry = {
            "sentence": cleaned_sentence_text,
            "spans": [{"start": s, "end": e, "type": t} for (s, e, t) in all_char_spans]
        }
        data_tokens.append(token_entry)

        if debug and len(data_tokens) < 5:
            logger.debug(f"Token sample #{len(data_tokens)}: {data_tokens[-1]}")

        # Relation classification data
        for relation in relations:
            indicator_text_cleaned = clean_sentence_fn(relation["indicator"])
                
            # --- Step 1: Check propositional negation ---            
            if _check_propositional_negation(relation):
                if prop_neg == 'skip':
                    # Drop relation entirely — no entries emitted
                    if debug:
                        logger.debug(
                            f"  PROP_NEG -> SKIP: '{indicator_text_cleaned}'"
                        )
                    continue
                elif prop_neg == 'include':
                    # Propositional negation neutralizes the relation -> NO_RELATION
                    for entity_data in relation["entities"]:
                        entity_text_cleaned = clean_sentence_fn(entity_data["entity"])
                        if cleaned_sentence_text.find(entity_text_cleaned) == -1:
                            logger.warning(f"Skipping: Entity '{entity_text_cleaned}' not found")
                            continue
                        
                        rel_entry = {
                            "sentence": cleaned_sentence_text,
                            "indicator": indicator_text_cleaned,
                            "entity": entity_text_cleaned,
                            # v2 format
                            "relation": RELATION_MAP_V2["NO_RELATION"],
                            "relation_label": "NO_RELATION",
                            # v3 format
                            "role": ROLE2ID["NO_RELATION"],
                            "influence": 0.0,
                            "polarity": 0,
                            "salience": 0.0, 
                        }
                        data_relations.append(rel_entry)
                        
                        if debug:
                            logger.debug(
                                f"  PROP_NEG -> NO_RELATION: "
                                f"'{indicator_text_cleaned}' x '{entity_text_cleaned}'"
                            )
                    
                    continue
                # else: prop_neg == 'ignore' -> fall through to normal processing
                elif debug:
                    logger.debug(
                        f"  PROP_NEG -> IGNORE (processing normally): "
                        f"'{indicator_text_cleaned}'"
                    )
            
            # --- Step 2: Indicator base polarity (NOT negation) ---
            indicator_polarity = _get_indicator_base_polarity(
                relation, indicator_text_cleaned, indicator_map, use_indicator_mapping
            )
            
            # --- Step 3: Salience flags from indicator + context markers ---
            D_Relation, P_Relation = _get_indicator_base_salience_flags(
                relation, indicator_text_cleaned, indicator_map, use_indicator_mapping
            )
            

            for entity_data in relation["entities"]:
                entity_text_cleaned = clean_sentence_fn(entity_data["entity"])
                
                if cleaned_sentence_text.find(entity_text_cleaned) == -1:
                    logger.warning(f"Skipping: Entity \'{entity_text_cleaned}\' not found")
                    continue

                original_relation_type = entity_data["relation"].upper()
                
                # Entity-level salience flags
                D_Own_Entity = _check_coefficient(entity_data, "Division")
                P_Own_Entity = _check_coefficient(entity_data, "Priority")

                # Apply propagation rules for salience (unchanged from v2)
                D_Entity = D_Own_Entity or (D_Relation and original_relation_type == "CAUSE")
                P_Entity = P_Own_Entity or (P_Relation and original_relation_type == "CAUSE")

                if original_relation_type == "EFFECT":
                    D_Entity = False
                    P_Entity = False
                
                # --- Step 4: Per-entity object negation ---
                entity_obj_negated = _check_object_negation(entity_data)
                entity_negation_sign = -1.0 if entity_obj_negated else 1.0
                
                if debug and entity_obj_negated:
                    logger.debug(
                        f"  OBJ_NEG on \'{entity_text_cleaned}\': "
                        f"indicator_pol={indicator_polarity:+.0f} x "
                        f"entity_neg={entity_negation_sign:+.0f}"
                    )
                
                # Entity-specific salience (magnitude of I)
                entity_salience = 0.75 if P_Entity else (0.5 if D_Entity else 1.0)
                
                # --- Step 5: Tuple-level influence ---
                tuple_influence = indicator_polarity * entity_negation_sign * entity_salience
                
                # Role from annotation (unchanged)
                if original_relation_type == "INTERDEPENDENCY":
                    role_id = ROLE2ID["CAUSE"]
                else:
                    role_id = ROLE2ID.get(original_relation_type, ROLE2ID["NO_RELATION"])
                
                
                # Reconstruct v2-compatible label from clean components
                if original_relation_type == "INTERDEPENDENCY":
                    v2_label = "INTERDEPENDENCY"
                else:
                    v2_polarity = "NEG" if tuple_influence < 0 else "POS"
                    v2_distribution = "PRIO" if P_Entity else ("DIST" if D_Entity else "MONO")
                    v2_label = f"{v2_distribution}_{v2_polarity}_{original_relation_type}"
                
                v2_label = normalize_relation_label(v2_label)
                if v2_label == 'NO_RELATION':
                    logger.warning(f"No relation \'{original_sentence}")
                v2_id = RELATION_MAP_V2[v2_label]
                
                # Store both formats for flexibility
                rel_entry = {
                    "sentence": cleaned_sentence_text,
                    "indicator": indicator_text_cleaned,
                    "entity": entity_text_cleaned,
                    # v2 format
                    "relation": v2_id,
                    "relation_label": v2_label,
                    # v3 format  
                    "role": role_id,
                    "influence": tuple_influence,
                    "polarity": 0 if tuple_influence >= 0 else 1,
                    "salience": SALIENCE2ID["PRIO"] if P_Entity else (SALIENCE2ID["DIST"] if D_Entity else SALIENCE2ID["MONO"]),
                }
                data_relations.append(rel_entry)
                
                if debug and len(data_relations) < 5:
                    logger.debug(f"Relation sample #{len(data_relations)}: {rel_entry}")

    return data_tokens, data_relations


def _split_sentences_by_group(sentences_data, test_size=0.2, seed=42):
    """
    Split raw sentence entries into train/test groups at the sentence level.
    
    This ensures that when augmentation is applied later, augmented variants
    of a sentence stay in the same partition as the original -- preventing
    data leakage between train and test.
    
    Returns:
        train_sentences: list of sentence entries for training
        test_sentences: list of sentence entries for testing
    """
    rng = random.Random(seed)
    indices = list(range(len(sentences_data)))
    rng.shuffle(indices)
    
    split_point = int(len(indices) * (1 - test_size))
    train_indices = set(indices[:split_point])
    
    train_sentences = [sentences_data[i] for i in range(len(sentences_data)) if i in train_indices]
    test_sentences = [sentences_data[i] for i in range(len(sentences_data)) if i not in train_indices]
    
    return train_sentences, test_sentences


def create_datasets(
    input_json: str,
    out_dir: str = '',
    include_empty: bool = False,
    debug: bool = False,
    indicator_csv: str = os.path.join(DATA_DIR, "indicator_taxonomy.csv"),
    use_indicator_mapping: bool = False,
    augment: int = 0,
    entities_yml: str = os.path.join(DATA_DIR, "entities.yml"),
    prop_neg: str = 'include',
    seed: int = 42,
):
    """
    Create train/test datasets from annotated JSON.
    
    Args:
        input_json: Path to annotated sentences JSON file
        out_dir: Output directory for datasets
        include_empty: Include sentences without relations
        debug: Enable debug logging
        augment: Augmentation mode (0=none, 1=augmented only, 2=original+augmented)
        entities_yml: Path to entity replacement YAML for augmentation
        prop_neg: How to handle propositional negation ('include'=emit as
            NO_RELATION [default], 'skip'=drop relation entirely, 
            'ignore'=process as if no propositional negation present)
    
    The output datasets include BOTH v2 and v3 label formats for flexibility:
        - v2: "relation" field (int 0-13)
        - v3: "role" field (int 0-2) and "influence" field (float in [-1,1])
    
    IMPORTANT: Augmentation is applied ONLY to the training partition to prevent
    data leakage. The split happens at the sentence level before augmentation,
    ensuring that augmented variants of a sentence never appear in the test set.
    """   
    def clean_sentence(text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\s([,.;:!?])', r'\1', text)
        return text.strip()
    
    assert prop_neg in ('include', 'skip', 'ignore'), \
        f"prop_neg must be 'include', 'skip', or 'ignore', got '{prop_neg}'"

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    with open(input_json, 'r', encoding='utf-8') as f:
        sentences_data = json.load(f)

    # Indicator mapping
    indicator_map = {}
    if use_indicator_mapping and indicator_csv:
        indicator_map = _load_indicator_csv(indicator_csv)
        logger.info(f"Loaded {len(indicator_map)} indicators from CSV")

    # =========================================================================
    # Step 1: Split raw sentences BEFORE any augmentation
    # =========================================================================
    train_sentences, test_sentences = _split_sentences_by_group(
        sentences_data, test_size=0.2, seed=seed
    )
    logger.info(f"Sentence-level split: {len(train_sentences)} train, {len(test_sentences)} test")

    # =========================================================================
    # Step 2: Augment ONLY the training sentences
    # =========================================================================
    if augment > 0:
        logger.info("Starting data augmentation (train partition only)...")
        entity_replacements = _load_replacement_entities(entities_yml)
        if not entity_replacements:
            logger.error("No entities in YAML. Skipping augmentation.")
        else:
            logger.info(f"Loaded {len(entity_replacements)} entities from {entities_yml}")
            # _augment_data returns original + augmented
            augmented_train = _augment_data(train_sentences, entity_replacements, seed)
            if augment == 1:
                # Augmented only (drop originals)
                train_sentences = augmented_train[len(train_sentences):]
            elif augment == 2:
                # Original + augmented
                train_sentences = augmented_train
            logger.info(f"Data augmentation complete. Train samples: {len(train_sentences)}")

    # =========================================================================
    # Step 3: Process both partitions into token/relation entries
    # =========================================================================
    train_tokens, train_relations = _process_sentences(
        train_sentences, indicator_map, use_indicator_mapping,
        include_empty, debug, clean_sentence, prop_neg=prop_neg
    )
    test_tokens, test_relations = _process_sentences(
        test_sentences, indicator_map, use_indicator_mapping,
        include_empty, debug, clean_sentence, prop_neg=prop_neg
    )

    logger.info(f"Processed -- Train: {len(train_tokens)} token entries, {len(train_relations)} relation entries")
    logger.info(f"Processed -- Test: {len(test_tokens)} token entries, {len(test_relations)} relation entries")

    # =========================================================================
    # Step 4: Build HuggingFace Datasets
    # =========================================================================
    
    # Token datasets
    tokens_train = Dataset.from_list(train_tokens) if train_tokens else Dataset.from_dict({"sentence": [], "spans": []})
    tokens_test = Dataset.from_list(test_tokens) if test_tokens else Dataset.from_dict({"sentence": [], "spans": []})

    # Relation datasets
    _empty_rel_dict = {
        "sentence": [], "indicator": [], "entity": [], 
        "relation": [], "relation_label": [],
        "role": [], "influence": [], "polarity": [], "salience": [], "task": []
    }
    rel_train = Dataset.from_list(train_relations) if train_relations else Dataset.from_dict(_empty_rel_dict)
    rel_test = Dataset.from_list(test_relations) if test_relations else Dataset.from_dict(_empty_rel_dict)

    # Add task column
    if len(tokens_train) > 0:
        tokens_train = tokens_train.add_column("task", ["token"] * len(tokens_train))
    if len(tokens_test) > 0:
        tokens_test = tokens_test.add_column("task", ["token"] * len(tokens_test))
    if len(rel_train) > 0:
        rel_train = rel_train.add_column("task", ["relation"] * len(rel_train))
    if len(rel_test) > 0:
        rel_test = rel_test.add_column("task", ["relation"] * len(rel_test))

    # Concatenate into multitask datasets
    multitask_train = concatenate_datasets([ds for ds in [tokens_train, rel_train] if len(ds) > 0])
    multitask_test = concatenate_datasets([ds for ds in [tokens_test, rel_test] if len(ds) > 0])

    # Save
    os.makedirs(out_dir, exist_ok=True)
    multitask_train.save_to_disk(os.path.join(out_dir, "train"))
    multitask_test.save_to_disk(os.path.join(out_dir, "test"))
    
    # Log statistics
    logger.info(f"Saved datasets to {out_dir}")
    logger.info(f"  Train: {len(multitask_train)} samples")
    logger.info(f"  Test: {len(multitask_test)} samples")
    
    if len(rel_train) > 0:
        # v2 class distribution
        v2_dist = Counter(rel_train["relation"])
        logger.info(f"  v2 relation distribution (train): {dict(v2_dist)}")
        
        # v3 role distribution
        role_dist = Counter(rel_train["role"])
        role_names = {0: "CAUSE", 1: "EFFECT", 2: "NO_RELATION"}
        logger.info(f"  v3 role distribution (train): " + "{" + ", ".join(f"{role_names[k]}: {v}" for k, v in sorted(role_dist.items())) + "}")

    print(f"Token and Relation Classification datasets with {len(multitask_train)} training samples and {len(multitask_test)} test samples created and saved to {out_dir}")