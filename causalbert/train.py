import os
import torch
import wandb
import logging
import random
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModel
from datasets.load import load_from_disk
from causalbert.model import CausalBERTMultiTaskModel, MultiTaskCollator
from collections import Counter


class RelationTransform:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        # This function now expects a batch (a dictionary of lists)
        tokenized_batches = []
        for i in range(len(examples["sentence"])):
            indicator = examples.get("indicator", [""])[i]
            entity = examples.get("entity", [""])[i]
            sentence = examples["sentence"][i]
            sep_token_str = "<|parallel_sep|>" 
            
            combined_input = f"{indicator} {sep_token_str} {entity} {sep_token_str} {sentence}"
            tokenized = self.tokenizer(combined_input, truncation=True, max_length=self.tokenizer.model_max_length)

            tokenized.update({
                "task": "relation",
                "labels_scalar": int(examples["relation"][i]),
                "labels_seq": [-100] * len(tokenized["input_ids"])
            })
            tokenized_batches.append(tokenized)

        combined_batch = {key: [d[key] for d in tokenized_batches] for key in tokenized_batches[0]}
        
        return combined_batch

def add_labels_seq(example):
    return {
        **example,
        "task": ["token"] * len(example["input_ids"]),
        "labels_seq": example.get("labels", [-100] * len(example["input_ids"]))
    }

def preprocess_relation(example, tokenizer):
    indicator = example.get("indicator", "")
    entity = example.get("entity", "")
    sentence = example["sentence"]
    sep_token_str = "<|parallel_sep|>" 
    
    combined_input = f"{indicator} {sep_token_str} {entity} {sep_token_str} {sentence}"
    tokenized = tokenizer(combined_input, truncation=True, max_length=tokenizer.model_max_length)

    tokenized.update({
        "task": "relation",
        "labels_scalar": int(example["relation"]),
        "labels_seq": [-100] * len(tokenized["input_ids"])
    })
    return tokenized

def train(
    base_dir='../data',
    model_name="EuroBERT/EuroBERT-610m",
    model_save_name="C-EBERT",
    epochs=3,
    batch_size=8,
    lr=2e-5,
    device=None,
    use_wandb=True,
    gradient_accumulation_steps=4,
    seed=42
):
    # ---------------------------------------------
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "train.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
        ],
        force=True
    )
    # ---------------------------------------------

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    model_save_path = os.path.join(base_dir, f"model/{model_save_name}")

    span_label_map = {"O": 0, "B-INDICATOR": 1, "I-INDICATOR": 2, "B-ENTITY": 3, "I-ENTITY": 4}
    relation_label_map = {"NO_RELATION": 0, "CAUSE": 1, "EFFECT": 2, "INTERDEPENDENCY": 3}
    id2span_label = {v: k for k, v in span_label_map.items()}
    id2relation_label = {v: k for k, v in relation_label_map.items()}
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True, trust_remote_code=True)
    tokenizer.model_max_length = 512
    
    if "<|parallel_sep|>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|parallel_sep|>"]})
        logging.info("Added '<|parallel_sep|>' to tokenizer vocabulary.")

    span_train_path = os.path.join(base_dir, 'dataset/base/token/train')
    span_train = None
    span_class_weights_list = None

    if os.path.exists(span_train_path):
        loaded_span_dataset = load_from_disk(span_train_path)
        if len(loaded_span_dataset) > 0:
            span_train = loaded_span_dataset.map(
                add_labels_seq,
                batched=True,
                remove_columns=["sentence", "labels"]
            )

            all_span_labels = []
            for example in span_train:
                if 'labels_seq' in example and isinstance(example['labels_seq'], list):
                    all_span_labels.extend([label for label in example['labels_seq'] if label != -100])

            span_class_counts = Counter(all_span_labels)
            num_span_classes = len(span_label_map)
            span_weights = [0.0] * num_span_classes

            for label_id in range(num_span_classes):
                count = span_class_counts.get(label_id, 0)
                if count > 0:
                    span_weights[label_id] = 1.0 / count
                else:
                    span_weights[label_id] = 1000.0

            sum_of_raw_span_weights = sum(span_weights)
            if sum_of_raw_span_weights > 0:
                span_class_weights_list = [w / sum_of_raw_span_weights * num_span_classes for w in span_weights]
            else:
                span_class_weights_list = [1.0] * num_span_classes

            logging.info(f"Calculated token (span) class weights (normalized): {span_class_weights_list}")
        else:
            logging.warning(f"Token dataset found at {span_train_path} but it is empty. Skipping token training.")
    else:
        logging.warning(f"No token dataset found at {span_train_path}. Skipping token training.")


    rel_train_path = os.path.join(base_dir, 'dataset/base/relation/train')
    rel_train = None
    relation_class_weights_list = None

    if os.path.exists(rel_train_path):
        loaded_rel_dataset = load_from_disk(rel_train_path)
        if isinstance(loaded_rel_dataset, dict):
            rel_train = loaded_rel_dataset['train']
        else:
            rel_train = loaded_rel_dataset

        if len(rel_train) > 0:
            rel_train = rel_train.add_column("task", ["relation"] * len(rel_train))

            logging.info("After preprocess_relation for rel_train:")
            debug_relation_labels = [example["relation"] for example in rel_train]
            debug_relation_counts = Counter(debug_relation_labels)
            logging.info(f"  Counts of labels: {debug_relation_counts}")
            logging.info(f"  First 10 labels: {debug_relation_labels[:10]}")
            logging.info(f"  Last 10 labels: {debug_relation_labels[-10:]}")

            relation_labels_for_weights = [example["relation"] for example in rel_train]
            class_counts = Counter(relation_labels_for_weights)

            num_relation_classes = len(relation_label_map)
            weights = [0.0] * num_relation_classes

            for label_id in range(num_relation_classes):
                count = class_counts.get(label_id, 0)
                if count > 0:
                    weights[label_id] = 1.0 / count
                else:
                    weights[label_id] = 1000.0

            sum_of_raw_weights = sum(weights)
            if sum_of_raw_weights > 0:
                relation_class_weights_list = [w / sum_of_raw_weights * num_relation_classes for w in weights]
            else:
                relation_class_weights_list = [1.0] * num_relation_classes

            logging.info(f"Calculated relation class weights (normalized): {relation_class_weights_list}")
        else:
            logging.warning(f"Relation dataset found at {rel_train_path} but it is empty. Skipping relation training.")
    else:
        logging.warning(f"No relation dataset found at {rel_train_path}. Skipping relation training.")


    if span_train:
        num_examples_to_log = min(5, len(span_train)) # Log up to 5 examples
        logging.info("--- Sample of Token Classification Data with Labels ---")
        
        for i in range(num_examples_to_log):
            example = span_train[i]
            input_ids = example['input_ids']
            labels_seq = example['labels_seq']

            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            labels = [id2span_label[l] if l != -100 else 'IGNORE' for l in labels_seq]
            
            # Format the output for readability
            token_label_pairs = list(zip(tokens, labels))
            formatted_pairs = "\n".join([f"    - {token:<20} {label}" for token, label in token_label_pairs])

            logging.info(f"Example {i+1}:")
            logging.info(f"Tokens and Labels:\n{formatted_pairs}")
        
        logging.info("---------------------------------------------------")

    collator = MultiTaskCollator(tokenizer)

    span_loader = DataLoader(span_train, batch_size=batch_size, shuffle=True, collate_fn=collator)
    
    all_loaders = [span_loader]
    if rel_train:
        rel_loader = DataLoader(rel_train, batch_size=batch_size, shuffle=True, collate_fn=collator)
        all_loaders.append(rel_loader)


    compute_dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16 if device == "cuda" else torch.float32
    base_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    logging.info("--- STARTING CONFIGURATION PROCESS ---")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    logging.info(f"Base model config loaded. Key attributes: hidden_size={config.hidden_size}, vocab_size={config.vocab_size}, model_type={config.model_type}")
    logging.debug(f"Full base config dict: {config.to_dict()}")
    
    # Multitask-specific configurations
    config.num_span_labels = len(span_label_map)
    config.num_relation_labels = len(relation_label_map)
    config.id2label_span = {str(k): v for k, v in id2span_label.items()}
    config.id2label_relation = {str(k): v for k, v in id2relation_label.items()}
    config.architectures = ["CausalBERTMultiTaskModel"]
    config.torch_dtype = str(compute_dtype).replace("torch.", "")
    config.vocab_size = len(tokenizer)
    config.relation_class_weights = relation_class_weights_list
    config.span_class_weights = span_class_weights_list
    config.model_type = "causalbert_multitask"
    config.base_model_name = model_name
    
    model = CausalBERTMultiTaskModel(config)
    
    model.to(device)

    logging.info(f"Custom attributes added. New key attributes: num_span_labels={config.num_span_labels}, vocab_size (after resize)={config.vocab_size}, base_model_name={config.base_model_name}")
    logging.debug(f"Full final config dict: {config.to_dict()}")
    logging.info("--- CONFIGURATION PROCESS COMPLETE ---")

    
    logging.info(f"Moved model to {device}")
    if hasattr(model, 'bert') and hasattr(model.bert.config, '_attn_implementation'):
        logging.info(f"Base model Attention Implementation: {model.bert.config._attn_implementation}")
    else:
        logging.info("Base model Attention Implementation not explicitly set or found on model.bert.config.")

    if use_wandb:
        wandb.init(project="CausalBERT", entity="norygano", config={
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "device": device,
            "compute_dtype": str(compute_dtype),
            "gradient_accumulation_steps": gradient_accumulation_steps
        })

    optimizer = AdamW(model.parameters(), lr=lr)

    use_grad_scaler = (device == "cuda" and compute_dtype == torch.float16)

    if use_grad_scaler:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    global_step = 0
    for epoch in range(epochs):
        model.train()
        log_str = f"Epoch {epoch+1}/{epochs}"
        print(log_str)
        logging.info(log_str)
        
        for loader in all_loaders:
            task_type = loader.dataset[0]["task"]
            loop = tqdm(loader, desc=f"Training {task_type}")
            if task_type == "token":
                epoch_token_label_counts = Counter()
            
            for step, batch in enumerate(loop):
                task = batch.pop("task")
                labels = batch.pop("labels")
                labels = labels.to(device)
                batch = {k: v.to(device) for k, v in batch.items()}

                if task_type == "token":
                    valid_labels = labels[labels != -100].tolist()
                    epoch_token_label_counts.update(valid_labels)
                
                with torch.autocast(device_type='cuda', dtype=compute_dtype, enabled=(device == "cuda")):
                    out = model(**batch, task=task, labels=labels)
                    loss = out["loss"]
                loss = loss / gradient_accumulation_steps

                if use_grad_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(loader):
                    if use_grad_scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1 
                
                loop.set_postfix(loss=loss.item() * gradient_accumulation_steps)
                if use_wandb:
                    wandb.log({f"{task_type}_loss": loss.item() * gradient_accumulation_steps, "epoch": epoch + 1, "global_step": global_step})
                logging.info({f"{task_type}_loss": loss.item() * gradient_accumulation_steps, "epoch": epoch + 1, "global_step": global_step})
            if task_type == "token":
               logging.info(f"\n(Epoch {epoch+1} Token Label Distribution): {epoch_token_label_counts}")

        log_str = f"Finished Epoch {epoch+1}"
        print(log_str)
        logging.info(log_str)

    if use_wandb:
        wandb.finish()

    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    logging.info(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train()