"""
Error analysis for C-BERT models (v2 and v3 architectures).

Provides decomposed evaluation of relation classification (per-head accuracy,
confusion matrices, error type breakdown) and span-level token classification
(strict/partial matching, boundary errors, type confusions).

Requires scikit-learn: ``pip install causalbert[eval]``

Usage::

    from causalbert.error_analysis import error_analysis, token_error_analysis

    # Full relation + token analysis
    error_analysis("path/to/model", "path/to/test_dataset")

    # Token span analysis only
    token_error_analysis("path/to/model", "path/to/test_dataset")
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
from collections import Counter
from datasets.load import load_from_disk
from causalbert.infer import load_model
from causalbert.model import (
    ID2ROLE, ID2POLARITY,
    ID2SALIENCE, SALIENCE_VALUES,
    PARALLEL_SEP_TOKEN, convert_v3_to_v2_label,
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
)


# ── Helpers ───────────────────────────────────────────────────────────

class _Tee:
    """Context manager that duplicates stdout to a file."""

    def __init__(self, filename):
        self.file = open(filename, 'w', encoding='utf-8')
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()


def _safe_classification_report(y_true, y_pred, id2name_dict):
    """Classification report restricted to classes present in the data."""
    present_classes = sorted(set(y_true) | set(y_pred))
    target_names = [id2name_dict.get(c, f"ID_{c}") for c in present_classes]
    return classification_report(
        y_true, y_pred,
        labels=present_classes, target_names=target_names,
        digits=3, zero_division=0,
    )


def _parse_v2_label(label):
    """Parse a v2 label like 'MONO_POS_CAUSE' into (role, polarity, salience).

    Returns ('UNK', 'UNK', 'UNK') for labels that don't follow the
    {salience}_{polarity}_{role} format (e.g. NO_RELATION, INTERDEPENDENCY).
    """
    parts = label.split("_")
    if len(parts) == 3:
        return parts[2], parts[1], parts[0]  # role, polarity, salience
    return "UNK", "UNK", "UNK"


# ═════════════════════════════════════════════════════════════════════
# RELATION EVALUATION
# ═════════════════════════════════════════════════════════════════════

def evaluate_relations(model, tokenizer, config, test_data, device, batch_size=32):
    """Run relation classification on test data and collect per-sample results.

    Returns a list of dicts, each containing ground-truth labels, predicted
    labels, and correctness flags for role, polarity, salience, and the
    reconstructed v2 label.  Works for both v2 and v3 architectures.
    """
    sep = PARALLEL_SEP_TOKEN
    rel_samples = [
        s for s in test_data
        if s.get("task") == "relation" and s.get("indicator") and s.get("entity")
    ]
    arch_version = getattr(config, "architecture_version", 2)
    results = []

    for i in range(0, len(rel_samples), batch_size):
        batch = rel_samples[i:i + batch_size]
        texts = [
            f"{s['indicator']} {sep} {s['entity']} {sep} {s['sentence']}"
            for s in batch
        ]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        with torch.no_grad():
            with torch.autocast(
                device_type="cuda", dtype=model.dtype,
                enabled=(str(device) == "cuda"),
            ):
                out = model(**inputs, task="relation")

        for j, sample in enumerate(batch):
            if arch_version >= 3:
                r = int(out["role_logits"].argmax(-1)[j])
                p = int(out["polarity_logits"].argmax(-1)[j])
                s = int(out["salience"].argmax(-1)[j])

                gt_role_name = ID2ROLE.get(sample["role"], f"UNK_{sample['role']}")
                gt_pol_name = ID2POLARITY.get(sample["polarity"], f"UNK_{sample['polarity']}")
                gt_sal_name = ID2SALIENCE.get(sample["salience"], "UNK")

                pred_sal_val = SALIENCE_VALUES[s]
                pred_v2 = convert_v3_to_v2_label(ID2ROLE[r], ID2POLARITY[p], pred_sal_val)

                results.append({
                    "sentence": sample["sentence"],
                    "indicator": sample["indicator"],
                    "entity": sample["entity"],
                    "gt_v2_label": sample.get("relation_label", ""),
                    "gt_role": sample["role"],
                    "gt_role_name": gt_role_name,
                    "gt_polarity": sample["polarity"],
                    "gt_polarity_name": gt_pol_name,
                    "gt_salience": sample["salience"],
                    "gt_salience_name": gt_sal_name,
                    "pred_v2_label": pred_v2,
                    "pred_role": r,
                    "pred_role_name": ID2ROLE[r],
                    "pred_polarity": p,
                    "pred_polarity_name": ID2POLARITY[p],
                    "pred_salience": s,
                    "pred_salience_name": ID2SALIENCE[s],
                    "role_correct": r == sample["role"],
                    "polarity_correct": p == sample["polarity"],
                    "salience_correct": s == sample["salience"],
                    "v2_correct": pred_v2 == sample.get("relation_label", ""),
                })
            else:
                # v2: single 14-class head
                logits = out["logits"]
                pred_id = int(logits.argmax(-1)[j])
                id2label_rel = {int(k): v for k, v in config.id2label_relation.items()}
                pred_label = id2label_rel[pred_id]
                gt_label = sample.get("relation_label", "")

                gt_role_name, gt_pol_name, gt_sal_name = _parse_v2_label(gt_label)
                pred_role_name, pred_pol_name, pred_sal_name = _parse_v2_label(pred_label)

                results.append({
                    "sentence": sample["sentence"],
                    "indicator": sample["indicator"],
                    "entity": sample["entity"],
                    "gt_v2_label": gt_label,
                    "gt_role": sample.get("role", -1),
                    "gt_role_name": gt_role_name,
                    "gt_polarity": sample.get("polarity", -1),
                    "gt_polarity_name": gt_pol_name,
                    "gt_salience": sample.get("salience", -1),
                    "gt_salience_name": gt_sal_name,
                    "pred_v2_label": pred_label,
                    "pred_role": -1,
                    "pred_role_name": pred_role_name,
                    "pred_polarity": -1,
                    "pred_polarity_name": pred_pol_name,
                    "pred_salience": -1,
                    "pred_salience_name": pred_sal_name,
                    "role_correct": gt_role_name == pred_role_name,
                    "polarity_correct": gt_pol_name == pred_pol_name,
                    "salience_correct": gt_sal_name == pred_sal_name,
                    "v2_correct": pred_label == gt_label,
                })

    return results


# ═════════════════════════════════════════════════════════════════════
# TOKEN EVALUATION
# ═════════════════════════════════════════════════════════════════════

def evaluate_tokens(model, tokenizer, config, test_data, device, batch_size=32):
    """Run token classification on test data and collect per-token results.

    Evaluates at first-subword positions only (consistent with training).
    Returns a dict with ground-truth/predicted label lists and per-type
    accuracy counts.
    """
    token_samples = [s for s in test_data if s.get("task") == "token"]
    id2label = {int(k): v for k, v in config.id2label_span.items()}
    all_gt, all_pred = [], []
    total_ent, ent_corr, total_ind, ind_corr = 0, 0, 0, 0

    for i in range(0, len(token_samples), batch_size):
        batch = token_samples[i:i + batch_size]
        inputs = tokenizer(
            [s["sentence"] for s in batch],
            padding=True, truncation=True, return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = model(**inputs, task="token")
        pred_ids = out["logits"].argmax(-1)

        for j, sample in enumerate(batch):
            enc = tokenizer(
                sample["sentence"],
                return_offsets_mapping=True, truncation=True,
            )
            offsets = enc.encodings[0].offsets
            tokens = enc.encodings[0].tokens
            word_ids = enc.encodings[0].word_ids
            gt_bio = ["O"] * len(enc["input_ids"])

            # Mark first-subword positions
            first_subword = set()
            prev_wid = None
            for k, wid in enumerate(word_ids):
                if wid is not None and wid != prev_wid:
                    first_subword.add(k)
                prev_wid = wid

            # Build gold BIOES labels
            for span in sample["spans"]:
                start, end, stype = (
                    (span["start"], span["end"], span["type"])
                    if isinstance(span, dict) else span
                )
                hit_indices = []
                for k, (ts, te) in enumerate(offsets):
                    if k not in first_subword:
                        continue
                    adj_ts = ts
                    tok = tokens[k]
                    if (tok.startswith("\u0120") or tok.startswith("\u2581")) and ts < te:
                        adj_ts = ts + 1
                    if max(start, adj_ts) < min(end, te):
                        if gt_bio[k] != "O":
                            continue
                        hit_indices.append(k)

                if len(hit_indices) == 1:
                    gt_bio[hit_indices[0]] = f"S-{stype}"
                elif len(hit_indices) > 1:
                    gt_bio[hit_indices[0]] = f"B-{stype}"
                    for idx in hit_indices[1:-1]:
                        gt_bio[idx] = f"I-{stype}"
                    gt_bio[hit_indices[-1]] = f"E-{stype}"

            # Collect predictions at first-subword positions
            for k in range(1, min(len(gt_bio), pred_ids.shape[1]) - 1):
                if k not in first_subword:
                    continue
                gt_l = gt_bio[k]
                pr_l = id2label.get(int(pred_ids[j, k]), "O")
                all_gt.append(gt_l)
                all_pred.append(pr_l)

                gt_type = gt_l.split("-")[-1] if "-" in gt_l else gt_l
                if gt_type == "ENTITY":
                    total_ent += 1
                    if pr_l == gt_l:
                        ent_corr += 1
                elif gt_type == "INDICATOR":
                    total_ind += 1
                    if pr_l == gt_l:
                        ind_corr += 1

    return {
        "gt": all_gt, "pred": all_pred,
        "total_ent": total_ent, "ent_corr": ent_corr,
        "total_ind": total_ind, "ind_corr": ind_corr,
    }


# ═════════════════════════════════════════════════════════════════════
# SPAN-LEVEL TOKEN ERROR ANALYSIS
# ═════════════════════════════════════════════════════════════════════

def _bio_to_spans(bio_labels, offsets, sentence, tokens=None):
    """Convert BIOES tag sequence to a list of (start_char, end_char, type, text).

    Applies the Ġ-prefix offset correction (matching the collator) so that
    reconstructed spans align with gold character spans from dataset.py.
    """
    spans = []
    cur_start = cur_end = None
    cur_type = None

    for i, tag in enumerate(bio_labels):
        ts, te = offsets[i]

        if tokens is not None and ts < te:
            tok = tokens[i]
            if tok.startswith("\u0120") or tok.startswith("\u2581"):
                ts += 1

        if tag.startswith("S-"):
            if cur_start is not None:
                spans.append((cur_start, cur_end, cur_type, sentence[cur_start:cur_end]))
                cur_start = cur_end = cur_type = None
            spans.append((ts, te, tag[2:], sentence[ts:te]))

        elif tag.startswith("B-"):
            if cur_start is not None:
                spans.append((cur_start, cur_end, cur_type, sentence[cur_start:cur_end]))
            cur_type = tag[2:]
            cur_start, cur_end = ts, te

        elif tag.startswith("I-"):
            tag_type = tag[2:]
            if cur_start is not None and tag_type == cur_type:
                cur_end = max(cur_end, te)
            else:
                if cur_start is not None:
                    spans.append((cur_start, cur_end, cur_type, sentence[cur_start:cur_end]))
                cur_type = tag_type
                cur_start, cur_end = ts, te

        elif tag.startswith("E-"):
            tag_type = tag[2:]
            if cur_start is not None and tag_type == cur_type:
                cur_end = max(cur_end, te)
                spans.append((cur_start, cur_end, cur_type, sentence[cur_start:cur_end]))
                cur_start = cur_end = cur_type = None
            else:
                if cur_start is not None:
                    spans.append((cur_start, cur_end, cur_type, sentence[cur_start:cur_end]))
                spans.append((ts, te, tag_type, sentence[ts:te]))
                cur_start = cur_end = cur_type = None

        else:  # O
            if cur_start is not None:
                spans.append((cur_start, cur_end, cur_type, sentence[cur_start:cur_end]))
                cur_start = cur_end = cur_type = None

    if cur_start is not None:
        spans.append((cur_start, cur_end, cur_type, sentence[cur_start:cur_end]))

    return spans


def _get_gold_spans(sample):
    """Extract gold spans from a dataset sample."""
    spans = []
    sentence = sample["sentence"]
    for span in sample.get("spans", []):
        if isinstance(span, dict):
            s, e, t = span["start"], span["end"], span["type"]
        else:
            s, e, t = span
        spans.append((s, e, t, sentence[s:e]))
    return spans


def _get_pred_spans(model, tokenizer, sentence, device, id2label):
    """Run model on a single sentence and return predicted spans.

    Predictions are aggregated to word level: only the first subword of each
    word receives a label, and the span covers all subwords of that word.
    """
    enc = tokenizer(
        sentence, return_offsets_mapping=True, truncation=True, return_tensors="pt",
    )
    offsets = enc.encodings[0].offsets
    tokens = enc.encodings[0].tokens
    word_ids = enc.encodings[0].word_ids
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        with torch.autocast(
            device_type="cuda", dtype=model.dtype,
            enabled=(str(device) == "cuda"),
        ):
            out = model(input_ids=input_ids, attention_mask=attention_mask, task="token")

    pred_ids = out["logits"].argmax(-1)[0].cpu().tolist()

    # Build word-level offsets: span from first subword start to last subword end
    word_offsets = {}
    for i, wid in enumerate(word_ids):
        if wid is None:
            continue
        ts, te = offsets[i]
        if wid not in word_offsets:
            word_offsets[wid] = [ts, te]
        else:
            word_offsets[wid][1] = max(word_offsets[wid][1], te)

    # Keep predictions at first-subword positions only, with full word offsets
    bio = []
    word_level_offsets = []
    prev_wid = None
    for i, pid in enumerate(pred_ids):
        wid = word_ids[i] if i < len(word_ids) else None
        if wid is not None and wid != prev_wid:
            bio.append(id2label.get(pid, "O"))
            wo = word_offsets[wid]
            word_level_offsets.append((wo[0], wo[1]))
        else:
            bio.append("O")
            word_level_offsets.append(offsets[i] if i < len(offsets) else (0, 0))
        prev_wid = wid

    return _bio_to_spans(bio, word_level_offsets, sentence, tokens=tokens)


def _overlap_iou(g, p):
    """IoU between two character spans (ignores type)."""
    o_start = max(g[0], p[0])
    o_end = min(g[1], p[1])
    if o_start >= o_end:
        return 0.0
    union = max(g[1], p[1]) - min(g[0], p[0])
    return (o_end - o_start) / union if union > 0 else 0.0


def _match_spans(gold_spans, pred_spans):
    """Greedy span matching: exact first, then best-IoU partial.

    Returns:
        matched: List of (gold, pred, match_type_str) tuples.
        missed: Gold spans with no matching prediction.
        spurious: Predicted spans with no matching gold.
    """
    matched = []
    used_pred = set()

    # Pass 1: strict (exact char boundaries + type)
    for g in gold_spans:
        for j, p in enumerate(pred_spans):
            if j not in used_pred and g[0] == p[0] and g[1] == p[1] and g[2] == p[2]:
                matched.append((g, p, "exact"))
                used_pred.add(j)
                break

    # Pass 2: partial (same type, any overlap)
    matched_gold = {id(m[0]) for m in matched}
    for g in gold_spans:
        if id(g) in matched_gold:
            continue
        best_j, best_iou = None, 0.0
        for j, p in enumerate(pred_spans):
            if j not in used_pred and g[2] == p[2]:
                iou = _overlap_iou(g, p)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
        if best_j is not None and best_iou > 0:
            matched.append((g, pred_spans[best_j], f"partial(iou={best_iou:.2f})"))
            used_pred.add(best_j)

    missed = [g for g in gold_spans if id(g) not in {id(m[0]) for m in matched}]
    spurious = [p for j, p in enumerate(pred_spans) if j not in used_pred]
    return matched, missed, spurious


def token_error_analysis(
    model_path, test_dir, output_dir="token_error_output",
    _model=None, _tokenizer=None, _config=None, _device=None,
    _test_data=None,
):
    """Span-level error analysis for token classification.

    Evaluates predicted spans against gold spans using strict (exact boundary)
    and partial (IoU overlap) matching.  Produces a text report, a detailed
    JSON export, and an annotation-review JSON with missed/spurious form
    frequencies.

    Can be called standalone or with a pre-loaded model::

        # Standalone
        token_error_analysis("path/to/model", "path/to/test_dataset")

        # With pre-loaded model (avoids redundant loading)
        token_error_analysis(
            "", "", _model=model, _tokenizer=tokenizer,
            _config=config, _device=device, _test_data=test_data,
        )
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "token_span_report.txt")
    json_path = os.path.join(output_dir, "token_errors_detail.json")
    review_path = os.path.join(output_dir, "for_annotation_review.json")

    if _model is None:
        _model, _tokenizer, _config, _device = load_model(model_path)
    if _test_data is None:
        test_ds = load_from_disk(test_dir)
        _test_data = [test_ds[i] for i in range(len(test_ds))]

    assert _config is not None and _tokenizer is not None and _device is not None

    id2label = {int(k): v for k, v in _config.id2label_span.items()}
    token_samples = [s for s in _test_data if s.get("task") == "token"]

    with _Tee(report_path):
        print(f"Loaded model on {_device}")
        print(f"Token samples: {len(token_samples)}\n")

        counts = {
            t: {
                "tp_strict": 0, "tp_partial": 0, "fn": 0, "fp": 0,
                "total_gold": 0, "total_pred": 0,
            }
            for t in ("ENTITY", "INDICATOR")
        }

        missed_entities = []
        missed_indicators = []
        spurious_entities = []
        spurious_indicators = []
        boundary_errors = []
        type_confusions = []
        all_results = []

        for sample in token_samples:
            sentence = sample["sentence"]
            gold_spans = _get_gold_spans(sample)
            pred_spans = _get_pred_spans(_model, _tokenizer, sentence, _device, id2label)
            matched, missed, spurious = _match_spans(gold_spans, pred_spans)

            for span_type in ("ENTITY", "INDICATOR"):
                c = counts[span_type]
                c["total_gold"] += sum(1 for g in gold_spans if g[2] == span_type)
                c["total_pred"] += sum(1 for p in pred_spans if p[2] == span_type)

                for g, p, mtype in matched:
                    if g[2] == span_type:
                        if mtype == "exact":
                            c["tp_strict"] += 1
                            c["tp_partial"] += 1
                        else:
                            c["tp_partial"] += 1
                            boundary_errors.append({
                                "type": span_type,
                                "gold_text": g[3], "pred_text": p[3],
                                "gold_span": (g[0], g[1]),
                                "pred_span": (p[0], p[1]),
                                "match": mtype, "sentence": sentence,
                            })

                for g in missed:
                    if g[2] == span_type:
                        c["fn"] += 1
                        dst = missed_entities if span_type == "ENTITY" else missed_indicators
                        dst.append({"text": g[3], "span": (g[0], g[1]), "sentence": sentence})

            for p in spurious:
                if p[2] in counts:
                    counts[p[2]]["fp"] += 1
                    dst = spurious_entities if p[2] == "ENTITY" else spurious_indicators
                    dst.append({"text": p[3], "span": (p[0], p[1]), "sentence": sentence})

            for g in gold_spans:
                for p in pred_spans:
                    if g[2] != p[2] and _overlap_iou(g, p) > 0.5:
                        type_confusions.append({
                            "gold_type": g[2], "pred_type": p[2],
                            "gold_text": g[3], "pred_text": p[3],
                            "sentence": sentence,
                        })

            all_results.append({
                "sentence": sentence,
                "gold_spans": [(g[0], g[1], g[2], g[3]) for g in gold_spans],
                "pred_spans": [(p[0], p[1], p[2], p[3]) for p in pred_spans],
                "matched": [(g[3], p[3], mt) for g, p, mt in matched],
                "missed": [(g[2], g[3]) for g in missed],
                "spurious": [(p[2], p[3]) for p in spurious],
            })

        # ── Report ────────────────────────────────────────────────────
        print("=" * 70)
        print("SPAN-LEVEL TOKEN ERROR ANALYSIS")
        print("=" * 70)

        for span_type in ("ENTITY", "INDICATOR"):
            c = counts[span_type]
            print(f"\n── {span_type} Spans ──")
            print(f"  Gold spans:    {c['total_gold']}")
            print(f"  Pred spans:    {c['total_pred']}")

            p_s = c["tp_strict"] / c["total_pred"] if c["total_pred"] else 0
            r_s = c["tp_strict"] / c["total_gold"] if c["total_gold"] else 0
            f1_s = 2 * p_s * r_s / (p_s + r_s) if (p_s + r_s) else 0
            print(f"  Strict  — P: {p_s:.3f}  R: {r_s:.3f}  F1: {f1_s:.3f}")

            p_p = c["tp_partial"] / c["total_pred"] if c["total_pred"] else 0
            r_p = c["tp_partial"] / c["total_gold"] if c["total_gold"] else 0
            f1_p = 2 * p_p * r_p / (p_p + r_p) if (p_p + r_p) else 0
            print(f"  Partial — P: {p_p:.3f}  R: {r_p:.3f}  F1: {f1_p:.3f}")

            print(f"  Missed (FN):   {c['fn']}")
            print(f"  Spurious (FP): {c['fp']}")

        print(f"\n── Boundary & Type Diagnostics ──")
        print(f"  Boundary errors (partial matches): {len(boundary_errors)}")
        print(f"  Type confusions (ENTITY↔INDICATOR): {len(type_confusions)}")

        missed_ent_texts = Counter(m["text"] for m in missed_entities)
        missed_ind_texts = Counter(m["text"] for m in missed_indicators)
        spurious_ent_texts = Counter(s["text"] for s in spurious_entities)
        spurious_ind_texts = Counter(s["text"] for s in spurious_indicators)

        print(f"\n── Top 20 Missed Entities (FN={len(missed_entities)}) ──")
        for text, count in missed_ent_texts.most_common(20):
            print(f"  {count:>3}x  '{text}'")

        print(f"\n── Top 20 Missed Indicators (FN={len(missed_indicators)}) ──")
        for text, count in missed_ind_texts.most_common(20):
            print(f"  {count:>3}x  '{text}'")

        print(f"\n── Top 20 Spurious Entities (FP={len(spurious_entities)}) ──")
        for text, count in spurious_ent_texts.most_common(20):
            print(f"  {count:>3}x  '{text}'")

        print(f"\n── Top 20 Spurious Indicators (FP={len(spurious_indicators)}) ──")
        for text, count in spurious_ind_texts.most_common(20):
            print(f"  {count:>3}x  '{text}'")

        print(f"\n── Sample Boundary Errors ({len(boundary_errors)} total) ──")
        for err in boundary_errors[:15]:
            print(f"  [{err['type']}] gold='{err['gold_text']}' → pred='{err['pred_text']}' ({err['match']})")
            print(f"           {err['sentence'][:120]}")

        if type_confusions:
            print(f"\n── Type Confusions ({len(type_confusions)} total) ──")
            for tc in type_confusions[:10]:
                print(f"  gold={tc['gold_type']}('{tc['gold_text']}') → pred={tc['pred_type']}('{tc['pred_text']}')")
                print(f"           {tc['sentence'][:120]}")

        error_counts = [
            (len(r["missed"]) + len(r["spurious"]), r)
            for r in all_results
            if r["missed"] or r["spurious"]
        ]
        error_counts.sort(key=lambda x: -x[0])

        print(f"\n── Sentences with Most Errors (top 10) ──")
        for n_err, r in error_counts[:10]:
            print(f"\n  ({n_err} errors) {r['sentence'][:140]}")
            for t, txt in r["missed"]:
                print(f"    MISSED   {t}: '{txt}'")
            for t, txt in r["spurious"]:
                print(f"    SPURIOUS {t}: '{txt}'")

        print(f"\n── Files ──")
        print(f"  Report:  {report_path}")
        print(f"  Detail:  {json_path}")
        print(f"  Review:  {review_path}")

    # Save JSON exports (outside the Tee context)
    export_data = {
        "summary": {k: dict(v) for k, v in counts.items()},
        "missed_entities": missed_entities[:200],
        "missed_indicators": missed_indicators[:200],
        "spurious_entities": spurious_entities[:200],
        "spurious_indicators": spurious_indicators[:200],
        "boundary_errors": boundary_errors[:100],
        "type_confusions": type_confusions[:100],
        "worst_sentences": [
            {
                "sentence": r["sentence"], "gold": r["gold_spans"],
                "pred": r["pred_spans"], "missed": r["missed"],
                "spurious": r["spurious"],
            }
            for _, r in error_counts[:50]
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    review_data = {
        "missed_indicator_forms": dict(missed_ind_texts.most_common()),
        "missed_entity_forms": dict(missed_ent_texts.most_common()),
        "spurious_indicator_forms": dict(spurious_ind_texts.most_common()),
        "spurious_entity_forms": dict(spurious_ent_texts.most_common()),
    }
    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(review_data, f, indent=2, ensure_ascii=False)

    return counts, all_results


# ═════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════

def error_analysis(
    model_path,
    test_dir,
    output_dir="error_analysis_output",
    random_seed=42,
    n_error_samples=100,
):
    """Full error analysis for a C-BERT model (relations + tokens).

    Loads a model and test dataset, then runs:
    1. Relation classification evaluation with per-head breakdown
       (role, polarity, salience accuracy; confusion matrices; error patterns).
    2. Token classification evaluation with per-token and per-type reports.

    All output is printed to console and simultaneously written to a report
    file.  Detailed per-sample results are saved as JSON.

    Args:
        model_path: Path to the saved C-BERT model directory.
        test_dir: Path to the test dataset (HuggingFace Datasets format).
        output_dir: Directory for output files.
        random_seed: Seed for reproducible error sampling.
        n_error_samples: Number of error samples to include in the report.

    Output files:
        validation_report.txt: Full text report.
        relation_errors.json: Per-sample relation predictions and labels.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "validation_report.txt")
    json_path = os.path.join(output_dir, "relation_errors.json")

    with _Tee(report_path):
        model, tokenizer, config, device = load_model(model_path)
        print(f"Loaded C-BERT on {device}\n")

        test_ds = load_from_disk(test_dir)
        test_data = [test_ds[i] for i in range(len(test_ds))]

        n_token = sum(1 for s in test_data if s.get("task") == "token")
        n_rel = sum(1 for s in test_data if s.get("task") == "relation")

        # ── 1. Relation evaluation ────────────────────────────────────
        if n_rel > 0:
            rel_results = evaluate_relations(model, tokenizer, config, test_data, device)

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(rel_results, f, indent=2, ensure_ascii=False)

            total = len(rel_results)
            errors = [r for r in rel_results if not r["v2_correct"]]
            correct_count = total - len(errors)

            print("=" * 70)
            print("RELATION CLASSIFICATION ERROR ANALYSIS")
            print("=" * 70)
            print(f"Total samples:    {total}")
            print(f"Correct (v2):     {correct_count} ({correct_count / total * 100:.1f}%)")
            print(f"Errors (v2):      {len(errors)} ({len(errors) / total * 100:.1f}%)")

            y_gt_r = [r["gt_role"] for r in rel_results]
            y_pr_r = [r["pred_role"] for r in rel_results]
            y_gt_p = [r["gt_polarity"] for r in rel_results]
            y_pr_p = [r["pred_polarity"] for r in rel_results]
            y_gt_s = [r["gt_salience"] for r in rel_results]
            y_pr_s = [r["pred_salience"] for r in rel_results]

            print(f"\n── Per-Head Accuracy ──")
            print(f"  Role:      {accuracy_score(y_gt_r, y_pr_r) * 100:.1f}%")
            print(f"  Polarity:  {accuracy_score(y_gt_p, y_pr_p) * 100:.1f}%")
            print(f"  Salience:  {accuracy_score(y_gt_s, y_pr_s) * 100:.1f}%")
            print(f"  Reconstr accuracy:  {accuracy_score([r['gt_v2_label'] for r in rel_results], [r['pred_v2_label'] for r in rel_results]) * 100:.1f}%")
            print(f"  Reconstr macro F1:  {f1_score([r['gt_v2_label'] for r in rel_results], [r['pred_v2_label'] for r in rel_results], average='macro', zero_division=0) * 100:.1f}%")

            print(f"\n── Role Classification Report ──")
            print(_safe_classification_report(y_gt_r, y_pr_r, ID2ROLE))

            print(f"\n── Polarity Classification Report ──")
            print(_safe_classification_report(y_gt_p, y_pr_p, ID2POLARITY))

            print(f"\n── Salience Classification Report ──")
            print(_safe_classification_report(y_gt_s, y_pr_s, ID2SALIENCE))

            # Confusion matrix
            print(f"\n── Salience Confusion Matrix (DIST / PRIO / MONO) ──")
            cm = confusion_matrix(y_gt_s, y_pr_s, labels=[0, 1, 2])
            print(f"{'':>14} {'DIST':>8} {'PRIO':>8} {'MONO':>8}")
            for i, label in enumerate(["DIST", "PRIO", "MONO"]):
                print(f"{label:>14} {cm[i, 0]:>8} {cm[i, 1]:>8} {cm[i, 2]:>8}")

            # Error breakdown
            if errors:
                role_only = sum(1 for r in errors if not r["role_correct"] and r["polarity_correct"] and r["salience_correct"])
                pol_only = sum(1 for r in errors if r["role_correct"] and not r["polarity_correct"] and r["salience_correct"])
                sal_only = sum(1 for r in errors if r["role_correct"] and r["polarity_correct"] and not r["salience_correct"])
                print(f"\n── Error Type Breakdown ({len(errors)} total errors) ──")
                print(f"  Role only:          {role_only:>4} ({role_only / len(errors) * 100:.1f}%)")
                print(f"  Polarity only:      {pol_only:>4} ({pol_only / len(errors) * 100:.1f}%)")
                print(f"  Salience only:      {sal_only:>4} ({sal_only / len(errors) * 100:.1f}%)")

                role_pats = Counter(
                    (r["gt_role_name"], r["pred_role_name"])
                    for r in errors if not r["role_correct"]
                )
                print(f"\n── Role Confusion Patterns ({sum(role_pats.values())} errors) ──")
                for (gt, pr), count in role_pats.most_common():
                    print(f"  {gt} → {pr}: {count}")

            # Sampled errors
            np.random.seed(random_seed)
            sampled = (
                np.random.choice(len(errors), min(n_error_samples, len(errors)), replace=False)
                if errors else []
            )
            print(f"\n── Sampled Errors ({len(sampled)} of {len(errors)}) ──")
            for idx in sampled[:10]:
                r = errors[idx]
                print(f"\n  Sentence:  {r['sentence']}")
                print(f"  Indicator: '{r['indicator']}'  Entity: '{r['entity']}'")
                print(f"  GT:        {r['gt_v2_label']} (role={r['gt_role_name']}, pol={r['gt_polarity_name']}, sal={r['gt_salience_name']})")
                print(f"  Pred:      {r['pred_v2_label']} (role={r['pred_role_name']}, pol={r['pred_polarity_name']}, sal={r['pred_salience_name']})")

            # Salience-specific errors
            sal_errors = [r for r in errors if r["role_correct"] and not r["salience_correct"]]
            if sal_errors:
                print(f"\n── Salience Errors ({len(sal_errors)} total, role correct but salience wrong) ──")
                sal_sampled = np.random.choice(
                    len(sal_errors), min(20, len(sal_errors)), replace=False,
                )
                for idx in sal_sampled:
                    r = sal_errors[idx]
                    print(f"\n  Sentence:  {r['sentence']}")
                    print(f"  Indicator: '{r['indicator']}'  Entity: '{r['entity']}'")
                    print(f"  GT:        {r['gt_v2_label']} (sal={r['gt_salience_name']})")
                    print(f"  Pred:      {r['pred_v2_label']} (sal={r['pred_salience_name']})")

        # ── 2. Token evaluation ───────────────────────────────────────
        if n_token > 0:
            tr = evaluate_tokens(model, tokenizer, config, test_data, device)
            print(f"\n{'=' * 70}\nTOKEN CLASSIFICATION ERROR ANALYSIS\n{'=' * 70}")
            print(f"Overall token accuracy: {accuracy_score(tr['gt'], tr['pred']) * 100:.1f}%")
            if tr["total_ent"] > 0:
                print(f"Entity token accuracy:    {tr['ent_corr'] / tr['total_ent'] * 100:.1f}% ({tr['ent_corr']}/{tr['total_ent']})")
            if tr["total_ind"] > 0:
                print(f"Indicator token accuracy: {tr['ind_corr'] / tr['total_ind'] * 100:.1f}% ({tr['ind_corr']}/{tr['total_ind']})")
            print(f"\n── Token Classification Report ──")
            print(classification_report(tr["gt"], tr["pred"], digits=3, zero_division=0))

            gt_type = [l.split("-")[-1] for l in tr["gt"]]
            pr_type = [l.split("-")[-1] for l in tr["pred"]]
            print(f"\n── Type-level Report (ignoring B/I/E/S prefix) ──")
            print(classification_report(gt_type, pr_type, digits=3, zero_division=0))

    print(f"\nDone. Report: {report_path}")
    print(f"JSON data: {json_path}")