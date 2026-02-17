"""Demo script for C-BERT inference.

Loads a trained C-BERT model and runs the full extraction pipeline
(span recognition → relation classification → tuple construction)
on a list of example sentences.

Usage::

    python -m causalbert.run_inference

Or from Python::

    from causalbert.run_inference import CausalAnalyzer
    analyzer = CausalAnalyzer("pdjohn/C-EBERT-V3-610m")
    results = analyzer.run_inference(["Industrie stoppt Arten."])
"""

import os
import logging
from causalbert.infer import load_model, sentence_analysis
from causalbert.utils import clean_tok
from causalbert.model import ID2SALIENCE, SALIENCE_VALUES


class CausalAnalyzer:
    """Convenience wrapper for loading a C-BERT model and running inference."""

    def __init__(self, model_dir, log_dir="log/causalbert"):
        self.model_dir = model_dir
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(levelname)s: %(name)s: %(message)s",
            filename=os.path.join(self.log_dir, "inference.txt"),
            filemode="w",
        )

        print(f"Loading model from {model_dir}...")
        self.model, self.tokenizer, self.config, self.device = load_model(model_dir)
        self.arch_version = getattr(self.config, "architecture_version", 2)
        print(f"Loaded C-BERT v{self.arch_version} model on {self.device}\n")

    @staticmethod
    def format_influence(influence_val):
        if influence_val is None:
            return "n/a"
        bar_len = int(abs(influence_val) * 10)
        symbol = "+" if influence_val >= 0 else "-"
        bar = symbol + "\u2588" * bar_len + "\u2591" * (10 - bar_len)
        return f"{influence_val:+.2f} {bar}"

    @staticmethod
    def format_role(role_name, confidence):
        role_icons = {"CAUSE": "\U0001F53A", "EFFECT": "\U0001F53B", "NO_RELATION": "\u26AA"}
        icon = role_icons.get(role_name, "\u2753")
        return f"{icon} {role_name} ({confidence:.2f})"

    @staticmethod
    def format_polarity(polarity, confidence):
        icon = "\u2795" if polarity == "POS" else "\u2796"
        return f"{icon} {polarity} ({confidence:.2f})"

    @staticmethod
    def format_salience(salience_val):
        if salience_val is None:
            return "n/a"
        try:
            sal_id = SALIENCE_VALUES.index(salience_val)
        except ValueError:
            return f"{salience_val:.2f} (unknown)"
        label = ID2SALIENCE[sal_id]
        bar_len = int(salience_val * 10)
        bar = "\u2588" * bar_len + "\u2591" * (10 - bar_len)
        return f"{salience_val:.2f} {bar} ({label})"

    def run_inference(self, sentences, batch_size=8):
        """Run the full C-BERT pipeline and print results."""
        all_analyses = sentence_analysis(
            self.model, self.tokenizer, self.config,
            sentences, device=self.device, batch_size=batch_size,
        )
        self._print_results(all_analyses)
        return all_analyses

    def _print_results(self, all_analyses):
        total_relations = 0

        print("=" * 80)
        print(f"C-BERT v{self.arch_version} INFERENCE RESULTS")
        print("=" * 80)

        for idx, analysis in enumerate(all_analyses, 1):
            sentence = analysis["sentence"]
            if not analysis["derived_relations"]:
                print(f"\n[{idx}] {sentence}\nNo causal relations detected.")
                continue

            print(f"\n{'\u2500' * 80}\n[{idx}] {sentence}\n{'\u2500' * 80}")

            # Token classification
            print("\nToken Classification:")
            for pred in analysis["token_predictions"]:
                token = clean_tok(pred["token"])
                label, conf = pred["label"], pred["confidence"]
                mark = " \u25C0" if label != "O" else ""
                print(f"   {token:<20} {label:<15} {conf:>6.2f}{mark}")

            # Relations
            print(f"\nRelations ({len(analysis['derived_relations'])} found):")

            if self.arch_version == 2:
                for (ind, ent), rel in analysis["derived_relations"]:
                    print(f"   {clean_tok(ind):<15} {clean_tok(ent):<20} "
                          f"{rel['label']:<25} {rel['confidence']:>6.2f}")
                    total_relations += 1
            else:
                causes_by_ind, effects_by_ind = {}, {}
                print(f"   {'Indicator':<15} {'Entity':<18} {'Role':<18} "
                      f"{'Polarity':<16} {'Salience':<28} {'I':>6}  {'Label'}")

                for (ind, ent), rel in analysis["derived_relations"]:
                    ind_c, ent_c = clean_tok(ind), clean_tok(ent)
                    role = rel.get("role")
                    polarity = rel.get("polarity", "POS")
                    pol_conf = rel.get("polarity_confidence", 0)
                    salience = rel.get("salience", 0)
                    influence = rel.get("influence", 0)

                    print(f"   {ind_c:<15} {ent_c:<18} "
                          f"{self.format_role(role, rel.get('role_confidence', 0)):<18} "
                          f"{self.format_polarity(polarity, pol_conf):<16} "
                          f"{self.format_salience(salience):<28} "
                          f"{influence:>+5.2f}  {rel['label']}")

                    target_dict = causes_by_ind if role == "CAUSE" else effects_by_ind
                    target_dict.setdefault(ind_c, []).append(
                        (ent_c, influence, polarity, salience)
                    )
                    total_relations += 1

                self._print_tuples(causes_by_ind, effects_by_ind)

        print(f"\n{'=' * 80}\nINFERENCE SUMMARY\n{'=' * 80}")
        print(f"   Total sentences: {len(all_analyses)}")
        print(f"   Total relations: {total_relations}")

    def _print_tuples(self, causes, effects):
        tuples = []
        for ind in causes:
            if ind in effects:
                for c_ent, c_pol, c_sal in causes[ind]:
                    for e_ent, e_inf, e_pol, e_sal in effects[ind]:
                        tuple_i = e_inf
                        tuples.append((c_ent, e_ent, tuple_i, c_pol, c_sal, e_pol, e_sal))

        if tuples:
            print(f"\nCausal Tuples (C, E, I):")
            for c, e, ti, c_pol, c_sal, e_pol, e_sal, flag in tuples:
                pol_icon = "\u2795" if e_pol == "POS" else "\u2796"
                print(f"      {c:<20} {e:<20} {ti:>+7.2f}  "
                      f"[{pol_icon} {e_pol}, sal: C={c_sal:.2f} E={e_sal:.2f}]{flag}")


def main():
    DEFAULT_MODEL = "pdjohn/C-EBERT-V3-610m"

    test_sentences = [
        "Industrie stoppt Arten.",
        "Mehr Industrie verursacht den Schwund von Arten.",
        "Weniger Pestizide f\u00FChren zu mehr Artenvielfalt.",
        "Autos sind die Hauptursache des Klimawandels.",
    ]

    analyzer = CausalAnalyzer(model_dir=DEFAULT_MODEL)
    analyzer.run_inference(test_sentences, batch_size=8)


if __name__ == "__main__":
    main()