import argparse
import logging
from causalbert.infer import load_model, sentence_analysis, clean_tok

logger = logging.getLogger(__name__)

def main(model_path, sentences, batch_size):
    """Loads the model and runs analysis on provided sentences."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    num_sentences_with_relations = 0
    model, tokenizer, config, device = load_model(model_path)
    all_analyses = sentence_analysis(model, tokenizer, config, sentences, device=device, batch_size=batch_size)
    
    for analysis in all_analyses:
        sentence = analysis['sentence']
        
        if analysis["derived_relations"]:
            num_sentences_with_relations += 1
            
            print(f"\n\n📌 **Sentence with Relation:** {sentence}\n")
          
            # Print token results with confidence
            print("🧠 Token Classification (with confidence):")
            for prediction in analysis["token_predictions"]:
                token = clean_tok(prediction["token"])
                label = prediction["label"]
                conf = prediction["confidence"]
                print(f"{token:15} {label:20} {conf:.4f}")
            
            # Print relation results
            print("\n🔗 Relation Classification (auto-derived pairs):")
            for (indicator, entity), rel in analysis["derived_relations"]:
                indicator = clean_tok(indicator)
                entity = clean_tok(entity)
                print(f"🔸 {indicator:15} {entity:15} ➜ {rel['label']} ({rel['confidence']:.4f})")
        else:
            print(f"📄 Sentence: {sentence}")
        print(f"---\n")

    total_sentences_queried = len(all_analyses)
    if total_sentences_queried > 0:
        percentage_with_relations = (num_sentences_with_relations / total_sentences_queried) * 100
        print(f"\n\n### Inference Summary ###")
        print(f"Total sentences queried: {total_sentences_queried}")
        print(f"Sentences with derived relations: {num_sentences_with_relations}")
        print(f"Percentage with relations: {percentage_with_relations:.2f}%")
    else:
        print("\nNo sentences were queried from the database.")
        
        print(f"\nCompleted analysis on {len(sentences)} sentences.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CBERT inference on provided sentences.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model directory (e.g., ../model/C-EBERT_P).")
    parser.add_argument("--sentences", nargs='+', required=True, help="List of sentences to analyze, separated by spaces.")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size.")
    
    args = parser.parse_args()

    main(args.model_path, args.sentences, args.batch_size)