"""Run evaluation on dataset

Format: python3 run_evaluation.py --model ... --dataset ... --fewshot-dataset ... --output_file_name ...
"""
import argparse
from evaluate import evaluate_all, evaluate_baseline
from datasets.abstract_dataset_loader import DatasetLoader
from models.abstract_model import AbstractModel
from models.prompt_generator import PromptGenerator
import sys
from datetime import datetime
from postprocess import postprocess_all, postprocess_all_baseline
import json

def main():
    parser = argparse.ArgumentParser(description="Process model and dataset flags.")
    parser.add_argument('--model', type=str, help='Model to use. Possible options: ' + ', '.join(AbstractModel.registered_models) + '.')
    parser.add_argument('--dataset', type=str, help='Dataset to use. Possible options: ' + ', '.join(DatasetLoader.registered_datasets) + '.')
    parser.add_argument('--fewshot-dataset', type=str, help='Dataset to use to collect few-shot examples. Possible options: ' + ', '.join(DatasetLoader.registered_datasets) + '.', default="morehopqa")
    parser.add_argument('--strategy', type=str, help="Prompting strategy to use. Possible options: zeroshot, zeroshot-cot, 2-shot, 2-shot-cot, 3-shot, 3-shot-cot")
    parser.add_argument('--output_file', type=str, help='First part of the name of the output file. Will also include model, strategy, dataset and timestamp. Default: output')

    args = parser.parse_args()

    if args.model is None or args.dataset is None or args.strategy is None:
        print("Missing arguments. Here are the possible options:")
        print("For --model: " + ", ".join(AbstractModel.registered_models))
        print("For --dataset: " + ", ".join(DatasetLoader.registered_datasets))
        print("For --strategy: zeroshot, zeroshot-cot, 2-shot, 2-shot-cot, 3-shot, 3-shot-cot")
        sys.exit(1)

    dataset = DatasetLoader.create(args.dataset)
    fewshot_dataset = DatasetLoader.create(args.fewshot_dataset)
    prompt_generator = PromptGenerator.create(args.strategy, fewshot_dataset)
    model = AbstractModel.create(args.model, args.output_file, prompt_generator) if args.output_file is not None else AbstractModel.create(model_name=args.model, output_file_name="output")

    print(f"Using model: {args.model}")
    print(f"Using strategy: {args.strategy}")
    print(f"Using dataset: {args.dataset}")
    print(f"Using few-shot dataset: {args.fewshot_dataset}")
    print(f"Using output file: {args.output_file}")

    answers = model.get_answers_and_cache(dataset)
    if args.model == "baseline":
        postprocessed = postprocess_all_baseline(answers, dataset)
        results = evaluate_baseline(postprocessed)
    else:
        postprocessed = postprocess_all(answers, dataset)
        results = evaluate_all(postprocessed)

    output_str = f"""

    Evaluation done. Results:
    - Model: {args.model}
    - Dataset: {args.dataset}
    - Strategy: {args.strategy}

    RESULT SUMMARY:
    - Total questions: {len(list(results.keys()))}
    - Correct answers in overall question: {[results[key]["case_1_em"] for key in results.keys()].count(True)}

    """

    print(output_str)

    with open(f"results/{args.output_file}_{args.model}_{args.strategy}_{args.dataset}_{datetime.now().strftime('%y%m%d-%H%M%S')}.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Results written to file.")

if __name__ == '__main__':
    main()