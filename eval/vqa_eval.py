import os
import time
import argparse
from functools import partial
from tqdm import tqdm
from tools.config import load_dataset_cfg
from tools.utils import preprocess_args, jsonl_generator, update_json, parallel_by_process


class AverageMeter:
    def __init__(self, name=""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


def save_results(dataset, split, ans_file, metrics):
    result_dir = os.path.dirname(ans_file)
    result_file = os.path.join(result_dir, f"{dataset}_{split}_results.json")
    
    answer_filename = os.path.basename(ans_file).split(".")[0]
    timestamp = time.strftime("%Y-%m-%d-%H:%M:%S")
    key = f"{answer_filename}_{timestamp}"

    result = {
        key: {
            "timestamp": timestamp,
            **metrics
        }
    }

    update_json(result_file, result, indent=4)


def evaluate_envqa(dataset, split, ans_file, device='cpu', num_processes=1):
    from eval.envqa_evaluator import evaluate_example

    # Prepare data
    examples = []
    question_type_map = {}

    data = jsonl_generator(ans_file)
    for item in data:
        reference = item["reference"]
        reference_list = [ref.strip() for ref in reference.split("|")]
        question_type=item["question_type"]
        
        if question_type == "infoseek":
            question_type = "templated"
        
        question_type_map[item["question_id"]] = question_type
        examples.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "reference_list": reference_list,
            "candidate": item["answer"],
            "question_type": question_type
        })

    # Evaluate
    evaluate_func = partial(evaluate_example, device=device)
    if num_processes > 1:
        results = parallel_by_process(examples, evaluate_func, max_processes=num_processes)
    else:
        results = {}
        for example in tqdm(examples):
            result = evaluate_func(example)
            results.update(result)
    
    # Calculate accuracy
    single_hop_acc = AverageMeter("Single-hop accuracy")
    two_hop_acc = AverageMeter("2-hop accuracy")
    all_acc = AverageMeter("All accuracy")

    for question_id in results.keys():
        question_type = question_type_map[question_id]
        score = results[question_id]

        if question_type != "2_hop":
            single_hop_acc.update(score)
        elif question_type == "2_hop":
            two_hop_acc.update(score)
        else:
            raise ValueError(f"Invalid question type: {question_type}")
        all_acc.update(score)

    # Save results
    metrics = {
        "single_hop_accuracy": single_hop_acc.avg,
        "two_hop_accuracy": two_hop_acc.avg,
        "all_accuracy": all_acc.avg
    }
    print(metrics)
    save_results(dataset, split, ans_file, metrics)


def evaluate_infoseek(dataset, split, ans_file):
    from eval.infoseek_evaluator import evaluate_infoseek_full, prepare_qid2example
    
    # load dataset configuration
    dataset_cfg = load_dataset_cfg().get(f"{dataset}-eval")

    # Prepare data
    reference = jsonl_generator(dataset_cfg.get("infoseek_reference_file"))
    reference_qtype = jsonl_generator(dataset_cfg.get("infoseek_qtype_file"))
    qid2example = prepare_qid2example(reference, reference_qtype)

    data = jsonl_generator(ans_file)
    predictions = [{"data_id": item["question_id"], "prediction": item["answer"]} for item in data]

    # split predictions into two splits: unseen_question and unseen_entity
    unseen_question = []
    unseen_entity = []

    for pred in predictions:
        data_id = pred['data_id']
        if data_id in qid2example:
            if qid2example[data_id]['data_split'].endswith('unseen_question'):
                unseen_question.append(pred)
            else:
                unseen_entity.append(pred)
        else:
            raise ValueError(f"Data ID {data_id} not found in qid2example")
    
    # Evaluate unseen_question and unseen_entity
    metrics = evaluate_infoseek_full(
        [unseen_question, unseen_entity], 
        [qid2example, qid2example]
        )
    print(metrics)
    save_results(dataset, split, ans_file, metrics)
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", help="Device to run the model on")
    parser.add_argument("--dataset", type=str, default="envqa", choices=["envqa", "infoseek"])
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Data split (val or test)")
    parser.add_argument("--ans_file", type=str, default="", help="Path to answers file")

    # Additional arguments for envqa evaluation
    parser.add_argument("--num_processes", type=int, default=1, help="Number of processes for parallel evaluation")

    args = parser.parse_args()
    args = preprocess_args(args)

    print(f"Evaluating on {args.ans_file} for {args.dataset} dataset")
    if args.dataset == "infoseek":
        evaluate_infoseek(args.dataset, args.split, args.ans_file)
    elif args.dataset == "envqa":
        evaluate_envqa(args.dataset, args.split, args.ans_file, 
                        device=args.device, num_processes=args.num_processes)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    