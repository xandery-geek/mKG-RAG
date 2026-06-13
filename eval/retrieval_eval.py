import os
import time
import argparse
import json
from tools.utils import update_json
from tools.initialization import load_vqa_dataset


def save_results(args, metrics):
    result_file = os.path.join(args.result_dir, f"{args.dataset}_{args.split}_recall.json")
    res_filename = os.path.basename(args.res_file).split(".")[0]
    timestamp = time.strftime("%Y-%m-%d-%H:%M:%S")
    key = f"{res_filename}_{timestamp}"

    result = {
        key: {
            "timestamp": timestamp,
            **metrics
        }
    }

    update_json(result_file, result, indent=4)


def calculate_recall(retrieved_docs, relevant_docs, top_ks, match_type="any"):
    recalls = {topk: 0 for topk in top_ks}
    for topk in top_ks:
        if match_type == "exact":
            recalls[topk] = len(set(retrieved_docs[:topk]) & set(relevant_docs)) / min(len(relevant_docs), topk)
        elif match_type == "any":
            # If any of the retrieved docs match any of the relevant docs, we consider it a hit
            recalls[topk] = 1 if set(retrieved_docs[:topk]) & set(relevant_docs) else 0
        else:
            raise ValueError(f"Unknown match_type: {match_type}")
    return recalls


def evaluate_doc_level_retrieval(dataset, retrieval_results):
    top_ks = [1, 5, 10, 20, 50]
    recalls = {top_k: 0 for top_k in top_ks}

    for idx in range(len(dataset)):
        question_id = dataset.get_question_id(idx) if hasattr(dataset, "get_question_id") else idx
        retrieved_docs = retrieval_results.get(str(question_id), "")
        retrieved_docs = retrieved_docs.split("|")

        relevant_docs = dataset.get_wiki_url(idx)

        recalls_ = calculate_recall(retrieved_docs, relevant_docs, top_ks=top_ks)
        for top_k in top_ks:
            recalls[top_k] += recalls_[top_k]
    
    metrics = {}
    for top_k in top_ks:
        recalls[top_k] /= len(dataset)
        metrics[f"recall@{top_k}"] = recalls[top_k]
    
    save_results(args, metrics)
    print(f"Recall metrics: {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate retrieval results')
    parser.add_argument('--dataset', type=str, required=True, help='dataset name')
    parser.add_argument('--split', type=str, default='val', help='dataset split')
    parser.add_argument('--res_file', type=str, required=True, help='retrieval results file')
    parser.add_argument('--result_dir', type=str, default='working_dir/retrieval', help='directory for evaluation results')

    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)

    dataset = load_vqa_dataset(args.dataset, args.split)
    with open(args.res_file, 'r') as f:
        retrieval_results = json.load(f)

    evaluate_doc_level_retrieval(dataset, retrieval_results)
