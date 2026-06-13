import os
import time
import json
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset.utils import vqa_collate_fn
from eval.vqa_eval import evaluate_infoseek, evaluate_envqa
from tools.utils import preprocess_args, setup_seed, truncate_token_length
from tools.initialization import load_vqa_dataset, load_answer_generator, load_retriever


def get_caption_filename(args):
    return os.path.join(args.working_dir, "caption", 
                        f"qwen_{args.dataset}_{args.split}_captions.json")

def get_context_filename(args):
    comment = f"_{args.comment}" if args.comment != "" else ""
    return os.path.join(args.working_dir, "context", 
                        f"{args.retriever}_{args.dataset}_{args.split}_context{comment}.json")

def get_answer_filename(args):
    comment = f"_{args.comment}" if args.comment != "" else ""
    return os.path.join(args.working_dir, "vqa", f'{args.model_family}-results',
                        f"{args.retriever}_{args.dataset}_{args.split}_answers{comment}.jsonl")


def load_caption(args):
    if args.use_caption:
        caption_file = get_caption_filename(args)
        if os.path.exists(caption_file):
            print(f"Loading caption from {caption_file}")
            with open(caption_file, "r") as f:
                caption_dict = json.load(f)
        else:
            raise ValueError(f"Caption file not found: {caption_file}")
    else:
        print("No caption used for retrieval.")
        caption_dict = None
    
    return caption_dict


def generate_caption(args):
    # load dataset
    dataset = load_vqa_dataset(args.dataset, args.split)    
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             collate_fn=vqa_collate_fn, pin_memory=False)
    
    # load answer generator
    answer_generator = load_answer_generator(args.model_family, args.device)
    
    idx = 0
    captions = {}
    for data in tqdm(data_loader, desc="Generating captions"):
        images = data[0]
        image_captions = answer_generator.captioning_forword(images)

        for image_caption in image_captions:
            question_id = dataset.get_question_id(idx)
            captions[str(question_id)] = image_caption
            idx += 1
    
    caption_file = get_caption_filename(args)
    os.makedirs(os.path.dirname(caption_file), exist_ok=True)
    with open(caption_file, "w") as f:
        json.dump(captions, f)


def generate_image_index(args):
    assert args.retriever == "embed", "Only embedding-based retriever supports indexing"
    
    retriever = load_retriever(args)
    retriever.update_faiss_index()


def retrieve_context(args):
    caption_dict = load_caption(args)
    dataset = load_vqa_dataset(args.dataset, args.split)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                         collate_fn=vqa_collate_fn, pin_memory=False)
    
    retriever = load_retriever(args)

    idx = 0
    context_dict = {}
    average_time = 0
    for data in tqdm(data_loader, desc="Retrieving context"):
        start_time = time.time()
        images, questions, _, image_paths = data[:4]

        attributes = []
        for i in range(len(images)):
            question_id = dataset.get_question_id(idx)
            caption = "" if caption_dict is None else caption_dict[str(question_id)]

            attributes.append({
                "question_id": question_id,
                "caption": caption,
                "image_path": image_paths[i],
                "wiki_url": dataset.get_wiki_url(idx),
                "evidence_section_id": dataset.get_evidence_id(idx)
            })
            idx += 1
        contexts = retriever.retrieve(questions, images, attributes)
        end_time = time.time()
        average_time += (end_time - start_time)

        for context, attribute in zip(contexts, attributes):
            question_id = attribute["question_id"]
            context_dict[str(question_id)] = context
    
    # check for failed retrieval
    fail_count = sum(1 for context in context_dict.values() if context == "")
    print(f"Successfully retrieved context for {len(context_dict) - fail_count} questions out of {len(dataset)}")
    
    average_time /= len(dataset)
    print(f"Average retrieval time per example: {average_time:.4f} seconds")
    
    # context file
    context_file = get_context_filename(args)
    os.makedirs(os.path.dirname(context_file), exist_ok=True)
    with open(context_file, "w") as f:
        json.dump(context_dict, f)


def generate_vqa(args, max_context_len=40960):
    # load retrieved context
    context_file = get_context_filename(args)
    if not os.path.exists(context_file):
        print(f"context file not found: {context_file}. Answering without context.")
        vqa_contexts = None
        args.retriever = "none"
    else:
        print(f"Loading context from {context_file}")
        with open(context_file, "r") as f:
            vqa_contexts = json.load(f)
    
    # load dataset
    dataset = load_vqa_dataset(args.dataset, args.split)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             collate_fn=vqa_collate_fn, pin_memory=False)
    
    # load answer generator
    answer_generator = load_answer_generator(args.model_family, args.device)
    
    # answers file
    answers_file = get_answer_filename(args)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    idx = 0
    average_time = 0
    for data in tqdm(data_loader, desc="Generating answers"):
        start_time = time.time()
        images, questions, gt_answers, image_paths = data[:4]

        contexts, question_ids, question_types = [], [], []
        for i in range(len(images)):
            question_id = dataset.get_question_id(idx)
            question_type = dataset.get_question_type(idx)

            if vqa_contexts is not None and vqa_contexts[str(question_id)] != "":
                context = vqa_contexts[str(question_id)]
                # context_split = context.split('-----Sources-----')
                # context = context_split[0] if len(context_split) == 1 else context_split[1]
                
                if args.model_family in ["llava-7b", "llava-13b", "deepseek-3b", "deepseek-16b"]:
                    # only keep the text chunks due to the limited context length
                    context = truncate_token_length(context, max_tokens=2560)
                else:
                    context = truncate_token_length(context, max_tokens=max_context_len)
            else:
                context = None
            
            contexts.append(context)
            question_ids.append(question_id)
            question_types.append(question_type)

            idx += 1
        
        answers = answer_generator.vqa_forword(questions, images, contexts=contexts)
        end_time = time.time()
        average_time += (end_time - start_time)

        for i in range(len(images)):
            ans_file.write(json.dumps({"question_id": question_ids[i],
                                       "question_type": question_types[i],
                                       "question": questions[i],
                                       "answer": answers[i],
                                       "reference": gt_answers[i],
                                       "image_path": image_paths[i]}) + "\n")
            ans_file.flush()

    ans_file.close()
    print(f"Answers saved to {answers_file}")

    average_time /= len(dataset)
    print(f"Average VQA time per example: {average_time:.4f} seconds")


def evaluate_vqa(args):
    answers_file = get_answer_filename(args)

    print(f"Evaluating on {answers_file} for {args.dataset} dataset")
    if args.dataset == "infoseek":
        evaluate_infoseek(args.dataset, args.split, answers_file)
    elif args.dataset == "envqa":
        evaluate_envqa(args.dataset, args.split, answers_file, device=args.device)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


if __name__ == "__main__":
    # set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=str, choices=['caption', 'index', 'retrieve', 'vqa', 'evaluate'])
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--working_dir", type=str, default="./working_dir")
    parser.add_argument("--model_family", type=str, default="qwen")
    parser.add_argument("--dataset", type=str, default="envqa", choices=["envqa", "infoseek"])
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for the model')
    parser.add_argument("--retriever", type=str, default="none", choices=["none", "oracle", "embed", "mmrag"])
    parser.add_argument("--num_workers", type=int, default=6, help="number of workers for data loading")
    parser.add_argument("--use_caption", type=bool, default=False, help="whether to use caption for retrieval")
    parser.add_argument("--comment", type=str, default=None, help="comment for the run")
    parser.add_argument("--options", nargs='+', default=[], help="additional options for the retriever")

    args = parser.parse_args()

    args = preprocess_args(args)
    setup_seed(42, enable_torch=True)

    print(f"Running {args.step} on {args.dataset} {args.split} ...")    
    if args.step == 'caption':
        generate_caption(args)
    elif args.step == 'index':
        generate_image_index(args)
    elif args.step == 'retrieve':
        retrieve_context(args)
    elif args.step == 'vqa':
        generate_vqa(args, max_context_len=8192)
    elif args.step == 'evaluate':
        evaluate_vqa(args)
    else:
        raise ValueError(f"Unsupported step: {args.step}")
