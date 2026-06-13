import os
import json
import fcntl
import requests
from tqdm import tqdm
from PIL import Image
from io import BytesIO


def preprocess_args(args):
    # set environment variables
    print(f"Using device: {args.device}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # change device to cuda or cpu
    args.device = "cuda" if args.device != "-1" else "cpu"

    return args


def setup_seed(seed=42, enable_torch=False, enable_cudnn=False):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    if enable_torch:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if enable_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def jsonl_generator(jsonl_file):
    with open(jsonl_file, 'r') as f:
        for line in f:
            yield json.loads(line)


def convert_jsonl_to_json(jsonl_file, json_file, indent=None):
    json_data = {}
    for data in jsonl_generator(jsonl_file):
        json_data.update(data)
    json.dump(json_data, open(json_file, "w"), indent=indent)


def update_json(json_file, data, file_lock=False, indent=None):
    # load previous results
    if os.path.exists(json_file):
        json_data = json.load(open(json_file, "r"))
    else:
        json_data = {}

    # update results
    json_data.update(data)
    with open(json_file, "w") as f:
        if not file_lock:
            json.dump(json_data, f, indent=indent)
        else:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                json.dump(json_data, f, indent=indent)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


def get_token_length(text, encoding_name="gpt-4"):
    import tiktoken

    tokenizer = tiktoken.encoding_for_model(encoding_name)
    tokens = tokenizer.encode(text)
    return len(tokens)


def truncate_token_length(text, encoding_name="gpt-4", max_tokens=7000):
    """
    Trim the question to a maximum number of tokens, refer to:
    https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    import tiktoken

    if max_tokens == -1:
        return text
    
    tokenizer = tiktoken.encoding_for_model(encoding_name)
    tokens = tokenizer.encode(text)

    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        trimmed_text = tokenizer.decode(tokens)
        return trimmed_text
    else:
        return text


def load_image(image_file, color_mode='RGB'):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert(color_mode)
    else:
        image = Image.open(image_file).convert(color_mode)
    return image


def parallel_by_thread(tasks, func, max_threads=10, **kwargs):
    from concurrent.futures import ThreadPoolExecutor

    print(f"Processing {len(tasks)} tasks using {max_threads} threads.")
    results = {}
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Submit tasks to the executor
        futures = [executor.submit(func, task, **kwargs) for task in tasks]

        # Wait for all tasks to complete
        for future in tqdm(futures, total=len(tasks), desc="Processing"):
            result = future.result()
            if result:
                results.update(result)
    print("All tasks processed.")
    
    return results

def parallel_by_process(tasks, func, max_processes=10, **kwargs):
    from concurrent.futures import ProcessPoolExecutor

    print(f"Processing {len(tasks)} tasks using {max_processes} processes.")
    results = {}
    with ProcessPoolExecutor(max_workers=max_processes) as executor:
        # Submit tasks to the executor
        futures = [executor.submit(func, task, **kwargs) for task in tasks]

        # Wait for all tasks to complete
        for future in tqdm(futures, total=len(tasks), desc="Processing"):
            result = future.result()
            if result:
                results.update(result)
    print("All tasks processed.")
    
    return results
