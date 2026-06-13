import os
import json
import argparse
import requests
from hashlib import md5
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from tools.utils import jsonl_generator, get_token_length, parallel_by_thread, update_json
from tools.markdown import convert_to_markdown


def sample_markdown(jsonl_list, sample_ids, convert_func, working_dir, **kwargs):
    markdwon_dir = os.path.join(working_dir, "markdown")
    if not os.path.exists(markdwon_dir):
        os.makedirs(markdwon_dir)
    
    avg_token_length = 0
    for idx in sample_ids:
        sample = jsonl_list[idx]
        sample_md = convert_func(sample, **kwargs)

        with open(os.path.join(markdwon_dir, f"sample_{idx}.md"), 'w') as f:
            f.write(sample_md)
        avg_token_length += get_token_length(sample_md)
    
    avg_token_length /= len(sample_ids)
    print(f"Average token length: {avg_token_length}")
    print(f"Markdown files saved in {markdwon_dir}")


def download_func(document, img_save_dir, img_path_prefix="", 
                  enable_resize=False, short_edge=640, enable_filter=True, min_size=2048,
                  ignore_exceptions=False):
    
    headers = {'User-Agent': 'CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org)'}

    def _download(url, img_filename):
        try:
            img_save_path = os.path.join(img_save_dir, img_filename)
            if os.path.exists(img_save_path):
                return img_filename
            
            response = requests.get(url, headers=headers)
            response.raise_for_status() # raise an exception for 4xx/5xx status codes
            
            # filter images by size
            if enable_filter:
                file_size = int(response.headers.get('Content-Length', min_size))
                if file_size < min_size:
                    return ''

            # save image to a file
            img_bytes = BytesIO(response.content)
            img = Image.open(img_bytes).convert("RGB")
            if enable_resize:
                img = resize_image(img, short_edge)
            img.save(img_save_path, format='JPEG')
            return img_filename

        except Exception as e:
            if not ignore_exceptions:
                print(f"Failed to download or save image from {img_url}: {e}")
            return ''

    url = list(document.keys())[0]
    document = document[url]

    url2path = {}
    img_urls = document["image_urls"]
    for img_url in img_urls:
        img_filename = md5(img_url.encode()).hexdigest() + ".jpg"
        ret = _download(img_url, img_filename)
        url2path[img_url] = os.path.join(img_path_prefix, img_filename) if ret != '' else ''

    return url2path


def download_kb_image(tasks, img_save_dir, img_path_map_file, max_threads=10,
                      img_path_prefix="", enable_resize=False, short_edge=640, ignore_exceptions=True):
    kwargs = {
        "img_save_dir": img_save_dir,
        "img_path_prefix": img_path_prefix,
        "enable_resize": enable_resize,
        "short_edge": short_edge,
        "ignore_exceptions": ignore_exceptions
    }
    all_url2path = parallel_by_thread(tasks, download_func, max_threads=max_threads, **kwargs)

    total_images = len(all_url2path)
    success_images= sum([1 for path in all_url2path.values() if path != ''])    
    print(f"Downloaded {success_images} out of {total_images} images")

    update_json(img_path_map_file, all_url2path)


def resize_image(img, short_edge):
    width, height = img.size
    if width >= height > short_edge:
        new_height = short_edge
        new_width = int(width * new_height / height)
    elif height >= width > short_edge:
        new_width = short_edge
        new_height = int(height * new_width / width)
    else:
        new_width = width
        new_height = height
    
    return img.resize((new_width, new_height))


def resize_func(img_path, resized_img_dir, short_edge=640):
    resized_img_path = os.path.join(resized_img_dir, os.path.basename(img_path))
    if os.path.exists(resized_img_path):
        return {img_path: resized_img_path}

    try:
        img = Image.open(img_path).convert("RGB")
        img = resize_image(img, short_edge)
        img.save(resized_img_path, format='JPEG')
        return {img_path: resized_img_path}
    except Exception:
        return {img_path: ''}


def resize_kb_image(ori_img_dir, resized_img_dir, short_edge=640, max_threads=10):
    img_files = os.listdir(ori_img_dir)
    tasks = [os.path.join(ori_img_dir, img_file) for img_file in img_files]
    kwargs = {"resized_img_dir": resized_img_dir, "short_edge": short_edge}
    img_path_map = parallel_by_thread(tasks, resize_func, max_threads=max_threads, **kwargs)

    for img_path in img_path_map:
        if img_path_map[img_path] == '':
            print(f"Failed to resize image {img_path}")


def integrity_func(document, url2path, img_save_dir):
    url = list(document.keys())[0]
    document = document[url]
    img_urls = document["image_urls"]

    download_tasks = {}
    for img_url in img_urls:
        if img_url not in url2path:
            print(f"Image {img_url} not found in the image map. It will be downloaded.")
            download_tasks[img_url] = img_url
        else:
            img_filename = os.path.basename(url2path[img_url])
            img_path = os.path.join(img_save_dir, img_filename)
            if not os.path.exists(img_path):
                print(f"Image {img_url} not found in the image directory. It will be downloaded.")
                download_tasks[img_url] = img_url

    return download_tasks


def check_kb_image_integrity(tasks, img_save_dir, img_path_map_file, img_path_prefix, 
                      max_threads=10, enable_resize=False, short_edge=640, ignore_exceptions=True):

    # check image integrity
    url2path = json.load(open(img_path_map_file, 'r'))
    integrity_kwargs = {
        "url2path": url2path,
        "img_save_dir": img_save_dir,
    }

    results = parallel_by_thread(tasks, integrity_func, max_threads=max_threads, **integrity_kwargs)

    # download missing images
    download_kwargs = {
        "img_save_dir": img_save_dir,
        "img_path_prefix": img_path_prefix,
        "enable_resize": enable_resize,
        "short_edge": short_edge,
        "ignore_exceptions": ignore_exceptions
    }

    download_tasks = []
    for img_url in results:
        download_tasks.append({img_url: {"image_urls": [img_url]}})

    print(f"Downloading {len(download_tasks)} missing images...")

    new_url2path = parallel_by_thread(download_tasks, download_func, max_threads=max_threads, **download_kwargs)
    
    new_images = len(new_url2path)
    success_images = sum([1 for path in new_url2path.values() if path != ''])
    print(f"Downloaded {success_images} out of {new_images} images")

    update_json(img_path_map_file, new_url2path)


def filter_kb_image(img_path_map_file, img_save_dir, min_size=2048):
    url2path = json.load(open(img_path_map_file, 'r'))

    count = 0
    for img_url in tqdm(url2path):
        if url2path[img_url] == '':
            continue

        img_filename = os.path.basename(url2path[img_url])
        img_path = os.path.join(img_save_dir, img_filename)
        if os.path.getsize(img_path) < min_size:
            os.remove(img_path)
            url2path[img_url] = ''
            count += 1
    
    print(f"Removed {count} images with size less than {min_size} bytes")
    json.dump(url2path, open(img_path_map_file, 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("function", type=str, choices=["sample", "download", "resize", "integrity"], help="Operation to run: sample markdown, download KB images, resize local images, or verify/download missing images")
    parser.add_argument("--kb_file", type=str, default="./data/kb.jsonl", help="Path to the KB JSONL file")
    parser.add_argument("--working_dir", type=str, default="./working_dir", help="Directory for generated sample markdown files")
    parser.add_argument("--max_threads", type=int, default=8, help="Maximum number of worker threads")
    parser.add_argument("--img_dir", type=str, default=None, help="Image directory for download/integrity output or resize input")
    parser.add_argument("--resized_img_dir", type=str, default=None, help="Output directory for resized images")
    parser.add_argument("--short_edge", type=int, default=640, help="Short edge size for resized images")
    args = parser.parse_args()

    if args.function == "sample":
        jsonl_list = list(jsonl_generator(args.kb_file))
        sample_ids = [151, 278, 309, 328, 385, 557, 601, 663, 964, 1216]

        convert_func = convert_to_markdown
        kwargs = {
            "load_img": True,
            "min_token_size": 200,
            "max_token_size": 500
        }

        sample_markdown(jsonl_list, sample_ids, convert_func, args.working_dir, **kwargs)

    elif args.function in ["download", "integrity"]:
        kb_dir = os.path.dirname(args.kb_file)
        kb_filename = os.path.basename(args.kb_file).split(".")[0]
        dataset_split = kb_filename.split("_")[-1]

        img_path_map_file = os.path.join(kb_dir, f"kb_images_map.json")

        if args.img_dir is None:
            img_save_dir = os.path.join(kb_dir, "kb_images_ori", dataset_split)
            print(f"Image directory not provided. Saving images to {img_save_dir}")
        else:
            img_save_dir = os.path.join(args.img_dir, dataset_split)
            print(f"Saving images to {img_save_dir}")

        os.makedirs(img_save_dir, exist_ok=True)

        jsonl_list = list(jsonl_generator(args.kb_file))

        if args.function == "integrity":
            check_kb_image_integrity(jsonl_list, img_save_dir, img_path_map_file, img_path_prefix=dataset_split,
                                     max_threads=args.max_threads, enable_resize=True, short_edge=args.short_edge)
        else:
            download_kb_image(jsonl_list, img_save_dir, img_path_map_file, img_path_prefix=dataset_split,
                            max_threads=args.max_threads, enable_resize=True, short_edge=args.short_edge)

    elif args.function == "resize":
        assert args.img_dir is not None, "Image directory not provided"
        assert args.resized_img_dir is not None, "Resized image directory not provided"

        os.makedirs(args.resized_img_dir, exist_ok=True)
        resize_kb_image(args.img_dir, args.resized_img_dir, short_edge=args.short_edge, max_threads=args.max_threads)

    else:
        raise ValueError("Invalid function")
