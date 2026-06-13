import os
import re
import json
import tiktoken
import numpy as np

from typing import Any
from functools import lru_cache
from PIL import Image
from tools.markdown import IMAGE_START, IMAGE_END
from rag.prompt import GRAPH_FIELD_SEP
from rag.multimodal.utils import (
    STORAGES_IMPORT_PATH, 
    STORAGE_META_FIELDS,
    NameSpace, 
    make_namespace, 
    lazy_external_import
)


_ENCODER = None


def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = _ENCODER.encode(content)
    return tokens


def chunking_by_character(
    content: str,
    split_by_character: str,
    tiktoken_model: str = "gpt-4o",
) -> list[dict[str, Any]]:

    results: list[dict[str, Any]] = []

    raw_chunks = content.split(split_by_character)
    new_chunks = []

    for chunk in raw_chunks:
        image_start_pos = chunk.find(IMAGE_START)
        image_end_pos = chunk.find(IMAGE_END)

        if image_start_pos != -1 and image_end_pos != -1:
            image_content = chunk[image_start_pos: image_end_pos + len(IMAGE_END)]
            image_content = image_content.replace(IMAGE_START, "").replace(IMAGE_END, "")
            images = json.loads(image_content)
            chunk = chunk[:image_start_pos] + chunk[image_end_pos + len(IMAGE_END):]
        else:
            images = []

        _tokens = encode_string_by_tiktoken(chunk, model_name=tiktoken_model)
        new_chunks.append((len(_tokens), chunk, images))

    for index, (_len, chunk, images) in enumerate(new_chunks):
        results.append(
            {
                "tokens": _len,
                "content": chunk.strip(),
                "images": images,
                "chunk_order_index": index,
            }
        )
    return results

@lru_cache(maxsize=1)
def load_images_map_file(images_map_file: str) -> dict[str, str]:
    with open(images_map_file, "r") as f:
        url2name = json.load(f)
    return url2name


def img_url_to_img_name(image_url: str, images_map_file: str) -> str:
    """
    Get the image name from the image URL.
    """
    url2name = load_images_map_file(images_map_file)
    image_name = url2name.get(image_url, "")
    return image_name


def img_path_to_sg_path(image_path: str, scene_graph_dir: str) -> str:
    """
    Get the scene graph path from the image path.
    """
    image_filename = os.path.basename(image_path).split(".")[0]
    scene_graph_file = f"{image_filename}.json"
    return os.path.join(scene_graph_dir, scene_graph_file)


def format_scene_graph(scene_graph: dict[str, Any]) -> str:
    template = "{objects}\n{relations}"
    objects, relations = [], []

    for obj in scene_graph["objects"]:
        obj_id = obj["id"]
        obj_category = obj["category"]
        obj_bbox = tuple([round(x, 2) for x in obj["bbox"]])
        objects.append(f"- <object_{obj_id}>: {obj_category}, {obj_bbox}")
    
    for rel in scene_graph["relations"]:
        rel_id = rel["id"]
        relation = rel["relation"]
        obj1, obj2 = rel["objects"]
        relations.append(f"- <relation_{rel_id}>: <object_{obj1}> {relation} <object_{obj2}>")
    
    scene_graph = template.format(objects="\n".join(objects), relations="\n".join(relations))
    
    return scene_graph


def load_knowledge_graph(
    graph_dir: str,
    config: dict[str, str]
):
    
    cls_config = {
        "working_dir": graph_dir,
        "node2vec_params": config["node2vec_params"],
    }

    storage_name = config["graph_storage"]
    graph_storage_cls = lazy_external_import(STORAGES_IMPORT_PATH[storage_name], storage_name)

    knowledge_graph = graph_storage_cls(
        namespace=make_namespace(config["namespace_prefix"], NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION),
        embedding_func=config["embedding_func"],
        global_config=cls_config,
    )

    return knowledge_graph


def load_text_chunks_db(
    graph_dir: str,
    config: dict[str, str],
    base_namespace: str = NameSpace.KV_STORE_TEXT_CHUNKS,
):
    
    cls_config = {
        "working_dir": graph_dir,
    }

    storage_name = config["kv_storage"]
    kv_storage_cls = lazy_external_import(STORAGES_IMPORT_PATH[storage_name], storage_name)

    text_chunks_bd = kv_storage_cls(
        namespace=make_namespace(config["namespace_prefix"], base_namespace),
        embedding_func=config["embedding_func"],
        global_config=cls_config,
    )

    return text_chunks_bd


def load_vector_db(
    graph_dir: str,
    config: dict[str, str],
    base_namespace: str,
):
    
    cls_config = {
        "vector_db_storage_cls_kwargs": config["vector_db_storage_cls_kwargs"],
        "unified_vector_storage": config["unified_vector_storage"],
        "vector_storage_dir": config["vector_storage_dir"],
        "embedding_batch_num": config["embedding_batch_num"],
        "remove_existing": config.get("remove_existing", False),
        "working_dir": graph_dir,
    }

    storage_name = config["vector_storage"]
    vector_storage_cls = lazy_external_import(STORAGES_IMPORT_PATH[storage_name], storage_name)

    if base_namespace == NameSpace.VECTOR_STORE_IMAGES:
        embedding_func = config["vision_embedding_func"]
    elif base_namespace == NameSpace.VECTOR_STORE_MULTIMODAL:
        embedding_func = config["mm_embedding_func"]
    else:
        embedding_func = config["embedding_func"]

    vector_db = vector_storage_cls(
        namespace=make_namespace(config["namespace_prefix"], base_namespace),
        embedding_func=embedding_func,
        meta_fields=STORAGE_META_FIELDS[base_namespace],
        global_config=cls_config,
    )

    return vector_db


def parse_images_attr(images_attr: str, image_dir: str, scene_graph_dir: str):

    def _merge_regions(region1: tuple, region2: tuple):
        x1, y1, x2, y2 = region1
        x3, y3, x4, y4 = region2
        return (min(x1, x3), min(y1, y3), max(x2, x4), max(y2, y4))

    images = images_attr.split(GRAPH_FIELD_SEP)
    images_with_roi = []

    for image in images:
        split_results = image.split("|")

        if len(split_results) == 3:
            image_relpath, reg_label, weight = split_results
            reg_sublabel = None
        elif len(split_results) == 4:
            image_relpath, reg_label, reg_sublabel, weight = split_results
        else:
            raise ValueError(f"Invalid image attribute: {image}")
    
        image_path = os.path.join(image_dir, image_relpath)
        scene_graph_path = img_path_to_sg_path(image_path, scene_graph_dir)
        scene_graph = json.load(open(scene_graph_path, "r"))

        objects = {obj["id"]: obj for obj in scene_graph["objects"]}
        relations = {rel["id"]: rel for rel in scene_graph["relations"]}

        if len(split_results) == 3 and reg_label == "image":
            # the overall image
            images_with_roi.append({
                "image_path": image_path, 
                "roi": (0, 0, 1, 1),
                "weight": float(weight)
            })
            continue

        results = re.search(r"<object_(\d+)>", reg_label)
        if results is not None:
            # the region of an object
            obj_id = int(results.group(1))
            if obj_id not in objects:
                continue

            obj = objects[obj_id]
            images_with_roi.append({
                "image_path": image_path, 
                "roi": tuple([round(x, 2) for x in obj["bbox"]]),
                "weight": float(weight)
            })
            continue

        results = re.search(r"<relation_(\d+)>", reg_label)
        if results is not None:
            rel_id = int(results.group(1))
            if rel_id not in relations:
                continue

            rel = relations[rel_id]
            source_obj, target_obj = rel["objects"]

            if reg_sublabel is not None:
                # only the source or target object is specified
                if reg_sublabel == "source":
                    obj = objects[source_obj]
                elif reg_sublabel == "target":
                    obj = objects[target_obj]
                else:
                    raise ValueError(f"Invalid image attribute: {image}")
                images_with_roi.append({
                    "image_path": image_path, 
                    "roi": tuple([round(x, 2) for x in obj["bbox"]]),
                    "weight": float(weight)
                })
            else:
                # for a relation, merge the regions of the source and target objects
                region = _merge_regions(
                    tuple([round(x, 2) for x in objects[source_obj]["bbox"]]),
                    tuple([round(x, 2) for x in objects[target_obj]["bbox"]])
                )
                images_with_roi.append({
                    "image_path": image_path, 
                    "roi": region,
                    "weight": float(weight)
                })
    
    # deduplicate the images
    images_with_roi_dict = {}
    for item in images_with_roi:
        key = (item["image_path"], item["roi"])
        if key in images_with_roi_dict:
            images_with_roi_dict[key]["weight"] += item["weight"]
        else:
            images_with_roi_dict[key] = item

    images_with_roi = list(images_with_roi_dict.values())
    return images_with_roi


def load_roi_image(
    image_path: str,
    regions: list[tuple],
    return_original: bool = False,
) -> list[np.ndarray]:    
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    roi_images = []
    for region in regions:
        x_min, y_min, x_max, y_max = region
        x_min, y_min, x_max, y_max = int(x_min * width), int(y_min * height), \
            int(x_max * width), int(y_max * height)
        
        roi_image = image.crop((x_min, y_min, x_max, y_max))
        roi_images.append(np.array(roi_image))

    if return_original:
        roi_images.insert(0, np.array(image))

    return roi_images
