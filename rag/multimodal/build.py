from __future__ import annotations

import os
import re
import time
import json
import asyncio
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from PIL import Image
from collections import Counter, defaultdict

from rag.utils import (
    logger,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    is_float_regex,
    pack_user_ass_to_openai_messages,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
    statistic_data,
)
from rag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
)

from rag.multimodal.operate import (
    format_scene_graph, 
    img_url_to_img_name, 
    img_path_to_sg_path,
    load_knowledge_graph, 
    load_vector_db,
    load_text_chunks_db,
    load_roi_image,
    parse_images_attr, 
    encode_string_by_tiktoken, 
)

from rag.multimodal.utils import split_string_by_multi_markers, clean_str
from rag.prompt import GRAPH_FIELD_SEP, PROMPTS, MM_PROMPTS


def _format_context_base(global_config):
    # add language and example number params to prompt
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    entity_types = global_config["addon_params"].get(
        "entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"]
    )

    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )

    return context_base

def _format_entity_prompt(context_base, global_config=None):
    example_number = global_config["addon_params"].get("example_number", None)
    example_number = example_number if example_number else len(PROMPTS["entity_extraction_examples"])
    examples = "\n".join(PROMPTS["entity_extraction_examples"][:example_number])
    examples = examples.format(**context_base)

    entity_extract_prompt = PROMPTS["entity_extraction"]
    entity_extract_prompt = entity_extract_prompt.format(**context_base, examples=examples, input_text="{input_text}")
    return entity_extract_prompt

def _format_entity_prompt_mm(context_base, global_config=None):
    examples = "\n".join(MM_PROMPTS["mm_examples"])
    examples = examples.format(**context_base)

    entity_extract_prompt = MM_PROMPTS["mm_entity_extraction"]
    entity_extract_prompt = entity_extract_prompt.format(**context_base, examples=examples,
                                                         input_text="{input_text}", image_desc="{image_desc}",
                                                         scene_graph="{scene_graph}")
    
    return entity_extract_prompt

def _format_mapping_prompt(context_base, global_config=None):
    example_number = global_config["addon_params"].get("example_number", None)
    example_number = example_number if example_number else len(MM_PROMPTS["mapping_examples"])
    examples = "\n".join(MM_PROMPTS["mapping_examples"][:example_number])
    examples = examples.format(**context_base)

    mapping_extract_prompt = MM_PROMPTS["mapping_extract_prompt"]
    mapping_extract_prompt = mapping_extract_prompt.format(**context_base, examples=examples, 
                                                           entities="{entities}", relationships="{relationships}", 
                                                           image_desc="{image_desc}", scene_graph="{scene_graph}")

    return mapping_extract_prompt

def _format_entity_relationship(entities, relationships, context_base):
    formatted_entities, formatted_relationships = [], []
    
    entity_format = MM_PROMPTS["entity_format"]
    relationship_format = MM_PROMPTS["relationship_format"]
    tuple_delimiter = context_base["tuple_delimiter"]    
    record_delimiter = context_base["record_delimiter"] + "\n"

    for entity in entities:
        for entity_data in entities[entity]:
            formatted_entities.append(entity_format.format(
                entity_name=entity_data["entity_name"],
                entity_type=entity_data["entity_type"],
                entity_description=entity_data["description"],
                tuple_delimiter=tuple_delimiter,
            ))

    for relationship in relationships:
        for relationship_data in relationships[relationship]:
            formatted_relationships.append(relationship_format.format(
                source_entity=relationship_data["src_id"],
                target_entity=relationship_data["tgt_id"],
                relationship_description=relationship_data["description"],
                relationship_strength=relationship_data["weight"],
                tuple_delimiter=tuple_delimiter,
            ))
    
    formatted_entities = record_delimiter.join(formatted_entities)
    formatted_relationships = record_delimiter.join(formatted_relationships)

    return formatted_entities, formatted_relationships

def _load_image_data(image_obj, images_map_file, image_dir, scene_graph_dir):
    image_url, image_desc = image_obj["url"], image_obj["desc"]
    image_name = img_url_to_img_name(image_url, images_map_file)

    if image_name == "":
        raise ValueError(f"Image name not found for {image_url}")

    image_path = f"{image_dir}/{image_name}"
    image_desc = f"Image Description: {image_desc}" if image_desc else ""

    scene_graph_path = img_path_to_sg_path(image_path, scene_graph_dir)
    scene_graph = json.load(open(scene_graph_path, "r"))
    return image_path, image_desc, scene_graph


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    """Handle entity relation summary
    For each entity or relation, input is the combined description of already existing description and new description.
    If too long, use LLM to summarize.
    """
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description

    if use_llm_func is None:
        return decode_tokens_by_tiktoken(
            tokens[:summary_max_tokens], model_name=tiktoken_model_name
        )
    else:
        prompt_template = PROMPTS["summarize_entity_descriptions"]
        use_description = decode_tokens_by_tiktoken(
            tokens[:llm_max_tokens], model_name=tiktoken_model_name
        )
        context_base = dict(
            entity_name=entity_or_relation_name,
            description_list=use_description.split(GRAPH_FIELD_SEP),
            language=language,
        )
        use_prompt = prompt_template.format(**context_base)
        logger.debug(f"Trigger summary: {entity_or_relation_name}")
        summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
        return summary

async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    
    # Skip empty entities
    if entity_name == '' or entity_description == '':
        return None
    
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
        images=[],
    )

async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])
    
    # skip empty or self-loop edges
    if source == '' or target == '' or source == target or edge_description == '':
        return None

    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        source_id=edge_source_id,
        images=[]
    )

async def _handle_single_mapping_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"mapping"':
        return None
    
    if len(record_attributes) == 4:
        # visual object to entity mapping
        # "mapping", object_name, entity_name, weight
        # "mapping", "image", entity_name, weight
        object_name = clean_str(record_attributes[1].lower())
        entity_name = clean_str(record_attributes[2].upper())
        weight = (
            float(record_attributes[3]) if is_float_regex(record_attributes[3]) else 1.0
        )
        return dict(
            mapping_type="object",
            src_name=object_name,
            tgt_name=entity_name,
            weight=weight,
            source_id=chunk_key
        )
    else:
        # visual relation to relationship mapping
        # "mapping", relation_name, source_entity_name, target_entity_name, weight
        relation_name = clean_str(record_attributes[1].lower())
        source = clean_str(record_attributes[2].upper())
        target = clean_str(record_attributes[3].upper())
        weight = (
            float(record_attributes[4]) if is_float_regex(record_attributes[4]) else 1.0
        )
        return dict(
            mapping_type="relation",
            src_name=relation_name,
            tgt_name=(source, target),
            weight=weight,
            source_id=chunk_key
        )

async def _handle_images_attribute(
    maybe_mappings: list[dict],
    maybe_nodes: dict[str, list[dict]],
    maybe_edges: dict[tuple[str, str], list[dict]],
    image_dir: str | None=None,
    image_path: str | None=None,
    scene_graph: dict | None=None,
    global_config: dict[str, str]=None,
):
    """
    Update images attribute of nodes and edges based on the mappings information.
    The image attributes of maybe_nodes and maybe_edges are updated in-place.
    """
    if image_path is None or scene_graph is None:
        return
    
    def _merge_images(already_images: list[str], new_images: str):
        if new_images not in already_images:
            already_images.append(new_images)
        return already_images
    
    def _add_image_attribute(node_or_edge: list[dict], image_attribute):
        for i in range(len(node_or_edge)):
            node_or_edge[i]["images"] = _merge_images(node_or_edge[i]["images"], image_attribute)

    def _find_max_degree_node(nodes: dict[str, list[dict]], edges: dict[tuple[str, str], list[dict]]):
        nodes_degree = defaultdict(int)
        for edge_key in edges.keys():
            nodes_degree[edge_key[0]] += 1
            nodes_degree[edge_key[1]] += 1
        
        sorted_nodes = sorted(nodes_degree.items(), key=lambda x: x[1], reverse=True)
        for node_id, _ in sorted_nodes:
            if node_id in nodes.keys():
                return node_id
        
        return ''

    overall_image_used = False
    image_relpath = os.path.relpath(image_path, image_dir)
    
    # Add the region information to the images attribute of nodes and edges
    objects_id = [f"<object_{obj['id']}>" for obj in scene_graph["objects"]]
    relations_id = [f"<relation_{rel['id']}>" for rel in scene_graph["relations"]]

    # add mapping information to the images attribute of nodes and edges
    for mapping in maybe_mappings:
        mapping_type = mapping["mapping_type"]
        if mapping_type == "object":
            src_name, tgt_name = mapping["src_name"], mapping["tgt_name"]
            weight = mapping["weight"]

            # update images attribute of nodes when both visual object and textual entity are valid
            if "image" in src_name and tgt_name in maybe_nodes.keys():
                overall_image_used = True
                _add_image_attribute(maybe_nodes[tgt_name], f"{image_relpath}|image|{weight}")
            elif src_name in objects_id and tgt_name in maybe_nodes.keys():
                _add_image_attribute(maybe_nodes[tgt_name], f"{image_relpath}|{src_name}|{weight}")
            
        elif mapping_type == "relation":
            src_name, tgt_name = mapping["src_name"], mapping["tgt_name"]
            weight = mapping["weight"]

            if ("image" not in src_name) and (src_name not in relations_id):
                continue

            rel_source, rel_target = tgt_name
            if (rel_source, rel_target) in maybe_edges.keys():
                edge_key = (rel_source, rel_target)
            elif (rel_target, rel_source) in maybe_edges.keys():
                edge_key = (rel_target, rel_source)
            else:
                edge_key = None
            
            # update images attribute of edges when the entire relationship is valid
            if edge_key:
                if "image" in src_name:
                    overall_image_used = True
                    image_attribute = f"{image_relpath}|image|{weight}"
                else:
                    image_attribute = f"{image_relpath}|{src_name}|{weight}"

                _add_image_attribute(maybe_edges[edge_key], image_attribute)
                continue
            
            # update images attribute of nodes when part of the relationship is valid
            weight = max(weight // 2, 1)
            if rel_source in maybe_nodes.keys():
                if "image" in src_name:
                    overall_image_used = True
                    image_attribute = f"{image_relpath}|image|{weight}"  
                else:
                    image_attribute = f"{image_relpath}|{src_name}|source|{weight}"

                _add_image_attribute(maybe_nodes[rel_source], image_attribute)

            if rel_target in maybe_nodes.keys():
                if "image" in src_name:
                    overall_image_used = True
                    image_attribute = f"{image_relpath}|image|{weight}"  
                else:
                    image_attribute = f"{image_relpath}|{src_name}|target|{weight}"

                _add_image_attribute(maybe_nodes[rel_target], image_attribute)
    
    if overall_image_used:
        return
    
    # If the overall image is not used, attach the whole image to the candidate_node
    # Pattern 1: attach the whole image to the node with the maximum degree
    candidate_node = _find_max_degree_node(maybe_nodes, maybe_edges)
    
    # Pattern 2: attach the whole image to the node with maximum image-text similarity
    if candidate_node == '' and len(maybe_nodes) > 0:
        text_embedding_func = global_config["embedding_func"]
        vision_embedding_func = global_config["vision_embedding_func"]

        image = Image.open(image_path).convert("RGB")
        image_embedding = await vision_embedding_func(image)
        image_embedding = image_embedding.reshape(1, -1)

        nodes_with_desc = []
        for node_id, node_data in maybe_nodes.items():
            nodes_with_desc.extend([(node_id, dp["description"]) for dp in node_data])
        
        descriptions = [node[1] for node in nodes_with_desc]
        text_emvbeddings = await text_embedding_func(descriptions)
        text_emvbeddings = text_emvbeddings.reshape(len(descriptions), -1)

        similarities = np.dot(text_emvbeddings, image_embedding.T)
        candidate_node = nodes_with_desc[np.argmax(similarities)][0]
    
    if candidate_node != '':
        _add_image_attribute(maybe_nodes[candidate_node], f"{image_relpath}|image|1.0")


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """Get existing nodes from knowledge graph use name,if exists, merge data, else create, then upsert."""
    already_entity_types = []
    already_source_ids = []
    already_description = []
    already_images = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])
        already_images.extend(
            split_string_by_multi_markers(already_node["images"], [GRAPH_FIELD_SEP])
        )

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    images = GRAPH_FIELD_SEP.join(
        set([image for dp in nodes_data for image in dp["images"]] + already_images)
    )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
        images=images,
        created_at=time.time(),
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    node_data.pop("images") # It is no need to store images in the vector db

    return node_data

async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_images = []

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        # Handle the case where get_edge returns None or missing fields
        if already_edge is not None:
            # Get weight with default 0.0 if missing
            already_weights.append(already_edge.get("weight", 0.0))
            already_source_ids.extend(
                split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
            )
            already_description.append(already_edge["description"])
            already_images.extend(
                split_string_by_multi_markers(already_edge["images"], [GRAPH_FIELD_SEP])
            )

    # Process edges_data with None checks
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data]+ already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data]+ already_source_ids)
    )
    images = GRAPH_FIELD_SEP.join(
        set([image for dp in edges_data for image in dp["images"]] + already_images)
    )

    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                    "images": "",
                    "created_at": time.time(),
                },
            )

    description = await _handle_entity_relation_summary(
        f"({src_id}, {tgt_id})", description, global_config
    )

    # insert the edge into the graph
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            source_id=source_id,
            images=images,
            created_at=time.time(),
        ),
    )

    # edge data for vector db
    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
    )

    return edge_data

async def _update_graph_and_vdb(
    maybe_nodes: dict[list],
    maybe_edges: dict[list],
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict[str, str],
):
    
    graph_id = global_config["graph_id"]

    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )

    all_relationships_data = await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )

    if not (all_entities_data or all_relationships_data):
        logger.info("Didn't extract any entities and relationships.")
        return

    if not all_entities_data:
        logger.info("Didn't extract any entities")
    if not all_relationships_data:
        logger.info("Didn't extract any relationships")

    logger.info(
        f"New entities or relationships extracted, entities:{all_entities_data}, relationships:{all_relationships_data}"
    )

    if entities_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(graph_id + dp["entity_name"], prefix="ent-"): {
                "graph_id": graph_id,
                "entity_name": dp["entity_name"],
                "content": dp["entity_name"] + dp["description"]
            }
            for dp in all_entities_data
        }
        await entities_vdb.upsert(data_for_vdb)

    if relationships_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(graph_id + dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                "graph_id": graph_id,
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "content": dp["src_id"] + dp["tgt_id"] + dp["description"]
            }
            for dp in all_relationships_data
        }
        await relationships_vdb.upsert(data_for_vdb)


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict[str, str],
    llm_response_cache: BaseKVStorage | None = None,
) -> None:
    image_dir = global_config["image_dir"]
    scene_graph_dir = global_config["scene_graph_dir"]
    images_map_file = global_config["images_map_file"]

    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    enable_llm_cache_for_entity_extract: bool = global_config[
        "enable_llm_cache_for_entity_extract"
    ]

    ordered_chunks = list(chunks.items())

    context_base = _format_context_base(global_config)
    entity_extract_prompt = _format_entity_prompt(context_base, global_config)
    entity_extract_prompt_mm = _format_entity_prompt_mm(context_base, global_config)
    mapping_extract_prompt = _format_mapping_prompt(context_base, global_config)

    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    async def _user_llm_func_with_cache(
        input_text: str, image: str|None, history_messages: list[dict[str, str]] = None, max_tokens: int = 512
    ) -> str:
        if enable_llm_cache_for_entity_extract and llm_response_cache:
            if history_messages:
                history = json.dumps(history_messages, ensure_ascii=False)
                _prompt = history + "\n" + input_text
            else:
                _prompt = input_text

            arg_hash = compute_args_hash(_prompt)
            cached_return, _1, _2, _3 = await handle_cache(
                llm_response_cache,
                arg_hash,
                _prompt,
                "default",
                cache_type="extract",
                force_llm_cache=True,
            )
            if cached_return:
                logger.debug(f"Found cache for {arg_hash}")
                statistic_data["llm_cache"] += 1
                return cached_return
            statistic_data["llm_call"] += 1
            if history_messages:
                res: str = await use_llm_func(input_text, image=image, history_messages=history_messages, max_tokens=max_tokens)
            else:
                res: str = await use_llm_func(input_text, image=image, max_tokens=max_tokens)
            await save_to_cache(
                llm_response_cache,
                CacheData(
                    args_hash=arg_hash,
                    content=res,
                    prompt=_prompt,
                    cache_type="extract",
                ),
            )
            return res

        if history_messages:
            return await use_llm_func(input_text, image=image, history_messages=history_messages, max_tokens=max_tokens)
        else:
            return await use_llm_func(input_text, image=image, max_tokens=max_tokens)

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        """ "Prpocess a single chunk
        Args:
            chunk_key_dp (tuple[str, TextChunkSchema]):
                ("chunck-xxxxxx", {"tokens": int, "content": str, "images": list, "full_doc_id": str, "chunk_order_index": int})
        """
        already_processed, already_entities, already_relations = 0, 0, 0
        chunk_key, chunk_dp = chunk_key_dp[0], chunk_key_dp[1]

        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(input_text=content)

        final_result = await _user_llm_func_with_cache(hint_prompt, image=None, max_tokens=1024)
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)

        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await _user_llm_func_with_cache(
                continue_prompt, image=None, history_messages=history, max_tokens=1024
            )

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await _user_llm_func_with_cache(
                if_loop_prompt, image=None, history_messages=history, max_tokens=128
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]], ignorecase=True
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)

        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]], ignorecase=True)

            if_entities = await _handle_single_entity_extraction(record_attributes, chunk_key)
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(record_attributes, chunk_key)
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(if_relation)
                continue

        # Mapping iamges to entities and relationships
        maybe_mappings = list()
        images = chunk_dp.get("images", [])        
        for image_obj in images:
            try:
                image_path, image_desc, scene_graph = _load_image_data(image_obj, images_map_file, image_dir, scene_graph_dir)
            except ValueError as e:
                logger.debug(f"Error: {e}")
                continue
            
            formatted_scene_graph = format_scene_graph(scene_graph)
            formatted_entities, formatted_relations = _format_entity_relationship(
                maybe_nodes, maybe_edges, context_base
            )
            input_prompt = mapping_extract_prompt.format(
                entities=formatted_entities, relationships=formatted_relations, 
                image_desc=image_desc, scene_graph=formatted_scene_graph
            )
            mapping_result = await _user_llm_func_with_cache(input_prompt, image=image_path, max_tokens=512)
        
            # Process the mappings for multiple images
            if mapping_result != "":
                records = split_string_by_multi_markers(
                    mapping_result,
                    [context_base["record_delimiter"], context_base["completion_delimiter"]], ignorecase=True
                )

                for record in records:
                    record = re.search(r"\((.*)\)", record)
                    if record is None:
                        continue
                    record = record.group(1)
                    record_attributes = split_string_by_multi_markers(
                        record, [context_base["tuple_delimiter"]], ignorecase=True)

                    if_mapping = await _handle_single_mapping_extraction(record_attributes, chunk_key)
                    if if_mapping is not None:
                        maybe_mappings.append(if_mapping)

            await _handle_images_attribute(maybe_mappings, maybe_nodes, maybe_edges, 
                                           image_dir, image_path, scene_graph, global_config)
            maybe_mappings.clear()

        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)

        logger.debug(
            f"Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
        )
        return dict(maybe_nodes), dict(maybe_edges)

    tasks = [_process_single_content(c) for c in ordered_chunks]
    results = await asyncio.gather(*tasks)

    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)

    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)

    await _update_graph_and_vdb(
        maybe_nodes, maybe_edges, knowledge_graph_inst, entities_vdb, relationships_vdb, global_config
    )


async def _collect_data_from_graph(
    graph_id: str,
    global_config: dict[str, str],
    graph_config: dict[str, str],
):
    image_dir = global_config["image_dir"]
    scene_graph_dir = global_config["scene_graph_dir"]
    graph_dir = f"{global_config['vector_storage_dir']}/{graph_id}"

    images_data, mm_data = [], []
    knowledge_graph = load_knowledge_graph(graph_dir, graph_config)

    # collect data from nodes
    node_ids = knowledge_graph._graph.nodes()
    for node_id in node_ids:
        node = await knowledge_graph.get_node(node_id)
        images = node.get("images", "")
        description = node.get("description", "")

        if images == "":
            # no images, just add data to multimodal data
            _data = {
                "graph_id": graph_id,
                "entity_name": node_id,
                "description": description,
                "image_path": None,
            }
            mm_data.append(_data)
        else:
            # add data to both images and multimodal data
            images_with_roi = parse_images_attr(images, image_dir, scene_graph_dir)
            _data = [{
                **dp,
                "graph_id": graph_id,
                "entity_name": node_id,
                "description": description,
            } for dp in images_with_roi]

            images_data.extend(_data)
            mm_data.extend(deepcopy(_data))
    
    # collect data from edges
    edge_ids = knowledge_graph._graph.edges()
    for edge_id in edge_ids:
        edge = await knowledge_graph.get_edge(edge_id[0], edge_id[1])
        images = edge.get("images", "")
        description = edge.get("description", "")
        if images == "":
            # no images, just add data to multimodal data
            _data = {
                "graph_id": graph_id,
                "src_id": edge_id[0],
                "tgt_id": edge_id[1],
                "description": description,
                "image_path": None,
            }
            mm_data.append(_data)
        else:
            # add data to both images and multimodal data
            images_with_roi = parse_images_attr(images, image_dir, scene_graph_dir)
            _data = [{
                **dp,
                "graph_id": graph_id,
                "src_id": edge_id[0],
                "tgt_id": edge_id[1],
                "description": description,
            } for dp in images_with_roi]
            images_data.extend(_data)
            mm_data.extend(deepcopy(_data))

    data_for_image_vdb = {}
    for dp in images_data:
        description = dp.pop("description", "")
        hash_id = compute_mdhash_id(''.join([str(val) for val in dp.values()]), prefix="img-")
        
        image = load_roi_image(dp["image_path"], [dp["roi"]], return_original=False)[0]
        data_for_image_vdb[hash_id] = {
            **dp,
            "content": image
        }
    
    data_for_mm_vdb = {}
    for dp in mm_data:
        description = dp.pop("description", "")
        hash_id = compute_mdhash_id(''.join([str(val) for val in dp.values()]), prefix="mm-")

        if dp["image_path"] is not None:
            image = load_roi_image(dp["image_path"], [dp["roi"]], return_original=False)[0]
        else:
            # If no image path, create a empty image
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        
        data_for_mm_vdb[hash_id] = {
            **dp,
            "content": (image, description)
        }
    
    return data_for_image_vdb, data_for_mm_vdb


async def build_separate_images_vdb(
    graph_ids: list[str],
    global_config: dict[str, str],
) -> None:
    from rag.multimodal.utils import NameSpace

    vdb_config = {
        "vector_storage":  global_config["vector_storage"],
        "vector_db_storage_cls_kwargs": global_config["vector_db_storage_cls_kwargs"],
        "unified_vector_storage": global_config["unified_vector_storage"],
        "vector_storage_dir": global_config["vector_storage_dir"],
        "embedding_batch_num": global_config["embedding_batch_num"],
        "namespace_prefix": global_config["namespace_prefix"],
        "embedding_func": global_config["embedding_func"],
        "vision_embedding_func": global_config["vision_embedding_func"],
        "mm_embedding_func": global_config["mm_embedding_func"],
        "remove_existing": True,
    }

    graph_config = {
        "node2vec_params": global_config["node2vec_params"],
        "graph_storage":  global_config["graph_storage"],
        "namespace_prefix": global_config["namespace_prefix"],
        "embedding_func": global_config["embedding_func"],
    }

    use_mm_embedding = global_config["use_mm_embedding"]

    for graph_id in tqdm(graph_ids, desc="Building separate images vector db"):
        graph_dir = f"{global_config['vector_storage_dir']}/{graph_id}"
        data_for_image_vdb, data_for_mm_vdb = await _collect_data_from_graph(graph_id, global_config, graph_config)

        if data_for_image_vdb:
            images_vdb = load_vector_db(graph_dir, vdb_config, NameSpace.VECTOR_STORE_IMAGES)
            await images_vdb.upsert(data_for_image_vdb)
            await images_vdb.index_done_callback()
        
        if use_mm_embedding and data_for_mm_vdb:
            multimodal_vdb = load_vector_db(graph_dir, vdb_config, NameSpace.VECTOR_STORE_MULTIMODAL)
            await multimodal_vdb.upsert(data_for_mm_vdb)
            await multimodal_vdb.index_done_callback()


async def build_unified_images_vdb(
    graph_ids: list[str],
    images_vdb: BaseVectorStorage,
    multimodal_vdb: BaseVectorStorage,
    global_config: dict[str, str],
) -> None:
    
    graph_config = {
        "node2vec_params": global_config["node2vec_params"],
        "graph_storage":  global_config["graph_storage"],
        "namespace_prefix": global_config["namespace_prefix"],
        "embedding_func": global_config["embedding_func"],
    }

    use_mm_embedding = global_config["use_mm_embedding"]

    for graph_id in tqdm(graph_ids, desc="Building unified images vector db"):
        data_for_image_vdb, data_for_mm_vdb = await _collect_data_from_graph(graph_id, global_config, graph_config)
        await images_vdb.upsert(data_for_image_vdb)

        if use_mm_embedding:
            await multimodal_vdb.upsert(data_for_mm_vdb)


async def build_unified_vector_db(
    graph_ids: list[str],
    chunks_vdb: BaseVectorStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict[str, str],
) -> None:
    from rag.multimodal.utils import NameSpace

    vdb_config = {
        "vector_storage":  global_config["vector_storage"],
        "vector_db_storage_cls_kwargs": global_config["vector_db_storage_cls_kwargs"],
        "unified_vector_storage": global_config["unified_vector_storage"],
        "vector_storage_dir": global_config["vector_storage_dir"],
        "embedding_batch_num": global_config["embedding_batch_num"],
        "namespace_prefix": global_config["namespace_prefix"],
        "embedding_func": global_config["embedding_func"]
    }

    for graph_id in tqdm(graph_ids, desc="Building unified vector db"):
        graph_dir = f"{global_config['vector_storage_dir']}/{graph_id}"
        
        sub_chunks_vdb = load_vector_db(graph_dir, vdb_config, NameSpace.VECTOR_STORE_CHUNKS)
        sub_entities_vdb = load_vector_db(graph_dir, vdb_config, NameSpace.VECTOR_STORE_ENTITIES)
        sub_relationships_vdb = load_vector_db(graph_dir, vdb_config, NameSpace.VECTOR_STORE_RELATIONSHIPS)

        tasks = [
            chunks_vdb.merge(sub_chunks_vdb),
            entities_vdb.merge(sub_entities_vdb),
            relationships_vdb.merge(sub_relationships_vdb)
        ]

        await asyncio.gather(*tasks)


async def _merge_knowledge_graph(
    knowledge_graph_inst: BaseGraphStorage,
    sub_graph: BaseGraphStorage,
    global_config: dict[str, str],
):
    config = global_config.copy()
    config["llm_model_func"] = None

    node_ids = sub_graph._graph.nodes()
    for node_id in node_ids:
        node = await sub_graph.get_node(node_id)
        node["images"] = split_string_by_multi_markers(node["images"], [GRAPH_FIELD_SEP])
        await _merge_nodes_then_upsert(node_id, [node], knowledge_graph_inst, config)
    
    edge_ids = sub_graph._graph.edges()
    for edge_id in edge_ids:
        edge = await sub_graph.get_edge(edge_id[0], edge_id[1])
        edge["images"] = split_string_by_multi_markers(edge["images"], [GRAPH_FIELD_SEP])
        await _merge_edges_then_upsert(edge_id[0], edge_id[1], [edge], knowledge_graph_inst, config)
 

async def _merge_text_chunks_db(
    text_chunks: BaseKVStorage,
    sub_text_chunks: BaseKVStorage,
):
    await text_chunks.upsert(sub_text_chunks._data)


async def build_dynamic_graph(
    graph_ids: list[str],
    knowledge_graph_inst: BaseGraphStorage,
    text_chunks: BaseKVStorage,
    chunks_vdb: BaseVectorStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    images_vdb: BaseVectorStorage,
    multimodal_vdb: BaseVectorStorage,
    global_config: dict[str, str],
):
    from rag.multimodal.utils import NameSpace

    vdb_config = {
        "vector_storage":  global_config["vector_storage"],
        "vector_db_storage_cls_kwargs": global_config["vector_db_storage_cls_kwargs"],
        "unified_vector_storage": global_config["unified_vector_storage"],
        "vector_storage_dir": global_config["vector_storage_dir"],
        "embedding_batch_num": global_config["embedding_batch_num"],
        "namespace_prefix": global_config["namespace_prefix"],
        "embedding_func": global_config["embedding_func"],
        "vision_embedding_func": global_config["vision_embedding_func"],
        "mm_embedding_func": global_config["mm_embedding_func"],
    }

    graph_config = {
        "node2vec_params": global_config["node2vec_params"],
        "graph_storage":  global_config["graph_storage"],
        "namespace_prefix": global_config["namespace_prefix"],
        "embedding_func": global_config["embedding_func"],
    }

    text_chunks_config = {
        "kv_storage": global_config["kv_storage"],
        "namespace_prefix": global_config["namespace_prefix"],
        "embedding_func": global_config["embedding_func"],
    }

    for graph_id in graph_ids:
        graph_dir = f"{global_config['vector_storage_dir']}/{graph_id}"
        sub_graph = load_knowledge_graph(graph_dir, graph_config)
        sub_text_chunks = load_text_chunks_db(graph_dir, text_chunks_config, NameSpace.KV_STORE_TEXT_CHUNKS)
        
        sub_chunks_vdb = load_vector_db(graph_dir, vdb_config, NameSpace.VECTOR_STORE_CHUNKS)
        sub_entities_vdb = load_vector_db(graph_dir, vdb_config, NameSpace.VECTOR_STORE_ENTITIES)
        sub_relationships_vdb = load_vector_db(graph_dir, vdb_config, NameSpace.VECTOR_STORE_RELATIONSHIPS)
        sub_images_vdb = load_vector_db(graph_dir, vdb_config, NameSpace.VECTOR_STORE_IMAGES)
        sub_multimodal_vdb = load_vector_db(graph_dir, vdb_config, NameSpace.VECTOR_STORE_MULTIMODAL)

        tasks = [
            _merge_knowledge_graph(knowledge_graph_inst, sub_graph, global_config),
            _merge_text_chunks_db(text_chunks, sub_text_chunks),
            chunks_vdb.merge(sub_chunks_vdb),
            entities_vdb.merge(sub_entities_vdb),
            relationships_vdb.merge(sub_relationships_vdb),
            images_vdb.merge(sub_images_vdb),
            multimodal_vdb.merge(sub_multimodal_vdb)
        ]

        await asyncio.gather(*tasks)
    
    tasks = [
        storage_inst.index_done_callback()
        for storage_inst in [  # type: ignore
            knowledge_graph_inst,
            text_chunks,
            chunks_vdb,
            entities_vdb,
            relationships_vdb,
            images_vdb,
            multimodal_vdb
        ]
        if storage_inst is not None
    ]
    await asyncio.gather(*tasks)
