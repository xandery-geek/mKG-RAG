from __future__ import annotations

import asyncio
import json
import re
import numpy as np

from typing import Any, AsyncIterator
from rag.utils import (
    logger,
    truncate_list_by_token_size,
    compute_args_hash,
    compute_mdhash_id,
    handle_cache,
    save_to_cache,
    CacheData,
    get_conversation_turns,
)
from rag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema
)

from rag.multimodal.operate import (
    encode_string_by_tiktoken,
    load_roi_image
)

from rag.multimodal.utils import (
    QueryParam, 
    split_string_by_multi_markers,
    list_of_list_to_csv,
    csv_string_to_list,
)

from rag.prompt import GRAPH_FIELD_SEP, PROMPTS, GRAPH_CONTEXT_TEMPLATE, CHUNK_CONTEXT_TEMPLATE


async def naive_query(
    query: dict[str, Any],
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam
) -> str | AsyncIterator[str]:
    
    question, caption = query["question"], query["caption"]
    text_query = f"{caption} {question}".strip()

    results = await chunks_vdb.query(text_query, top_k=query_param.top_k)
    if not len(results):
        return PROMPTS["fail_response"]

    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    # Filter out invalid chunks
    valid_chunks = [
        chunk for chunk in chunks if chunk is not None and "content" in chunk
    ]

    if not valid_chunks:
        logger.warning("No valid chunks found after filtering")
        return PROMPTS["fail_response"]

    maybe_trun_chunks = truncate_list_by_token_size(
        valid_chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    if not maybe_trun_chunks:
        logger.warning("No chunks left after truncation")
        return PROMPTS["fail_response"]

    logger.debug(
        f"Truncate chunks from {len(chunks)} to {len(maybe_trun_chunks)} (max tokens:{query_param.max_token_for_text_unit})"
    )

    section = "\n--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])

    return section


async def _query_by_entity(
    query: str,
    entities_vdb: BaseVectorStorage,
    top_k: int,
):
    # get similar entities
    logger.info(
        f"Query nodes: {query}, top_k: {top_k}, cosine: {entities_vdb.cosine_better_than_threshold}"
    )
    results = await entities_vdb.query(query, top_k=top_k)

    return results

def _format_entities_context(
    node_datas: list[dict[str, Any]],
    query_param: QueryParam
):
    header = ["id", "entity", "type", "description", "rank"]
    if query_param.return_image:
        header.append("images")

    entites_section_list = [header]
    for i, n in enumerate(node_datas):
        row = [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        if query_param.return_image:
            row.append(n.get("images", "UNKNOWN"))
        entites_section_list.append(row)
    entities_context = list_of_list_to_csv(entites_section_list)

    return entities_context


def _format_relations_context(
    relations: list[dict[str, Any]],
    query_param: QueryParam
):
    header = ["id", "source", "target", "description", "weight", "rank"]
    if query_param.return_image:
        header.append("images")

    relations_section_list = [header]
    for i, e in enumerate(relations):
        row = [
                i,
                e["src_id"],
                e["tgt_id"],
                e["description"],
                e["weight"],
                e["rank"]
            ]
        if query_param.return_image:
            row.append(e.get("images", "UNKNOWN"))
        relations_section_list.append(row)
    relations_context = list_of_list_to_csv(relations_section_list)

    return relations_context


def _format_text_units_context(
    text_units: list[dict[str, Any]],
    query_param: QueryParam
):
    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    return text_units_context


async def _get_node_data(
    results: list[dict[str, Any]],
    knowledge_graph_inst: BaseGraphStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
):
    if not len(results):
        return "", "", ""
    
    # get entity information
    node_datas, node_degrees = await asyncio.gather(
        asyncio.gather(
            *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
        ),
        asyncio.gather(
            *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
        ),
    )

    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees, strict=True)
        if n is not None
    ]

    node_datas = await _filter_graph_data(node_datas, query_param.filter_param)

    # get entitytext chunk
    use_text_units, use_relations = await asyncio.gather(
        _find_most_related_text_unit_from_entities(
            node_datas, query_param, text_chunks_db, knowledge_graph_inst
        ),
        _find_most_related_edges_from_entities(
            node_datas, query_param, knowledge_graph_inst
        ),
    )

    len_node_datas = len(node_datas)
    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )
    logger.debug(
        f"Truncate entities from {len_node_datas} to {len(node_datas)} (max tokens:{query_param.max_token_for_local_context})"
    )

    logger.info(
        f"Local query uses {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} chunks"
    )

    # build prompt
    entities_context = _format_entities_context(node_datas, query_param)
    relations_context = _format_relations_context(use_relations, query_param)
    text_units_context = _format_text_units_context(use_text_units, query_param)
    return entities_context, relations_context, text_units_context


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage,
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])

    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )

    # Add null check for node data
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data, strict=True)
        if v is not None and "source_id" in v  # Add source_id check
    }

    all_text_units_lookup = {}
    tasks = []
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges, strict=True)):
        for c_id in this_text_units:
            if c_id not in all_text_units_lookup:
                tasks.append((c_id, index, this_edges))

    results = await asyncio.gather(
        *[text_chunks_db.get_by_id(c_id) for c_id, _, _ in tasks]
    )

    for (c_id, index, this_edges), data in zip(tasks, results, strict=True):
        all_text_units_lookup[c_id] = {
            "data": data,
            "order": index,
            "relation_counts": 0,
        }

        if this_edges:
            for e in this_edges:
                if (
                    e[1] in all_one_hop_text_units_lookup
                    and c_id in all_one_hop_text_units_lookup[e[1]]
                ):
                    all_text_units_lookup[c_id]["relation_counts"] += 1

    # Filter out None values and ensure data has content
    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not all_text_units:
        logger.warning("No valid text units found")
        return []

    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )

    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    logger.debug(
        f"Truncate chunks from {len(all_text_units_lookup)} to {len(all_text_units)} (max tokens:{query_param.max_token_for_text_unit})"
    )

    all_text_units = [t["data"] for t in all_text_units]
    return all_text_units


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = []
    seen = set()

    for this_edges in all_related_edges:
        for e in this_edges:
            sorted_edge = tuple(sorted(e))
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge)

    all_edges_pack, all_edges_degree = await asyncio.gather(
        asyncio.gather(*[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]),
        asyncio.gather(
            *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
        ),
    )
    all_edges_data = [
        {
            "src_id": k[0],
            "tgt_id": k[1],
            "rank": d, 
            **v
        }
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree, strict=True)
        if v is not None
    ]

    all_edges_data = await _filter_graph_data(all_edges_data, query_param.filter_param)

    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )

    logger.debug(
        f"Truncate relations from {len(all_edges)} to {len(all_edges_data)} (max tokens:{query_param.max_token_for_global_context})"
    )

    return all_edges_data


async def _query_by_edge(
    query: str,
    relationships_vdb: BaseVectorStorage,
    top_k: int,
):
    logger.info(
        f"Query edges: {query}, top_k: {top_k}, cosine: {relationships_vdb.cosine_better_than_threshold}"
    )
    results = await relationships_vdb.query(query, top_k=top_k)

    return results


async def _get_edge_data(
    results: list[dict[str, Any]],
    knowledge_graph_inst: BaseGraphStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
):
    if not len(results):
        return "", "", ""

    edge_datas, edge_degree = await asyncio.gather(
        asyncio.gather(
            *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in results]
        ),
        asyncio.gather(
            *[
                knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"])
                for r in results
            ]
        ),
    )

    edge_datas = [
        {
            "src_id": k["src_id"],
            "tgt_id": k["tgt_id"],
            "rank": d,
            **v,
        }
        for k, v, d in zip(results, edge_datas, edge_degree, strict=True)
        if v is not None
    ]

    edge_datas = await _filter_graph_data(edge_datas, query_param.filter_param)

    edge_datas = sorted(
        edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )

    use_entities, use_text_units = await asyncio.gather(
        _find_most_related_entities_from_relationships(
            edge_datas, query_param, knowledge_graph_inst
        ),
        _find_related_text_unit_from_relationships(
            edge_datas, query_param, text_chunks_db
        ),
    )

    logger.info(
        f"Global query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} chunks"
    )

    entities_context = _format_entities_context(use_entities, query_param)
    relations_context = _format_relations_context(edge_datas, query_param)
    text_units_context = _format_text_units_context(use_text_units, query_param)

    return entities_context, relations_context, text_units_context


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    entity_names = []
    seen = set()

    for e in edge_datas:
        if e["src_id"] not in seen:
            entity_names.append(e["src_id"])
            seen.add(e["src_id"])
        if e["tgt_id"] not in seen:
            entity_names.append(e["tgt_id"])
            seen.add(e["tgt_id"])

    node_datas, node_degrees = await asyncio.gather(
        asyncio.gather(
            *[
                knowledge_graph_inst.get_node(entity_name)
                for entity_name in entity_names
            ]
        ),
        asyncio.gather(
            *[
                knowledge_graph_inst.node_degree(entity_name)
                for entity_name in entity_names
            ]
        ),
    )
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(entity_names, node_datas, node_degrees, strict=True)
    ]

    node_datas = await _filter_graph_data(node_datas, query_param.filter_param)

    len_node_datas = len(node_datas)
    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )
    logger.debug(
        f"Truncate entities from {len_node_datas} to {len(node_datas)} (max tokens:{query_param.max_token_for_local_context})"
    )

    return node_datas


async def _find_related_text_unit_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]
    all_text_units_lookup = {}

    async def fetch_chunk_data(c_id, index):
        if c_id not in all_text_units_lookup:
            chunk_data = await text_chunks_db.get_by_id(c_id)
            # Only store valid data
            if chunk_data is not None and "content" in chunk_data:
                all_text_units_lookup[c_id] = {
                    "data": chunk_data,
                    "order": index,
                }

    tasks = []
    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            tasks.append(fetch_chunk_data(c_id, index))

    await asyncio.gather(*tasks)

    if not all_text_units_lookup:
        logger.warning("No valid text chunks found")
        return []

    all_text_units = [{"id": k, **v} for k, v in all_text_units_lookup.items()]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])

    # Ensure all text chunks have content
    valid_text_units = [
        t for t in all_text_units if t["data"] is not None and "content" in t["data"]
    ]

    if not valid_text_units:
        logger.warning("No valid text chunks after filtering")
        return []

    truncated_text_units = truncate_list_by_token_size(
        valid_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    logger.debug(
        f"Truncate chunks from {len(valid_text_units)} to {len(truncated_text_units)} (max tokens:{query_param.max_token_for_text_unit})"
    )

    all_text_units: list[TextChunkSchema] = [t["data"] for t in truncated_text_units]

    return all_text_units


async def extract_keywords(
    text: str,
    param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
) -> tuple[list[str], list[str]]:
    """
    Extract high-level and low-level keywords from the given 'text' using the LLM.
    This method does NOT build the final RAG context or provide a final answer.
    It ONLY extracts keywords (hl_keywords, ll_keywords).
    """

    # 1. Handle cache if needed - add cache type for keywords
    args_hash = compute_args_hash(param.mode, text, cache_type="keywords")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, text, param.mode, cache_type="keywords"
    )
    if cached_response is not None:
        try:
            keywords_data = json.loads(cached_response)
            return keywords_data["high_level_keywords"], keywords_data[
                "low_level_keywords"
            ]
        except (json.JSONDecodeError, KeyError):
            logger.warning(
                "Invalid cache format for keywords, proceeding with extraction"
            )

    # 2. Build the examples
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["keywords_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["keywords_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["keywords_extraction_examples"])
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    # 3. Process conversation history
    history_context = ""
    if param.conversation_history:
        history_context = get_conversation_turns(
            param.conversation_history, param.history_turns
        )

    # 4. Build the keyword-extraction prompt
    kw_prompt = PROMPTS["keywords_extraction"].format(
        query=text, examples=examples, language=language, history=history_context
    )

    len_of_prompts = len(encode_string_by_tiktoken(kw_prompt))
    logger.debug(f"[kg_query]Prompt Tokens: {len_of_prompts}")

    # 5. Call the LLM for keyword extraction
    use_model_func = global_config["llm_model_func"]
    result = await use_model_func(kw_prompt, keyword_extraction=True)

    # 6. Parse out JSON from the LLM response
    match = re.search(r"\{.*\}", result, re.DOTALL)
    if not match:
        logger.error("No JSON-like structure found in the LLM respond.")
        return [], []
    try:
        keywords_data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return [], []

    hl_keywords = keywords_data.get("high_level_keywords", [])
    ll_keywords = keywords_data.get("low_level_keywords", [])

    # 7. Cache only the processed keywords with cache type
    if hl_keywords or ll_keywords:
        cache_data = {
            "high_level_keywords": hl_keywords,
            "low_level_keywords": ll_keywords,
        }
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=json.dumps(cache_data),
                prompt=text,
                quantized=quantized,
                min_val=min_val,
                max_val=max_val,
                mode=param.mode,
                cache_type="keywords",
            ),
        )
    return hl_keywords, ll_keywords


def extract_image_query(
    image: str,
    regions: list[tuple],
)-> tuple[np.ndarray, list[np.ndarray]]:
    """
    Extract visual query from the given image and regions.
    """
    roi_images = load_roi_image(image, regions, return_original=True)
    hl_images, ll_images = roi_images[0], roi_images[1:]
    return hl_images, ll_images


def _separate_results_by_type(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Separate the results into two groups: entities and relationships.
    """
    entity_results, relation_results = [], []
    for result in results:
        if result.get("entity_name", None):
            entity_results.append(result)
        else:
            relation_results.append(result)
    
    return entity_results, relation_results


def _deduplicate_results(results: list[dict[str, Any]], enable_sort=True) -> list[dict[str, Any]]:
    """
    1. Deduplicate the results based on the entity_name or src_id-tgt_id pair.
    2. Sort the results by the weight in descending order.
    """
    results_dict = {}
    for result in results:
        if result.get("entity_name", None):
            key = result["entity_name"]
        else:
            key = tuple(sorted([result["src_id"], result["tgt_id"]]))

        if key not in results_dict:
            result["weight"] = result["weight"] * result["distance"]
            results_dict[key] = result
        else:
            results_dict[key]["weight"] += result["weight"] * result["distance"]
    
    results = list(results_dict.values())

    if enable_sort:
        results = sorted(results, key=lambda x: x["weight"], reverse=True)
    return results 


async def _query_by_image(
    images: np.ndarray | list[np.ndarray],
    images_vdb: BaseVectorStorage,
    top_k: int
)-> list[dict[str, Any]]:
    if isinstance(images, np.ndarray):
        images = [images]
    
    tasks = [images_vdb.query(image, top_k=top_k) for image in images]
    results_list = await asyncio.gather(*tasks)
    results = [r for results in results_list for r in results]

    return results


async def _query_by_mm(
    images: str | list[str],
    texts: str | list[str],
    multimodal_vdb: BaseVectorStorage,
    top_k: int
)-> list[dict[str, Any]]:
    images = images if isinstance(images, list) else [images]
    texts = texts if isinstance(texts, list) else [texts]
    assert len(images) == len(texts), "The number of images and texts must be the same"

    images = [load_roi_image(image, regions=[], return_original=True)[0] for image in images]
    tasks = [multimodal_vdb.query((image, text), top_k=top_k) for image, text in zip(images, texts, strict=True)]
    results_list = await asyncio.gather(*tasks)
    results = [r for results in results_list for r in results]

    return results


async def _collect_data_from_entity_and_relation(
    entity_results: list[dict[str, Any]],
    relation_results: list[dict[str, Any]],
    knowledge_graph: BaseGraphStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam
)-> tuple[list[str], list[str], list[str]]:

    entities_context, relations_context, text_units_context = [], [], []
    if len(entity_results) > 0:
        entities, relations, text_units = await _get_node_data(
            entity_results,
            knowledge_graph,
            text_chunks_db,
            query_param
        )
        entities_context.append(entities)
        relations_context.append(relations)
        text_units_context.append(text_units)
    
    if len(relation_results) > 0:
        entities, relations, text_units = await _get_edge_data(
            relation_results,
            knowledge_graph,
            text_chunks_db,
            query_param
        )
    
        entities_context.append(entities)
        relations_context.append(relations)
        text_units_context.append(text_units)

    return entities_context, relations_context, text_units_context


async def _collect_data_without_traversal(
    entity_results: list[dict[str, Any]],
    relation_results: list[dict[str, Any]],
    knowledge_graph_inst: BaseGraphStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam
)-> tuple[list[str], list[str], list[str]]:

    node_datas, node_degrees = await asyncio.gather(
        asyncio.gather(
            *[knowledge_graph_inst.get_node(r["entity_name"]) for r in entity_results]
        ),
        asyncio.gather(
            *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in entity_results]
        ),
    )
    
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(entity_results, node_datas, node_degrees, strict=True)
        if n is not None
    ]

    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )

    edge_datas, edge_degree = await asyncio.gather(
        asyncio.gather(
            *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in relation_results]
        ),
        asyncio.gather(
            *[knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"]) for r in relation_results]
        ),
    )

    edge_datas = [
        {"src_id": k["src_id"], "tgt_id": k["tgt_id"], "rank": d, **v}
        for k, v, d in zip(relation_results, edge_datas, edge_degree, strict=True)
        if v is not None
    ]

    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )

    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )

    entity_text_units = await _find_related_text_unit_from_relationships(node_datas, query_param, text_chunks_db)
    relation_text_units = await _find_related_text_unit_from_relationships(edge_datas, query_param, text_chunks_db)

    text_units = entity_text_units + relation_text_units
    entities_context = _format_entities_context(node_datas, query_param)
    relations_context = _format_relations_context(edge_datas, query_param)
    text_units_context = _format_text_units_context(text_units, query_param)

    return [entities_context], [relations_context], [text_units_context]


async def _collect_data_from_graph(
    results: list[dict[str, Any]],
    knowledge_graph_inst: BaseGraphStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam
)-> tuple[list[str], list[str], list[str]]:
    
    if not len(results):
        return [], [], []
    
    # separate entity and relation results
    entity_results, relation_results = _separate_results_by_type(results)
    entity_results = _deduplicate_results(entity_results)[:query_param.top_k]
    relation_results = _deduplicate_results(relation_results)[:query_param.top_k]

    if query_param.traverse_hop > 0:
        data = await _collect_data_from_entity_and_relation(
            entity_results, relation_results, knowledge_graph_inst, text_chunks_db, query_param)
    else:
        data = await _collect_data_without_traversal(
            entity_results, relation_results, knowledge_graph_inst, text_chunks_db, query_param
        )

    return data


async def _filter_graph_data(
    data: list[dict[str, Any]],
    filter_param: dict[str, Any]
):
    if not len(data) or filter_param is None:
        return data

    question = filter_param["question"]
    embedding_func = filter_param["embedding_func"]
    cosine_threshold = filter_param["cosine_threshold"]
    min_data_number = filter_param["min_data_number"]

    data_desc = [d["description"] for d in data]
    embeddings = await embedding_func([question] + data_desc)

    scores = np.dot(embeddings[1:], embeddings[0].T).flatten()
    filtered_data = [d for d, s in zip(data, scores, strict=True) if s > cosine_threshold]

    if len(filtered_data) < min_data_number:
        sorted_indices = np.argsort(scores)[::-1]
        filtered_data = [data[i] for i in sorted_indices[:min_data_number]]
        logger.info(f"No data satisfy the cosine threshold, use top {min_data_number} data instead")
    else:
        logger.info(f"Filter data by question: {len(data)} -> {len(filtered_data)}")

    return filtered_data


def _init_graph_filter_param(
    query: dict[str, Any],
    query_param: QueryParam, 
    global_config: dict[str, str]
):
    if query_param.filter_strategy == "none":
        query_param.filter_param = None

    elif query_param.filter_strategy == "quesiton":
        query_param.filter_param = {
            "question": query["question"],
            "embedding_func": global_config["embedding_func"],
            "cosine_threshold": 0.5,
            "min_data_number": 3
        }
    else:
        raise ValueError(f"Invalid filter strategy: {query_param.filter_strategy}")


async def mm_kg_query(
    query: dict[str, Any],
    knowledge_graph_inst: BaseGraphStorage,
    text_chunks_db: BaseKVStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    images_vdb: BaseVectorStorage,
    multimodal_vdb: BaseVectorStorage,
    query_param: QueryParam,
    global_config: dict[str, str]
) -> str | AsyncIterator[str]:

    # initialize the query parameters
    _init_graph_filter_param(query, query_param, global_config)

    question, caption, image, regions = \
        query["question"], query["caption"], query["image_path"], query["regions"]
    
    # # retrieve rlevant results by image and text
    mm_results = []
    if query_param.strategy in ["multimodal"]:
        text_query = question

        mm_results = await _query_by_mm(
            image,
            question,
            multimodal_vdb,
            query_param.top_k * 2
        )

    # retrieve rlevant results by image
    image_results = []
    if query_param.strategy in ["image", "text-image"]:
        top_k = query_param.top_k
        if query_param.strategy == "text-image":
            top_k = top_k // 2

        hl_images, ll_images = extract_image_query(image, regions)

        if query_param.mode == "local":
            image_results.extend(await _query_by_image(ll_images, images_vdb, top_k * 2))
        elif query_param.mode == "global":
            image_results.extend(await _query_by_image(hl_images, images_vdb, top_k * 2))
        elif query_param.mode == "hybrid":
            ll_results, hl_results = await asyncio.gather(
                _query_by_image(ll_images, images_vdb, top_k),
                _query_by_image(hl_images, images_vdb, top_k),
            )
            image_results.extend(ll_results + hl_results)
        else:
            raise ValueError(f"Invalid query mode: {query_param.mode}")
    
    # retrieve relevant results by text
    text_results = []
    if query_param.strategy in ["text", "text-image"]:
        top_k = query_param.top_k
        if query_param.strategy == "text-image":
            top_k = top_k // 2
        
        text_query = f"{caption} {question}".strip()
        hl_keywords, ll_keywords = text_query, text_query

        if query_param.mode == "local":
            text_results.extend(await _query_by_entity(ll_keywords, entities_vdb, top_k * 2))
        elif query_param.mode == "global":
            text_results.extend(await _query_by_edge(hl_keywords, relationships_vdb, top_k * 2))
        elif query_param.mode == "hybrid":
            ll_results, hl_results = await asyncio.gather(
                _query_by_entity(ll_keywords, entities_vdb, top_k),
                _query_by_edge(hl_keywords, relationships_vdb, top_k),
            )
            text_results.extend(ll_results + hl_results)
        else:
            raise ValueError(f"Invalid query mode: {query_param.mode}")

    # build the context
    entities_context, relations_context, text_units_context = [], [], []
    if len(image_results) > 0:
        # normalize the weight by the max weight
        max_weight = max([result["weight"] for result in image_results])
        for result in image_results:
            result["weight"] = result["weight"] / max_weight if max_weight > 0 else 1.0

        data = await _collect_data_from_graph(
            image_results,
            knowledge_graph_inst,
            text_chunks_db,
            query_param
        )

        entities_context.extend(data[0])
        relations_context.extend(data[1])
        text_units_context.extend(data[2])
    
    if len(text_results) > 0:
        # initialize the weight for text results
        for result in text_results:
            result["weight"] = 1.0

        data = await _collect_data_from_graph(
            text_results,
            knowledge_graph_inst,
            text_chunks_db,
            query_param
        )

        entities_context.extend(data[0])
        relations_context.extend(data[1])
        text_units_context.extend(data[2])
    
    if len(mm_results) > 0:
        # initialize the weight for multimodal results
        for result in mm_results:
            result["weight"] = 1.0

        data = await _collect_data_from_graph(
            mm_results,
            knowledge_graph_inst,
            text_chunks_db,
            query_param
        )

        entities_context.extend(data[0])
        relations_context.extend(data[1])
        text_units_context.extend(data[2])

    context = _build_mm_query_context(entities_context, relations_context, text_units_context, query_param)

    return context


def _process_combine_contexts(
    contexts_list: list[str], 
    sort_by_rank: bool = False, 
    rank_idx: int = -1
) -> str:
    
    if not contexts_list:
        return ""

    header = None
    all_item_list = []
    for contexts in contexts_list:
        item_list = csv_string_to_list(contexts.strip())
        
        if header is None:
            header = item_list[0]
        else:
            assert header == item_list[0], "Header mismatch"

        all_item_list.extend(item_list[1:])
    
    seen = set()
    combined_items = []
    for item in all_item_list:
        item_str = ",\t".join(item[1:])
        rank = int(item[rank_idx]) if sort_by_rank else 0
        if item_str not in seen:
            seen.add(item_str)
            combined_items.append((item_str, rank))

    if sort_by_rank:
        combined_items = sorted(combined_items, key=lambda x: x[1], reverse=True)
    
    combined_items = [item[0] for item in combined_items]

    combined_context = [",\t".join(header)]
    for i, item in enumerate(combined_items, start=1):
        combined_context.append(f"{i},\t{item}")

    combined_context = "\n".join(combined_context)
    return combined_context


def _combine_contexts(entities, relationships, sources) -> tuple[str, str, str]:   
    # Combine and deduplicate the entities, relationships, and sources
    if isinstance(entities, str):
        entities = [entities]
    if isinstance(relationships, str):
        relationships = [relationships]
    if isinstance(sources, str):
        sources = [sources]

    combined_entities = _process_combine_contexts(entities, sort_by_rank=True, rank_idx=-1)
    combined_relationships = _process_combine_contexts(relationships, sort_by_rank=True, rank_idx=-1)
    combined_sources = _process_combine_contexts(sources)

    return combined_entities, combined_relationships, combined_sources


def _build_mm_query_context(
    entities_context: list[str],
    relations_context: list[str],
    text_units_context: list[str],
    query_param: QueryParam
) -> str:
    # combine all contexts by process_combine_contexts
    entities_context, relations_context, text_units_context = _combine_contexts(
        entities_context, relations_context, text_units_context
    )
    
    if not entities_context.strip() and not relations_context.strip():
        return None

    graph_context = GRAPH_CONTEXT_TEMPLATE.format(
        entities_context=entities_context,
        relations_context=relations_context)
    chunk_context = CHUNK_CONTEXT_TEMPLATE.format(
        text_units_context=text_units_context)

    if query_param.context_mode == "hybrid":
        return f"{graph_context}\n{chunk_context}"
    elif query_param.context_mode == "chunk":
        return chunk_context
    elif query_param.context_mode == "graph":
        return graph_context
    else:
        raise ValueError(f"Invalid context mode: {query_param.context_mode}")