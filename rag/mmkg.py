""" Multimodal Knowledge Graph Construction with Multimodal Large Language Model
1. convert html to markdown
2. split the markdown into sections with images
3. for each section, check if the section has an image
    - if yes, extract the scene graph and caption the image
4. construct the prompt based the section, scene graph and caption
    - [multimodal prompt]
    - [textual prompt]
5. get the response from multimodal large language model by providing the prompt
6. construct the knowledge graph based on the response
7. save the knowledge graph into the embedding database
"""

import os
import json
import logging
import argparse
import numpy as np

from tqdm import tqdm
from rag.utils import EmbeddingFunc

from rag.llm.hf import hf_text_embed, hf_vision_embed, hf_mm_embed
from rag.multimodal.operate import img_path_to_sg_path
from rag.multimodal.utils import QueryParam
from rag.multimodal.mmrag import MultimodalRAG

from tools.markdown import SECTION_DELIMITER
from tools.markdown import convert_to_markdown, get_document_url
from tools.utils import jsonl_generator, setup_seed, preprocess_args, update_json
from model.embed import load_processor_and_model, load_tokenizer_and_model, load_blip2_model


def get_ragdir_and_logpath(working_dir, graph_id=None):
    rag_dir = os.path.join(working_dir, graph_id) if graph_id else working_dir
    log_file_path = os.path.join(rag_dir, "mmrag.log")
    return rag_dir, log_file_path


def get_ollam_kwargs(args):
    from rag.llm.ollama import ollama_model_complete
    kwargs = dict(
        llm_model_func=ollama_model_complete,
        llm_model_name=args.llm_model_name,
        llm_model_max_async=args.llm_model_max_async,
        llm_model_max_token_size=args.llm_model_max_token_size,
        llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 16000}}
    )

    return kwargs


def get_hf_kwargs(args):
    from rag.llm.hf import hf_model_complete
    kwargs = dict(
        llm_model_func=hf_model_complete,
        llm_model_name=args.llm_model_name,
    )
    return kwargs


def get_vllm_kwargs(args):
    from rag.llm.vllm import vllm_model_complete
    kwargs = dict(
        llm_model_func=vllm_model_complete,
        llm_model_name=args.llm_model_name,
        llm_model_max_async=args.llm_model_max_async,
        llm_model_max_token_size=args.llm_model_max_token_size,
    )
    return kwargs


def init_mmrag(args, rag_dir, log_file_path, 
               unified_vector_storage=False, 
               vector_storage_dir=None,
               use_mm_embedding=False):
    log_level = getattr(logging, args.log_level.upper())

    kb_dir = os.path.dirname(args.kb_file)
    image_dir = os.path.join(kb_dir, "kb_images_640")
    images_map_file = os.path.join(kb_dir, "kb_images_map.json")
    scene_graph_dir = args.scene_graph_dir

    tokenizer, text_model = load_tokenizer_and_model(args.text_embed_model, device=args.device)
    text_embedding_func = EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: hf_text_embed(texts, tokenizer=tokenizer, embed_model=text_model),
    )

    processor, vision_model = load_processor_and_model(args.vision_embed_model, device=args.device)
    vision_embedding_func = EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda images: hf_vision_embed(images, processor=processor, embed_model=vision_model,),
    )

    if use_mm_embedding:
        mm_model, vis_processor, txt_processor = load_blip2_model(args.mm_embed_model, device=args.device)
        mm_embedding_func = EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda data: hf_mm_embed(data,
                vis_processor=vis_processor,
                txt_processor=txt_processor,
                embed_model=mm_model,
                text_type="evidence"
            ),
        )        
    else:
        mm_embedding_func = None

    if args.llm_model_type == "hf":
        llm_kwargs = get_hf_kwargs(args)
    elif args.llm_model_type == "ollama":
        llm_kwargs = get_ollam_kwargs(args)
    elif args.llm_model_type == "vllm":
        llm_kwargs = get_vllm_kwargs(args)
    else:
        raise ValueError(f"Unknown llm_model_type: {args.llm_model_type}")

    rag = MultimodalRAG(
        working_dir=rag_dir,
        log_level=log_level,
        log_file_path=log_file_path,
        image_dir=image_dir,
        scene_graph_dir=scene_graph_dir,
        images_map_file=images_map_file,
        use_mm_embedding=use_mm_embedding,
        embedding_func=text_embedding_func,
        vision_embedding_func=vision_embedding_func,
        mm_embedding_func=mm_embedding_func,
        unified_vector_storage=unified_vector_storage,
        vector_storage_dir=vector_storage_dir,
        vector_storage="FaissVectorDBStorage",
        vector_db_storage_cls_kwargs={
            "cosine_better_than_threshold": 0.2
        },
        entity_extract_max_gleaning=0,
        **llm_kwargs
    )

    return rag


def mmrag_build(args, sample_ids, jsonl_reader, post_process=False):

    if args.max_section_num is not None:
        print("The maximum section number is set to {}, the section over this number will be ignored.".format(args.max_section_num))

    os.makedirs(args.working_dir, exist_ok=True)
    
    doc2kg = {}
    mmrag = None
    progress_bar = tqdm(sample_ids, desc="Building KG")
    for idx in progress_bar:
        progress_bar.set_postfix(id=idx)
        rag_dir, log_file_path = get_ragdir_and_logpath(args.working_dir, graph_id=f'rag_{idx}')

        if mmrag is None:
            # Use separate vector storage for each document when building the KG, enabling parallel processing
            # Then, merge separate vector storages into a unified vector storage, facilitating query
            mmrag = init_mmrag(args, rag_dir, log_file_path, 
                               unified_vector_storage=args.unified_vector_storage, 
                               vector_storage_dir=args.working_dir,
                               use_mm_embedding=args.use_mm_embedding)
        else:
            mmrag.reinit_storages(rag_dir, log_file_path)

        sample = jsonl_reader[idx]
        sample_url = get_document_url(sample)
        sample_md = convert_to_markdown(sample, load_img=True, min_token_size=args.min_token_size, 
                                           max_token_size=args.max_token_size, max_section_num=args.max_section_num)
        mmrag.insert(sample_md, split_by_character=SECTION_DELIMITER)
        doc2kg[sample_url] = rag_dir        
    
    if post_process and mmrag is not None:
        graph_ids = [f'rag_{idx}' for idx in sample_ids]
        rag_dir, log_file_path = get_ragdir_and_logpath(args.working_dir)
        mmrag.post_process_vdb(graph_ids=graph_ids, working_dir=rag_dir, log_file_path=log_file_path)


    json_file = os.path.join(args.working_dir, "doc2kg.json")
    update_json(json_file, doc2kg, file_lock=True)


def check_process_status(args, sample_ids, jsonl_reader):
    mmrag = None

    doc2kg = {}
    failed_ids = []
    for idx in tqdm(sample_ids):
        rag_dir, log_file_path = get_ragdir_and_logpath(args.working_dir, graph_id=f'rag_{idx}')

        if mmrag is None:
            # Use separate vector storage for each document when building the KG, enabling parallel processing
            # Then, merge separate vector storages into a unified vector storage, facilitating query
            mmrag = init_mmrag(args, rag_dir, log_file_path, 
                               unified_vector_storage=args.unified_vector_storage, 
                               vector_storage_dir=args.working_dir,
                               use_mm_embedding=args.use_mm_embedding)
        else:
            mmrag.reinit_storages(rag_dir, log_file_path)

        
        if mmrag.is_processed():
            sample = jsonl_reader[idx]
            sample_url = get_document_url(sample)
            doc2kg[sample_url] = rag_dir
        else:
            failed_ids.append(idx)
            
    print(f"Failed to process {len(failed_ids)} documents.")
    # print(f"Failed document ids: {failed_ids}")

    json_file = os.path.join(args.working_dir, "doc2kg.json")
    update_json(json_file, doc2kg, file_lock=True)


def mmrag_post_process(args, sample_ids):
    # Post-process the vector storage after building the KG
    # 1. Merge separate vector storages into a unified vector storage
    # 2. Build image vector storage for image retrieval
    rag_dir, log_file_path = get_ragdir_and_logpath(args.working_dir)

    mmrag = init_mmrag(args, rag_dir, log_file_path, 
                        unified_vector_storage=args.unified_vector_storage,
                        vector_storage_dir=args.working_dir,
                        use_mm_embedding=args.use_mm_embedding)
    
    graph_ids = [f'rag_{idx}' for idx in sample_ids]
    mmrag.post_process_vdb(graph_ids, working_dir=rag_dir, log_file_path=log_file_path, unified_storage=False)


def mmrag_query(args, question, image_path):
    scene_graph_path = img_path_to_sg_path(image_path, args.scene_graph_dir)
    scene_graph = json.load(open(scene_graph_path, "r"))

    regions = []
    for obj in scene_graph["objects"]:
        regions.append(tuple([round(x, 2) for x in obj["bbox"]]))
    
    query = (question, image_path, regions)

    rag_dir = args.working_dir
    log_file_path = os.path.join(args.working_dir, "query.log")
    
    # Use unified vector storage for query
    mmrag = init_mmrag(args, rag_dir, log_file_path, 
                       unified_vector_storage=True, 
                       vector_storage_dir=args.working_dir,
                       use_mm_embedding=args.use_mm_embedding)
    
    results = mmrag.mm_query(query, 
                             QueryParam(mode="hybrid", top_k=10, 
                                        max_token_for_local_context=200,
                                        max_token_for_global_context=200,
                                        return_image=True))
    print(results)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="build", choices=["build", "check", "post_process", "query"],
                        help="Mode of operation: build the KG, check the process status, post-process the KG, or query the KG.")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--working_dir", type=str, default="./working_dir")
    parser.add_argument("--scene_graph_dir", type=str, default="./scene_graph")
    parser.add_argument("--log_level", type=str, default="warn", 
                        choices=["debug", "info", "warn", "error", "critical"])
    
    # arguments for unified vector storage
    parser.add_argument("--unified_vector_storage", action="store_true", default=False)

    # arguments for large language model and embedding model
    parser.add_argument("--text_embed_model", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--vision_embed_model", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--use_mm_embedding", action="store_true", default=False)
    parser.add_argument("--mm_embed_model", type=str, default="")
    parser.add_argument("--llm_model_type", type=str, default="vllm", choices=["hf", "ollama", "vllm"])
    parser.add_argument("--llm_model_name", type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    parser.add_argument("--llm_model_max_async", type=int, default=16)
    parser.add_argument("--llm_model_max_token_size", type=int, default=32768)

    # arguments for documents
    parser.add_argument("--kb_file", type=str, default="./data/kb.json")
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--idx_file", type=str, default=None)
    parser.add_argument("--min_token_size", type=int, default=200)
    parser.add_argument("--max_token_size", type=int, default=500)
    parser.add_argument("--max_section_num", type=int, default=30)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args = preprocess_args(args)
    setup_seed(seed=42)

    jsonl_reader = list(jsonl_generator(args.kb_file))
    
    if args.start_idx is not None and args.end_idx is not None:
        args.start_idx = max(args.start_idx, 0)
        args.end_idx = min(args.end_idx, len(jsonl_reader))
        
        if args.idx_file is not None:
            print(f"Loading sample ids from {args.idx_file}")
            sample_ids = np.load(args.idx_file).tolist()
            args.end_idx = min(args.end_idx, len(sample_ids))
            sample_ids = sample_ids[args.start_idx: args.end_idx]
        else:
            sample_ids = range(args.start_idx, args.end_idx)
    else:
        # sample_ids = [151, 278, 309, 328, 385, 557, 601, 663, 964, 1216]
        sample_ids = range(0, len(jsonl_reader))
    
    print(f"Total {len(jsonl_reader)} samples, using {len(sample_ids)} samples for building the KG.")

    if args.mode == "build":
        mmrag_build(args, sample_ids, jsonl_reader, post_process=False)
    elif args.mode == "check":
        check_process_status(args, sample_ids, jsonl_reader)
    elif args.mode == "post_process":
        mmrag_post_process(args, sample_ids)
    elif args.mode == "query":
        question = "What is the name of the building?"
        image_path = ".cache/examples/img/merlion.png"
        mmrag_query(args, question, image_path)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
