from tools.config import load_retriever_cfg, load_model_cfg, load_dataset_cfg
from model.generator.vllm import AnswerGeneratorForVLLM


def load_vqa_dataset(dataset_name, split, **kwargs):
    dataset_cfg = load_dataset_cfg().get(dataset_name)
    dataset_cfg.update(kwargs)

    data_dir = dataset_cfg.pop("data_dir", None)
    image_dir = dataset_cfg.pop("image_dir", None)
    
    if dataset_name == "envqa":
        from dataset.en_vqa import EncyclopdicVQA
     
        inat21_dir = dataset_cfg.pop("inat21_dir", None)
        gldv2_dir = dataset_cfg.pop("gldv2_dir", None)
        dataset = EncyclopdicVQA(data_dir, inat21_dir, gldv2_dir, split, **dataset_cfg)
    elif dataset_name == "infoseek":
        from dataset.infoseek import InfoseekVQA
        dataset = InfoseekVQA(data_dir, image_dir, split, **dataset_cfg)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    
    print(f"Using dataset: {dataset_name} with split: {split}")
    return dataset


def load_answer_generator(model_family, device):
    model_cfg = load_model_cfg().get(model_family)
    model_path = model_cfg.pop("model_path")
    answer_generator = AnswerGeneratorForVLLM(model_path, device, **model_cfg)
    print(f"Using answer generator: {model_family} with model path: {model_path}")
    return answer_generator


def load_retriever(args):
    retriever_name = args.retriever
    retriever_cfg = load_retriever_cfg(args.dataset, args.options)

    if retriever_name == "none":
        retriever = None
    elif retriever_name == "oracle":
        from model.retriever import OracleRetriever
        
        oracle_config = retriever_cfg.get("oracle")
        retriever = OracleRetriever(config=oracle_config)
    elif retriever_name == "embed":
        from model.retriever import DocumentRetriever

        embed_models = retriever_cfg.get("embed-models")
        naive_embed_config = retriever_cfg.get("embed")
        embed_model_config = embed_models.get(naive_embed_config.get("model"), {})        
        naive_embed_config.update(embed_model_config)
        
        retriever = DocumentRetriever(config={
            "device": args.device,
            **naive_embed_config
        })
    elif retriever_name == "mmrag":
        from model.retriever import MMRAGRetriever
        
        mmrag_config = retriever_cfg.get("mmrag")
        retriever = MMRAGRetriever(config={
            "device": args.device,
            "working_dir": args.working_dir,
            **mmrag_config
        })
    else:
        raise ValueError(f"Unsupported retriever: {retriever_name}")

    print(f"Using retriever: {retriever_name}")
    return retriever
