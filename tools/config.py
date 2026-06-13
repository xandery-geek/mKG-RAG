import os
from omegaconf import OmegaConf


def merge_options(cfg, options):
    if options:
        options = OmegaConf.from_dotlist(options)
        cfg = OmegaConf.merge(cfg, options)
    return cfg


def load_dataset_cfg(options=None):
    dataset_cfg_file = "config/dataset.yaml"
    if os.path.exists(dataset_cfg_file):
        print(f"Loading dataset configuration from {dataset_cfg_file}")
        dataset_cfg = OmegaConf.load(dataset_cfg_file)
        dataset_cfg = merge_options(dataset_cfg, options)
        dataset_cfg = OmegaConf.to_container(dataset_cfg, resolve=True)
    else:
        print(f"No dataset configuration found at {dataset_cfg_file}.")
        dataset_cfg = {}
    return dataset_cfg


def load_model_cfg(options=None):
    model_cfg_file = f"config/model.yaml"
    if os.path.exists(model_cfg_file):
        print(f"Loading model configuration from {model_cfg_file}")
        model_cfg = OmegaConf.load(model_cfg_file)
        model_cfg = merge_options(model_cfg, options)
        model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
    else:
        print(f"No model configuration found at {model_cfg_file}.")
        model_cfg = {}
    return model_cfg


def load_retriever_cfg(dataset, options=None):
    retriever_cfg_file = f"config/{dataset}.yaml"

    if os.path.exists(retriever_cfg_file):
        print(f"Loading retriever configuration from {retriever_cfg_file}")
        retriever_cfg = OmegaConf.load(retriever_cfg_file)
        retriever_cfg = merge_options(retriever_cfg, options)
        retriever_cfg = OmegaConf.to_container(retriever_cfg, resolve=True)
    else:
        print(f"No retriever configuration found for {dataset}")
        retriever_cfg = {}
    return retriever_cfg
