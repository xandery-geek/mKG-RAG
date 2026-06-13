import os
import json
import json
import faiss

from abc import ABC, abstractmethod
from tqdm import tqdm
from faiss import write_index, read_index

from dataset.knowledge_base import KnowledgeBase
from model.embed import ClipEmbedding, NomicVisionEmbedding, NomicTextEmbedding, EvaClipEmbedding


class BaseRetriever(ABC):
    def __init__(self, config):
        self.config = config
        self.retrieval_param = self.config.get("params", {})
        print("Config: ", config)
    
    @abstractmethod
    def retrieve(self, queries, images=None, attributes=None, **kwargs):
        """
        Retrieve contexts based on the queries and images.
        Args:
            queries (list): List of queries to retrieve contexts for.
            images (list, optional): List of images corresponding to the queries.
            attributes (list, optional): Additional attributes of queries.
            **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError()

    def _load_map_file(self, map_file):
        if map_file is None or not os.path.exists(map_file):
            raise ValueError(f"Map file {map_file} does not exist")
        else:
            print(f"Loading map file from {map_file}")
            with open(map_file, "r") as f:
                return json.load(f)


class OracleRetriever(BaseRetriever):
    """"
    Oracle retriever that returns the ground truth article or section from the knowledge base
    """
    RETURN_TYPE = ["article", "section", "url"]
    def __init__(self, config):
        super().__init__(config)
        self.knowledge_base = KnowledgeBase(config.get("kb_path"))

        self.top_k = self.retrieval_param.get("top_k", 1)
        self.return_type = self.retrieval_param.get("return_type", "article")
        
        assert self.return_type in self.RETURN_TYPE, f"Unsupported return type {self.return_type}"

    def get_evidence_articles(self, wiki_url):
        context = ""
        for url in wiki_url:
            article = self.knowledge_base.get_article(url)
            if article != "":
                context += article + "\n"
        return context

    def get_evidence_sections(self, wiki_url, section_id):
        context = ""
        for url, sec_id in zip(wiki_url, section_id, strict=True):
            section = self.knowledge_base.get_section(url, sec_id)
            if section != "":
                context += section + "\n"
        return context
    
    def retrieve(self, queries, images=None, attributes=None, **kwargs):
        contexts = []
        for _, attribute in zip(queries, attributes, strict=True):
            wiki_url = attribute.get("wiki_url", None)
            if wiki_url is None:
                contexts.append("")
                continue
            else:
                wiki_url = wiki_url[:self.top_k]
            
            if self.return_type == "url":
                context = "|".join(wiki_url)
            elif self.return_type == "article":
                context = self.get_evidence_articles(wiki_url)
            elif self.return_type == "section":
                evidence_section_id = attribute.get("evidence_section_id", None)
                if evidence_section_id is None or len(wiki_url) != len(evidence_section_id):
                    context = self.get_evidence_articles(wiki_url)
                else:
                    context = self.get_evidence_sections(wiki_url, evidence_section_id)
            else:
                raise ValueError(f"Unsupported return type {self.return_type}")
            
            contexts.append(context)
        return contexts


class MMRAGRetriever(BaseRetriever):
    def __init__(self, config):
        from rag.prompt import PROMPTS
        from rag.llm.hf import hf_model_complete
        from rag.multimodal.operate import img_path_to_sg_path

        super().__init__(config)
        # basic config
        self.device = config.get("device", "cpu")
        self.kb_path = config.get("kb_path")
        self.scene_graph_dir = config.get("scene_graph_dir")
        self.img_path_to_sg_path = img_path_to_sg_path
        
        # multimodal rag config
        self.rag_dir = config.get("rag_dir")
        self.log_file_path = os.path.join(config.get("working_dir"), config.get("log_dirname", "logs"), 
                                          "mmrag_retriever.log")

        # llm config
        self.llm_model_name = config.get("llm_model_name", "llama3.2-vision:latest")
        self.llm_model_func = hf_model_complete

        # embedding config
        self.text_embed_model = config.get("text_embed_model", "nomic-ai/nomic-embed-text-v1.5")
        self.text_embedding_func = self._init_embedding_func(modal="text")
        
        self.vision_embed_model = config.get("vision_embed_model", "openai/clip-vit-large-patch14-336")
        self.vision_embedding_func = self._init_embedding_func(modal="vision")

        self.use_mm_embedding = config.get("use_mm_embedding", False)
        if self.use_mm_embedding:
            self.mm_embed_model = config.get("mm_embed_model", "blip2_qm_retriever")
            self.mm_embedding_func = self._init_embedding_func(modal="mm")
        else:
            self.mm_embedding_func = None

        # response config
        self.fail_response = PROMPTS["fail_response"]

        # other config
        self.id_to_docurl = self._load_map_file(config.get("id_to_docurl", None))
        self.doc2kg = self._load_map_file(os.path.join(self.rag_dir, "doc2kg.json"))
        
        self.rag = self._init_mmrag(self.rag_dir, vector_storage_dir=self.rag_dir)
        self.query_param = self._init_query_param()

        self.retrieve_strategy = self.query_param.strategy
        self.retrieve_from_dynamic_graph = self.query_param.retrieve_from_dynamic_graph
        
    def _init_mmrag(self, rag_dir, vector_storage_dir=None):
        import logging
        from rag.multimodal.mmrag import MultimodalRAG
        
        rag = MultimodalRAG(
            working_dir=rag_dir,
            log_level=getattr(logging, 'WARN'),
            log_file_path=self.log_file_path,
            llm_model_func=self.llm_model_func,
            llm_model_name=self.llm_model_name,
            use_mm_embedding=self.use_mm_embedding,
            embedding_func=self.text_embedding_func,
            vision_embedding_func=self.vision_embedding_func,
            mm_embedding_func=self.mm_embedding_func,
            unified_vector_storage=False,
            vector_storage_dir=vector_storage_dir,
            vector_storage="FaissVectorDBStorage",
            vector_db_storage_cls_kwargs={
                "cosine_better_than_threshold": 0.2
            }
        )

        return rag
    
    def _init_embedding_func(self, modal="text"):
        from rag.utils import EmbeddingFunc
        from rag.llm.hf import hf_vision_embed, hf_text_embed, hf_mm_embed
        from model.embed import load_processor_and_model, load_tokenizer_and_model, load_blip2_model

        if modal == "text":
            tokenizer, text_model = load_tokenizer_and_model(self.text_embed_model, device=self.device)
            embedding_func = EmbeddingFunc(
                embedding_dim=768,
                max_token_size=8192,
                func=lambda texts: hf_text_embed(
                    texts,
                    tokenizer=tokenizer,
                    embed_model=text_model,
                ),
            )
        elif modal == "vision":
            processor, vision_model = load_processor_and_model(self.vision_embed_model, device=self.device)
            embedding_func = EmbeddingFunc(
                embedding_dim=768,
                max_token_size=8192,
                func=lambda images: hf_vision_embed(
                    images,
                    processor=processor,
                    embed_model=vision_model,
                ),
            )
        elif modal == "mm":
            mm_model, vis_processor, txt_processor = load_blip2_model(self.mm_embed_model, device=self.device)
            embedding_func = EmbeddingFunc(
                embedding_dim=768,
                max_token_size=8192,
                func=lambda data: hf_mm_embed(
                    data,
                    vis_processor=vis_processor,
                    txt_processor=txt_processor,
                    embed_model=mm_model,
                    text_type="question"
                ),
            )
        else:
            raise ValueError(f"Invalid modal {modal}")

        return embedding_func
    
    def _init_query_param(self):
        from rag.multimodal.utils import QueryParam

        query_param = QueryParam(strategy=self.retrieval_param.get("strategy", "image"),
                                 mode=self.retrieval_param.get("mode", "hybrid"),
                                 top_k=self.retrieval_param.get("top_k", 10),
                                 traverse_hop=self.retrieval_param.get("traverse_hop", 1),
                                 max_token_for_local_context=self.retrieval_param.get("max_token_for_graph", 512),
                                 max_token_for_global_context=self.retrieval_param.get("max_token_for_graph", 512),
                                 retrieve_from_dynamic_graph=self.retrieval_param.get("retrieve_from_dynamic_graph", True),
                                 context_mode=self.retrieval_param.get("context_mode", "hybrid"),
                                 return_image=False)

        return query_param
    
    def _get_candidate_graph_ids(self, question_id):
        doc_top_k = self.retrieval_param.get("doc_top_k", 1)

        # get similar documents which contain images similar to the input image
        doc_urls = self.id_to_docurl.get(str(question_id), "")
        doc_urls = doc_urls.split("|")[:doc_top_k]
        doc_urls = [doc_url for doc_url in doc_urls if doc_url in self.doc2kg]

        # get the graph ids for the documents
        graph_ids = []
        for doc_url in doc_urls:
            kg_dir = self.doc2kg.get(doc_url, "")
            if kg_dir != "":
                graph_ids.append(os.path.basename(kg_dir))
        
        return graph_ids
    
    def _mm_retrieve(self, query, param, graph_ids=[]):
        try:
            result = self.rag.mm_query(query, param, graph_ids)
            if result is None or result == self.fail_response:
                result = ""
        except Exception as e:
            print(f"Error in retrieving context for {query}: {e}")
            result = ""
        
        return result

    def _retrieve(self, query, image=None, attribute=None, **kwargs):
        # get the candidate graph ids based on the question id
        graph_ids = self._get_candidate_graph_ids(attribute["question_id"])
        if len(graph_ids) == 0:
            return ""
        
        query_dict = {"question": query, "caption": "", "image_path": "", "regions": []}
        # prepare the image query
        if self.retrieve_strategy in ["image", "multimodal", "text-image"]:
            image_path = attribute["image_path"]
            scene_graph_path = self.img_path_to_sg_path(image_path, self.scene_graph_dir)
            scene_graph = json.load(open(scene_graph_path, "r"))

            regions = []
            for obj in scene_graph["objects"]:
                regions.append(tuple([round(x, 2) for x in obj["bbox"]]))
            
            if regions == [] and self.query_param.mode == "local":
                # if no object detected, use the entire image
                regions.append((0, 0, 1, 1))
            
            query_dict["image_path"] = image_path
            query_dict["regions"] = regions

        # prepare the text query
        if self.retrieve_strategy in ["naive", "text", "multimodal", "text-image"]:
            query_dict["caption"] = attribute["caption"]
        
        if self.retrieve_from_dynamic_graph:
            # merge all graphs dynamically and retrieve from the merged graph
            context = self._mm_retrieve(query_dict, self.query_param, graph_ids)
        else:
            # retrieve from each graph separately and merge the results
            context = ""
            for graph_id in graph_ids:
                kg_dir = os.path.join(self.rag_dir, graph_id)
                assert os.path.exists(kg_dir), f"Graph {graph_id} does not exist"
                
                self.rag.reinit_storages(kg_dir, self.log_file_path, self.llm_model_func)
                context += self._mm_retrieve(query_dict, self.query_param)
        return context
    
    def retrieve(self, queries, images=None, attributes=None, **kwargs):
        contexts = []
        for query, image, attribute in zip(queries, images, attributes, strict=True):
            context = self._retrieve(query, image, attribute, **kwargs)
            contexts.append(context)
        return contexts


class DocumentRetriever(BaseRetriever):
    """
    Document retriever that uses faiss index to retrieve documents, support both text and image retrieval
    """
    SUPPORTED_MODE = ["i2i", "t2t", "i2t", "t2i", 'mm']

    def __init__(self, config):
        super().__init__(config)
        
        self.device = self.config.get("device", "cpu")
        self.embed_type = self.config.get("embed_type", "nomic")
        self.embed_path = self.config.get("embed_path")
        self.kb_path = self.config.get("kb_path")
        self.index_path = self.config.get("index_path")
        self.mode = self.config.get("mode", "i2i")

        assert self.mode in self.SUPPORTED_MODE, f"Unsupported mode {self.mode}"
        
        self.src_modality, self.tgt_modality = self._parse_modality()
        self.embed_model = self._load_embed_model()
        self.knowledge_base = KnowledgeBase(self.kb_path)
        self.faiss_index, self.faiss_index_meta = self._load_faiss_index(self.index_path)

        if config.get("id_to_docurl", None):
            try:
                self.id_to_docurl = self._load_map_file(config.get("id_to_docurl", None))
            except ValueError:
                self.id_to_docurl = None
        else:
            self.id_to_docurl = None
    
    def _parse_modality(self):
        if self.mode == "mm":
            src_modality, tgt_modality = "mm", "mm"
        else:
            src_modality = "image" if self.mode in ["i2i", "i2t"] else "text"
            tgt_modality = "image" if self.mode in ["t2i", "i2i"] else "text"
        return src_modality, tgt_modality
    
    def _load_embed_model(self):
        if self.embed_type == "nomic":
            if self.src_modality == "text":
                embed_model = NomicTextEmbedding(self.embed_path, self.device)
            elif self.src_modality == "image":
                embed_model = NomicVisionEmbedding(self.embed_path, self.device)
            else:
                raise ValueError(f"Unsupported mode {self.src_modality} for nomic embedding")
        elif self.embed_type == "clip":
            if self.src_modality in ["image", "text"]:
                embed_model = ClipEmbedding(self.embed_path, self.device)
            else:
                raise ValueError(f"Unsupported mode {self.src_modality} for clip embedding")
        elif self.embed_type == "eva":
            if self.src_modality in ["image", "text"]:
                embed_model = EvaClipEmbedding(self.embed_path, self.device)
            else:
                raise ValueError(f"Unsupported mode {self.src_modality} for eva embedding")
        else:
            raise ValueError(f"Unsupported embedding type {self.embed_type}")
    
        return embed_model
    
    def update_faiss_index(self, save_index_per_batch=100):
        resume = self.config.get("resume", False)
        if resume:
            print(f"Updating faiss index for {self.tgt_modality} ...")
        else:
            if self.faiss_index is not None:
                raise ValueError("Faiss index already exists, please set resume=True to update the index")
            else:
                print(f"Building faiss index for {self.tgt_modality} ...")

        if self.tgt_modality == "image":
            image_dir = self.config.get("image_dir")
            batch_size = self.config.get("batch_size", 32)
            self.build_image_index(image_dir, batch_size=batch_size, save_index_per_batch=save_index_per_batch, resume=resume)
        elif self.tgt_modality == "text":
            batch_size = self.config.get("batch_size", 32)
            self.build_text_index(batch_size, save_index_per_batch=save_index_per_batch, resume=resume)
        elif self.tgt_modality == "mm":
            image_dir = self.config.get("image_dir")
            batch_size = self.config.get("batch_size", 32)
            self.build_mm_index(image_dir, batch_size=batch_size, save_index_per_batch=save_index_per_batch, resume=resume)
        else:
            raise ValueError(f"Unsupported modality {self.tgt_modality} for faiss index update")

    def retrieve(self, texts=None, images=None, attributes=None, **kwargs):
        if self.faiss_index is None:
            raise ValueError("Faiss index is not loaded. Please update the index.")

        top_k = self.retrieval_param.get("top_k", 10)
        return_doc = self.retrieval_param.get("return_doc", False)

        texts = [texts] if not isinstance(texts, (list, tuple)) else texts
        images = [images] if not isinstance(images, (list, tuple)) else images
        attributes = [attributes] if not isinstance(attributes, (list, tuple)) else attributes
        queries = [
            self._get_valid_query(text, image, attribute) 
            for text, image, attribute in zip(texts, images, attributes, strict=True)
        ]

        if self.tgt_modality == "image":
            results = self._retrieve_image(queries, top_k=top_k)
        elif self.tgt_modality == "text":
            results = self._retrieve_text(queries, top_k=top_k)
        elif self.tgt_modality == "mm":
            results = self._retrieve_mm(queries, top_k=top_k)
        else:
            raise ValueError(f"Unsupported modality {self.tgt_modality} for retrieval")

        contexts = []
        for top_k_entries in results:
            # remove duplicated documents
            unique_doc_urls = []
            for entry in top_k_entries:
                doc_url = entry["doc_url"]
                
                if not isinstance(doc_url, list):
                    doc_url = [doc_url]
                unique_doc_urls.extend([url for url in doc_url if url not in unique_doc_urls])

            unique_doc_urls = unique_doc_urls[:top_k]

            if return_doc:
                context = [self.knowledge_base.get_article(doc_url) for doc_url in unique_doc_urls]
                context = "\n".join(context)
            else:
                context = '|'.join(unique_doc_urls)
            
            contexts.append(context)

        return contexts
    
    def faiss_cpu_to_gpu(self):
        res = faiss.StandardGpuResources()
        self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)

    def _load_faiss_index(self, index_path):
        """
        Load the faiss index.
        Args:
            index_path: The path to load the faiss index.
        Returns:
            faiss_index: The faiss index.
            faiss_index_meta: The meta data of the faiss index.
        """
        if os.path.exists(index_path):
            print("Loading faiss index from {} ...".format(index_path))
            faiss_index = read_index(index_path)
            faiss_index_meta = json.load(open(index_path.replace(".faiss", "_meta.json"), "r"))
            print("Faiss index loaded with {} entries.".format(faiss_index.ntotal))
        else:
            faiss_index = None
            faiss_index_meta = None
            print("Faiss index not found at {}. Please create the index.".format(index_path))

        return faiss_index, faiss_index_meta
    
    def _init_faiss_index(self):
        if self.faiss_index is None:
            # initialize faiss index if not exist
            faiss_index = faiss.IndexFlatIP(self.embed_model.embedding_dim)
            faiss_index_meta = {"embed_type": self.embed_type, "embed_path": self.embed_path, 
                                "index_to_image": {}, "index_to_doc": {}}
            
            return faiss_index, faiss_index_meta
        else:
            if self.faiss_index_meta["embed_type"] != self.embed_type \
                or self.faiss_index_meta["embed_path"] != self.embed_path:
                raise ValueError("The embedding type or path does not match the existing faiss index.")
            if self.faiss_index.d != self.embed_model.embedding_dim:
                raise ValueError("The dimension of the faiss index does not match the input features.")
            
        return self.faiss_index, self.faiss_index_meta
    
    def build_mm_index(self, image_dir, batch_size=1, save_index_per_batch=100, resume=False):
        from torch.utils.data import DataLoader
        from dataset.image import ImageTextDataset
        from dataset.utils import image_text_collate_fn

        with open(os.path.join(os.path.dirname(image_dir), "kb_images_map.json"), "r") as f:
            kb_images_map = json.load(f)

        print("Collecting data from the knowledge base ...")
        collected_data = {}
        for doc_url, doc_data in self.knowledge_base.get_all_items():
            ## Using section text as text information
            section_texts = doc_data["section_texts"]
            for img_url, img_sec_idx in zip(doc_data["image_urls"], doc_data["image_section_indices"], strict=True):
                image_file = kb_images_map.get(img_url, "")
                
                if image_file == "":
                    continue
                image_path = os.path.join(image_dir, image_file)

                if image_path in collected_data:
                    # check if the section text is already collected
                    for idx, item in enumerate(collected_data[image_path]):
                        if item["sec_text"] == section_texts[img_sec_idx]:
                            # if the section text is already collected, append the doc_url
                            collected_data[image_path][idx]["doc_url"].append(doc_url)
                            break
                    else:
                        # if the section text is not collected, append a new entry
                        collected_data[image_path].append({
                            "img_url": img_url,
                            "sec_text": section_texts[img_sec_idx],
                            "doc_url": [doc_url]
                        })
                else:
                    collected_data[image_path] = [{
                        "img_url": img_url,
                        "sec_text": section_texts[img_sec_idx],
                        "doc_url": [doc_url]
                    }]
        
        if len(collected_data) == 0:
            raise ValueError("No data found in the knowledge base.")
        
        image_paths, image_urls, sec_texts, doc_urls = [], [], [], []
        for image_path, data_list in collected_data.items():
            for data in data_list:
                image_paths.append(image_path)
                image_urls.append(data["img_url"])
                sec_texts.append(data["sec_text"])
                doc_urls.append(data["doc_url"])

        self.faiss_index, self.faiss_index_meta = self._init_faiss_index()
        
        cur_idx = self.faiss_index.ntotal
        if resume:
            image_paths = image_paths[cur_idx:]
            image_urls = image_urls[cur_idx:]
            sec_texts = sec_texts[cur_idx:]
            doc_urls = doc_urls[cur_idx:]
        
        dataset = ImageTextDataset(image_paths, sec_texts, transform=None)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=False, collate_fn=image_text_collate_fn)

        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Updating faiss index"):
            batch_images, batch_texts = batch[0], batch[1]
            batch_size = len(batch_images)

            # update faiss index
            mm_feature = self.embed_model.embed(list(zip(batch_images, batch_texts, strict=True)), text_type="evidence")
            mm_feature = mm_feature.float().cpu().numpy()
            
            faiss.normalize_L2(mm_feature)
            self.faiss_index.add(mm_feature)

            # update meta data
            batch_doc_urls = doc_urls[:batch_size]
            batch_image_urls = image_urls[:batch_size]

            doc_urls = doc_urls[batch_size:]
            image_urls = image_urls[batch_size:]

            for i, (doc_url, image_url) in enumerate(zip(batch_doc_urls, batch_image_urls, strict=True)):
                self.faiss_index_meta["index_to_image"][str(cur_idx + i)] = image_url
                self.faiss_index_meta["index_to_doc"][str(cur_idx + i)] = doc_url
            
            cur_idx += batch_size

            if (batch_idx + 1) % save_index_per_batch == 0 or batch_idx == len(dataloader) - 1:
                # check faiss index and meta data
                assert self.faiss_index.ntotal == len(self.faiss_index_meta["index_to_doc"])
                self._save_faiss_index(self.faiss_index, self.faiss_index_meta, self.index_path)
    
    def build_text_index(self, batch_size=1, save_index_per_batch=100, resume=False):
        texts, doc_urls = [], []
        for doc_url, doc_data in self.knowledge_base.get_all_items():
            title = doc_data["title"]
            sections = doc_data["section_texts"]
            
            if len(sections) > 0:
                texts.append(f"{title}: {sections[0]}")
                doc_urls.append(doc_url)

        self.faiss_index, self.faiss_index_meta = self._init_faiss_index()
        cur_idx = self.faiss_index.ntotal
        
        if resume:
            texts = texts[cur_idx:]
            doc_urls = doc_urls[cur_idx:]

        # update faiss index
        dataloader = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        for batch_idx, batch_texts in tqdm(enumerate(dataloader), total=len(dataloader), desc="Updating faiss index"):
            batch_size = len(batch_texts)

            # update faiss index
            text_feature = self.embed_model.embed(batch_texts, modality="text")
            text_feature = text_feature.float().cpu().numpy()
            
            faiss.normalize_L2(text_feature)
            self.faiss_index.add(text_feature)

            # update meta data
            batch_doc_urls = doc_urls[:batch_size]
            doc_urls = doc_urls[batch_size:]

            for i, doc_url in enumerate(batch_doc_urls):
                self.faiss_index_meta["index_to_doc"][str(cur_idx + i)] = doc_url
            
            cur_idx += batch_size

            if (batch_idx + 1) % save_index_per_batch == 0 or batch_idx == len(dataloader) - 1:
                # check faiss index and meta data
                assert self.faiss_index.ntotal == len(self.faiss_index_meta["index_to_doc"])
                self._save_faiss_index(self.faiss_index, self.faiss_index_meta, self.index_path)

    def build_image_index(self, image_dir, batch_size=1, save_index_per_batch=100, resume=False):
        from torch.utils.data import DataLoader
        from dataset.image import ImagePathDataset
        from dataset.utils import image_path_collate_fn

        with open(os.path.join(os.path.dirname(image_dir), "kb_images_map.json"), "r") as f:
            kb_images_map = json.load(f)

        print("Collecting images from the knowledge base ...")
        images = {}
        for doc_url, doc_data in self.knowledge_base.get_all_items():
            for img_url in doc_data["image_urls"]:
                image_file = kb_images_map.get(img_url, "")
                
                if image_file == "":
                    continue
                image_path = os.path.join(image_dir, image_file)

                if image_path in images:
                    # one image may appear in multiple documents
                    images[image_path]["doc_url"].append(doc_url)
                else:
                    images[image_path] = {
                        "img_url": img_url,
                        "doc_url": [doc_url]
                    }
        
        if len(images) == 0:
            raise ValueError("No images found in the knowledge base.")
        
        image_paths, image_urls, doc_urls = [], [], []
        for path in images:
            image_paths.append(path)
            image_urls.append(images[path]["img_url"])
            doc_urls.append(images[path]["doc_url"])

        self.faiss_index, self.faiss_index_meta = self._init_faiss_index()
        
        cur_idx = self.faiss_index.ntotal
        if resume:
            image_paths = image_paths[cur_idx:]
            doc_urls = doc_urls[cur_idx:]
            image_urls = image_urls[cur_idx:]
        
        dataset = ImagePathDataset(image_paths, transform=None)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=False, collate_fn=image_path_collate_fn)

        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Updating faiss index"):
            batch_images = batch[0]
            batch_size = len(batch_images)

            # update faiss index
            image_feature = self.embed_model.embed(batch_images, modality="image")
            image_feature = image_feature.float().cpu().numpy()
            
            faiss.normalize_L2(image_feature)
            self.faiss_index.add(image_feature)

            # update meta data
            batch_doc_urls = doc_urls[:batch_size]
            batch_image_urls = image_urls[:batch_size]

            doc_urls = doc_urls[batch_size:]
            image_urls = image_urls[batch_size:]

            for i, (doc_url, image_url) in enumerate(zip(batch_doc_urls, batch_image_urls, strict=True)):
                self.faiss_index_meta["index_to_image"][str(cur_idx + i)] = image_url
                self.faiss_index_meta["index_to_doc"][str(cur_idx + i)] = doc_url
            
            cur_idx += batch_size

            if (batch_idx + 1) % save_index_per_batch == 0 or batch_idx == len(dataloader) - 1:
                # check faiss index and meta data
                assert self.faiss_index.ntotal == len(self.faiss_index_meta["index_to_doc"])
                self._save_faiss_index(self.faiss_index, self.faiss_index_meta, self.index_path)

    def _save_faiss_index(self, faiss_index, faiss_index_meta, index_path):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        write_index(faiss_index, index_path)
        with open(index_path.replace(".faiss", "_meta.json"), "w") as f:
            json.dump(faiss_index_meta, f)
        
        print("Faiss index updated, total index: {}".format(faiss_index.ntotal))

    def _get_valid_query(self, text=None, image=None, attribute=None):
        if self.src_modality == "image":
            if image is not None:
                return image
            else:
                raise ValueError("Image is required for image retrieval.")
        elif self.src_modality == "text":
            if text is not None:
                return f"{text} {attribute['caption']}"
            else:
                raise ValueError("Text is required for text retrieval.")
        elif self.src_modality == "mm":
            if text is not None and image is not None:
                return (image, text)
            else:
                raise ValueError("Text and image are required for multimodal retrieval.")

    def _retrieve_faiss(self, queries, top_k=10, **kwargs):
        query_feature = self.embed_model.embed(queries, **kwargs)
        
        query_feature = query_feature.float().cpu().numpy()
        faiss.normalize_L2(query_feature)
        distances, indices = self.faiss_index.search(query_feature, top_k)
        
        return distances, indices
    
    def _retrieve_image(self, queries, top_k=10):
        distances, indices = self._retrieve_faiss(queries, top_k, modality="image")

        results = []
        for query_idx in range(len(queries)):  # for each query
            top_k_entries = []
            for top_idx in range(top_k):  # for each image in the top k
                idx, similarity = indices[query_idx][top_idx], distances[query_idx][top_idx]
                doc_url = self.faiss_index_meta["index_to_doc"][str(idx)]
                image_url = self.faiss_index_meta["index_to_image"][str(idx)]
                
                top_k_entries.append({
                    "similarity": similarity, 
                    "image_url": image_url, 
                    "doc_url": doc_url
                })
            results.append(top_k_entries)
        return results
    
    def _retrieve_text(self, queries, top_k=10):
        distances, indices = self._retrieve_faiss(queries, top_k, modality="text")

        results = []
        for query_idx in range(len(queries)):  # for each query
            top_k_entries = []
            for top_idx in range(top_k):
                idx, similarity = indices[query_idx][top_idx], distances[query_idx][top_idx]
                doc_url = self.faiss_index_meta["index_to_doc"][str(idx)]
                
                top_k_entries.append({
                    "similarity": similarity, 
                    "doc_url": doc_url
                })
            results.append(top_k_entries)
        return results

    def _retrieve_mm(self, queries, top_k=10):
        distances, indices = self._retrieve_faiss(queries, top_k, text_type="question")

        results = []
        for query_idx in range(len(queries)):  # for each query
            top_k_entries = []
            for top_idx in range(top_k):  # for each image in the top k
                idx, similarity = indices[query_idx][top_idx], distances[query_idx][top_idx]
                doc_url = self.faiss_index_meta["index_to_doc"][str(idx)]
                image_url = self.faiss_index_meta["index_to_image"][str(idx)]
                
                top_k_entries.append({
                    "similarity": similarity, 
                    "image_url": image_url, 
                    "doc_url": doc_url
                })
            results.append(top_k_entries)
        return results
