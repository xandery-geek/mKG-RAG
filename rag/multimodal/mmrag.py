from __future__ import annotations

import os
import shutil
import logging
import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Any, Callable, cast, final

from rag.kg import (
    STORAGE_ENV_REQUIREMENTS,
    verify_storage_implementation,
)

from rag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
    StorageNameSpace,
    StoragesStatus,
)

from rag.utils import (
    logger,
    always_get_an_event_loop,
    compute_mdhash_id,
    convert_response_to_json,
    limit_async_func_call,
    set_logger,
    EmbeddingFunc,
)

from rag.types import KnowledgeGraph

from rag.multimodal.operate import chunking_by_character
from rag.multimodal.build import (
    extract_entities, 
    build_unified_images_vdb, 
    build_unified_vector_db,
    build_separate_images_vdb,
    build_dynamic_graph
)
from rag.multimodal.utils import (
    STORAGES_IMPORT_PATH,
    STORAGE_META_FIELDS,
    make_namespace, 
    lazy_external_import, 
    NameSpace,
    QueryParam
)
from rag.multimodal.query import mm_kg_query, naive_query


@final
@dataclass
class MultimodalRAG:
    """MultimodalRAG class for handling multimodal RAG operations."""

    graph_id: str | None = None
    
    # Directory
    # ---

    working_dir: str = field(
        default=f"./working_dir/mmrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    """Directory where graph, cache and temporary files are stored."""

    vector_storage_dir: str | None = None
    """Directory where vector storage is stored. Used when unified_vector_storage is True."""

    image_dir: str | None = None
    """Directory where images are stored."""

    images_map_file: str | None = None
    """Path to the file containing the mapping of image URLs to image names."""

    scene_graph_dir: str | None = None
    """Directory where scene graphs are stored."""

    unified_vector_storage: bool = field(default=False)
    """If True, uses a single vector storage for all entities and relationships."""

    use_mm_embedding: bool = field(default=False)
    """If True, uses a multimodal embedding function for images."""

    # Logging
    # ---

    log_level: int = field(default=logger.level)
    """Logging level for the system (e.g., 'DEBUG', 'INFO', 'WARNING')."""

    log_file_path: str = field(default=os.path.join(os.getcwd(), "mmrag.log"))
    """Log file path."""

    # Storage
    # ---

    kv_storage: str = field(default="JsonKVStorage")
    """Storage backend for key-value data."""

    vector_storage: str = field(default="NanoVectorDBStorage")
    """Storage backend for vector embeddings."""

    graph_storage: str = field(default="NetworkXStorage")
    """Storage backend for knowledge graphs."""

    doc_status_storage: str = field(default="JsonDocStatusStorage")
    """Storage type for tracking document processing statuses."""

    # Entity extraction
    # ---

    entity_extract_max_gleaning: int = field(default=1)
    """Maximum number of entity extraction attempts for ambiguous content."""

    entity_summary_to_max_tokens: int = field(
        default=int(os.getenv("MAX_TOKEN_SUMMARY", 500))
    )

    # Text chunking
    # ---
    tiktoken_model_name: str = field(default="gpt-4o-mini")
    """Model name used for tokenization when chunking text."""

    chunking_func: Callable[
        [str, str, str],
        list[dict[str, Any]],
    ] = field(default_factory=lambda: chunking_by_character)
    """
    Custom chunking function for splitting text into chunks before processing.

    The function should take the following parameters:

        - `content`: The text to be split into chunks.
        - `split_by_character`: The character to split the text on.
        - `tiktoken_model_name`: The name of the tiktoken model to use for tokenization.

    The function should return a list of dictionaries, where each dictionary contains the following keys:
        - `tokens`: The number of tokens in the chunk.
        - `content`: The text content of the chunk.

    Defaults to `chunking_by_character` if not specified.
    """

    # Node embedding
    # ---

    node_embedding_algorithm: str = field(default="node2vec")
    """Algorithm used for node embedding in knowledge graphs."""

    node2vec_params: dict[str, int] = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )
    """Configuration for the node2vec embedding algorithm:
    - dimensions: Number of dimensions for embeddings.
    - num_walks: Number of random walks per node.
    - walk_length: Number of steps per random walk.
    - window_size: Context window size for training.
    - iterations: Number of iterations for training.
    - random_seed: Seed value for reproducibility.
    """

    # Embedding
    # ---

    embedding_func: EmbeddingFunc | None = field(default=None)
    """Function for computing text embeddings. Must be set before use."""

    embedding_batch_num: int = field(default=32)
    """Batch size for embedding computations."""

    embedding_func_max_async: int = field(default=16)
    """Maximum number of concurrent embedding function calls."""

    embedding_cache_config: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "similarity_threshold": 0.95,
            "use_llm_check": False,
        }
    )
    """Configuration for embedding cache.
    - enabled: If True, enables caching to avoid redundant computations.
    - similarity_threshold: Minimum similarity score to use cached embeddings.
    - use_llm_check: If True, validates cached embeddings using an LLM.
    """

    vision_embedding_func: EmbeddingFunc | None = field(default=None)
    """Function for computing vision embeddings. Must be set before use."""

    mm_embedding_func: EmbeddingFunc | None = field(default=None)
    """Function for computing multimodal embeddings. Must be set before use."""

    # LLM Configuration
    # ---

    llm_model_func: Callable[..., object] | None = field(default=None)
    """Function for interacting with the large language model (LLM). Must be set before use."""

    llm_model_name: str = field(default="gpt-4o-mini")
    """Name of the LLM model used for generating responses."""

    llm_model_max_token_size: int = field(default=int(os.getenv("MAX_TOKENS", 32768)))
    """Maximum number of tokens allowed per LLM response."""

    llm_model_max_async: int = field(default=int(os.getenv("MAX_ASYNC", 16)))
    """Maximum number of concurrent LLM calls."""

    llm_model_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments passed to the LLM model function."""

    # Storage
    # ---

    vector_db_storage_cls_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional parameters for vector database storage."""

    namespace_prefix: str = field(default="")
    """Prefix for namespacing stored data across different environments."""

    enable_llm_cache: bool = field(default=True)
    """Enables caching for LLM responses to avoid redundant computations."""

    enable_llm_cache_for_entity_extract: bool = field(default=True)
    """If True, enables caching for entity extraction steps to reduce LLM costs."""

    # Extensions
    # ---

    max_parallel_insert: int = field(default=int(os.getenv("MAX_PARALLEL_INSERT", 20)))
    """Maximum number of parallel insert operations."""

    addon_params: dict[str, Any] = field(default_factory=dict)

    # Storages Management
    # ---

    auto_manage_storages_states: bool = field(default=True)
    """If True, mmrag will automatically calls initialize_storages and finalize_storages at the appropriate times."""

    # Storages Management
    # ---

    convert_response_to_json_func: Callable[[str], dict[str, Any]] = field(
        default_factory=lambda: convert_response_to_json
    )
    """
    Custom function for converting LLM responses to JSON format.

    The default function is :func:`.utils.convert_response_to_json`.
    """

    cosine_better_than_threshold: float = field(
        default=float(os.getenv("COSINE_THRESHOLD", 0.2))
    )

    _storages_status: StoragesStatus = field(default=StoragesStatus.NOT_CREATED)

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        set_logger(self.log_file_path)
        logger.setLevel(self.log_level)
        logger.info(f"Logger initialized for working directory: {self.working_dir}")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        if self.use_mm_embedding:
            if self.mm_embedding_func is None:
                raise ValueError(
                    "Multimodal embedding function is required when use_mm_embedding is True."
                )
            else:
                logger.info("Using multimodal embedding function for images.")

        # Set graph_id as the basename of the working directory
        self.graph_id = os.path.basename(self.working_dir)
        
        # Verify storage implementation compatibility and environment variables
        storage_configs = [
            ("KV_STORAGE", self.kv_storage),
            ("VECTOR_STORAGE", self.vector_storage),
            ("GRAPH_STORAGE", self.graph_storage),
            ("DOC_STATUS_STORAGE", self.doc_status_storage),
        ]

        for storage_type, storage_name in storage_configs:
            # Verify storage implementation compatibility
            verify_storage_implementation(storage_type, storage_name)
            # Check environment variables
            # self.check_storage_env_vars(storage_name)

        # Ensure vector_db_storage_cls_kwargs has required fields
        self.vector_db_storage_cls_kwargs = {
            "cosine_better_than_threshold": self.cosine_better_than_threshold,
            **self.vector_db_storage_cls_kwargs,
        }

        # Show config
        global_config = asdict(self)
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in global_config.items()])
        logger.debug(f"MultimodalRAG init with param:\n  {_print_config}\n")

        # Init Embedding Function
        if self.embedding_func:
            self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(  # type: ignore
                self.embedding_func
            )

        if self.vision_embedding_func:
            self.vision_embedding_func = limit_async_func_call(self.embedding_func_max_async)(  # type: ignore
                self.vision_embedding_func
            )
        
        if self.mm_embedding_func:
            self.mm_embedding_func = limit_async_func_call(self.embedding_func_max_async)(  # type: ignore
                self.mm_embedding_func
            )

        if self.llm_model_func:
            # Init LLM Model Function
            self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
                partial(
                    self.llm_model_func,  # type: ignore
                    llm_model_name=self.llm_model_name,
                    **self.llm_model_kwargs,
                )
            )

        # Initialize storages classes
        self._init_storage_classes(global_config, reinit=False)

    def __del__(self):
        # Finalize storages
        if self.auto_manage_storages_states:
            loop = always_get_an_event_loop()
            loop.run_until_complete(self._finalize_storages())
    
    def _init_storage_classes(self, global_config, reinit=False, reinit_vdb=False):
        if reinit:
            # Finalize storages before reinitializing
            if self.auto_manage_storages_states:
                loop = always_get_an_event_loop()
                loop.run_until_complete(self._finalize_storages())

        # Initialize kv and graph storage classes
        self.key_string_value_json_storage_cls: type[BaseKVStorage] = self._get_storage_class(self.kv_storage)
        self.key_string_value_json_storage_cls = partial(self.key_string_value_json_storage_cls, global_config=global_config)

        self.graph_storage_cls: type[BaseGraphStorage] = self._get_storage_class(self.graph_storage)
        self.graph_storage_cls = partial(self.graph_storage_cls, global_config=global_config)

        self.llm_response_cache: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
            ),
            embedding_func=self.embedding_func,
        )
        self.full_docs: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.KV_STORE_FULL_DOCS
            ),
            embedding_func=self.embedding_func,
        )
        self.text_chunks: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.KV_STORE_TEXT_CHUNKS
            ),
            embedding_func=self.embedding_func,
        )
        self.chunk_entity_relation_graph: BaseGraphStorage = self.graph_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION
            ),
            embedding_func=self.embedding_func,
        )

        # Initialize document status storage
        self.doc_status_storage_cls = self._get_storage_class(self.doc_status_storage)

        self.doc_status: DocStatusStorage = self.doc_status_storage_cls(
            namespace=make_namespace(self.namespace_prefix, NameSpace.DOC_STATUS),
            global_config=global_config,
            embedding_func=None,
        )

        # Initialize vector storages for the first time or when reinit_vdb is True
        if not reinit or reinit_vdb:
            # TODO: support other vector storage types
            assert self.vector_storage == "FaissVectorDBStorage"
            self.vector_db_storage_cls: type[BaseVectorStorage] = self._get_storage_class(self.vector_storage)
            self.vector_db_storage_cls = partial(self.vector_db_storage_cls, global_config=global_config)

            self.entities_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
                namespace=make_namespace(
                    self.namespace_prefix, NameSpace.VECTOR_STORE_ENTITIES
                ),
                embedding_func=self.embedding_func,
                meta_fields=STORAGE_META_FIELDS[NameSpace.VECTOR_STORE_ENTITIES],
            )
            self.relationships_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
                namespace=make_namespace(
                    self.namespace_prefix, NameSpace.VECTOR_STORE_RELATIONSHIPS
                ),
                embedding_func=self.embedding_func,
                meta_fields=STORAGE_META_FIELDS[NameSpace.VECTOR_STORE_RELATIONSHIPS],
            )
            self.chunks_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
                namespace=make_namespace(
                    self.namespace_prefix, NameSpace.VECTOR_STORE_CHUNKS
                ),
                embedding_func=self.embedding_func,
                meta_fields=STORAGE_META_FIELDS[NameSpace.VECTOR_STORE_CHUNKS],
            )
            self.images_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
                namespace=make_namespace(
                    self.namespace_prefix, NameSpace.VECTOR_STORE_IMAGES
                ),
                embedding_func=self.vision_embedding_func,
                meta_fields=STORAGE_META_FIELDS[NameSpace.VECTOR_STORE_IMAGES],
            )
            self.multimodal_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
                namespace=make_namespace(
                    self.namespace_prefix, NameSpace.VECTOR_STORE_MULTIMODAL
                ),
                embedding_func=self.mm_embedding_func,
                meta_fields=STORAGE_META_FIELDS[NameSpace.VECTOR_STORE_MULTIMODAL],
            )

        self._storages_status = StoragesStatus.CREATED

        # Initialize storages
        if self.auto_manage_storages_states:
            loop = always_get_an_event_loop()
            loop.run_until_complete(self._initialize_storages())

    def reinit_storages(self, working_dir, log_file_path, reinit_vdb=False):
        # update working_dir and log_file_path
        self.working_dir = working_dir
        self.log_file_path = log_file_path
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        
        # Reinitialize logger
        logger = logging.getLogger("mkg-rag")
        logger.removeHandler(logger.handlers[0])
        
        set_logger(self.log_file_path)
        logger.setLevel(self.log_level)
        logger.info(f"Logger initialized for working directory: {self.working_dir}")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.graph_id = os.path.basename(self.working_dir)

        # Get global config
        global_config = self.llm_response_cache.global_config
        global_config["working_dir"] = self.working_dir
        global_config["log_file_path"] = self.log_file_path
        global_config["graph_id"] = self.graph_id

        # When using separate vector storage, reinitialize vector storages for each document
        reinit_vdb = reinit_vdb or not self.unified_vector_storage
        self._init_storage_classes(global_config, reinit=True, reinit_vdb=reinit_vdb)

    def _init_hashing_kv(self) -> BaseKVStorage:
        """
        Initialize hashing_kv storage.
        If the llm_response_cache is already initialized, return it. Otherwise, initialize a new instance of the storage.
        """
        if self.llm_response_cache and hasattr(self.llm_response_cache, "global_config"):
            return self.llm_response_cache
        else: 
            return self.key_string_value_json_storage_cls(
                namespace=make_namespace(
                    self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                ),
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            )

    def _get_storage_class(self, storage_name: str) -> Callable[..., Any]:
        import_path = STORAGES_IMPORT_PATH[storage_name]
        storage_class = lazy_external_import(import_path, storage_name)
        return storage_class

    async def _initialize_storages(self):
        """Asynchronously initialize the storages"""
        if self._storages_status == StoragesStatus.CREATED:
            tasks = []

            for storage in (
                self.full_docs,
                self.text_chunks,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.images_vdb,
                self.multimodal_vdb,
                self.chunk_entity_relation_graph,
                self.llm_response_cache,
                self.doc_status,
            ):
                if storage:
                    tasks.append(storage.initialize())

            await asyncio.gather(*tasks)

            self._storages_status = StoragesStatus.INITIALIZED
            logger.debug("Initialized Storages")

    async def _finalize_storages(self):
        """Asynchronously finalize the storages"""
        if self._storages_status == StoragesStatus.INITIALIZED:
            tasks = []

            for storage in (
                self.full_docs,
                self.text_chunks,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.images_vdb,
                self.multimodal_vdb,
                self.chunk_entity_relation_graph,
                self.llm_response_cache,
                self.doc_status,
            ):
                if storage:
                    tasks.append(storage.finalize())

            await asyncio.gather(*tasks)

            self._storages_status = StoragesStatus.FINALIZED
            logger.debug("Finalized Storages")

    async def get_graph_labels(self):
        text = await self.chunk_entity_relation_graph.get_all_labels()
        return text

    async def get_knowledge_graph(
        self, nodel_label: str, max_depth: int
    ) -> KnowledgeGraph:
        return await self.chunk_entity_relation_graph.get_knowledge_graph(
            node_label=nodel_label, max_depth=max_depth
        )

    def insert(
        self,
        input: str | list[str],
        split_by_character: str,
    ) -> None:
        """Sync Insert documents with checkpoint support

        Args:
            input: Single document string or list of document strings
            split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
            split_by_character_only: if split_by_character_only is True, split the string by character only, when
            split_by_character is None, this parameter is ignored.
        """

        loop = always_get_an_event_loop()
        loop.run_until_complete(
            self.ainsert(input, split_by_character)
        )

    async def ainsert(
        self,
        input: str | list[str],
        split_by_character: str,
    ) -> None:
        """Async Insert documents with checkpoint support

        Args:
            input: Single document string or list of document strings
            split_by_character: Split the string by character
        """
        await self.apipeline_enqueue_documents(input)
        await self.apipeline_process_enqueue_documents(split_by_character)

    async def apipeline_enqueue_documents(self, input: str | list[str]) -> None:
        """
        Pipeline for Processing Documents

        1. Remove duplicate contents from the list
        2. Generate document IDs and initial status
        3. Filter out already processed documents
        4. Enqueue document in status
        """
        if isinstance(input, str):
            input = [input]

        # 1. Remove duplicate contents from the list
        unique_contents = list(set(doc.strip() for doc in input))

        # 2. Generate document IDs and initial status
        new_docs: dict[str, Any] = {
            compute_mdhash_id(content, prefix="doc-"): {
                "content": content,
                "content_summary": self._get_content_summary(content),
                "content_length": len(content),
                "status": DocStatus.PENDING,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            for content in unique_contents
        }

        # 3. Filter out already processed documents
        # Get docs ids
        all_new_doc_ids = set(new_docs.keys())
        # Exclude IDs of documents that are already in progress
        unique_new_doc_ids = await self.doc_status.filter_keys(all_new_doc_ids)
        # Filter new_docs to only include documents with unique IDs
        new_docs = {doc_id: new_docs[doc_id] for doc_id in unique_new_doc_ids}

        if not new_docs:
            logger.info("No new unique documents were found.")
            return

        # 4. Store status document
        await self.doc_status.upsert(new_docs)
        logger.info(f"Stored {len(new_docs)} new unique documents")

    async def apipeline_process_enqueue_documents(
        self,
        split_by_character: str,
    ) -> None:
        """
        Process pending documents by splitting them into chunks, processing
        each chunk for entity and relation extraction, and updating the
        document status.

        1. Get all pending, failed, and abnormally terminated processing documents.
        2. Split document content into chunks
        3. Process each chunk for entity and relation extraction
        4. Update the document status
        """
        # 1. Get all pending, failed, and abnormally terminated processing documents.
        # Run the asynchronous status retrievals in parallel using asyncio.gather
        processing_docs, failed_docs, pending_docs = await asyncio.gather(
            self.doc_status.get_docs_by_status(DocStatus.PROCESSING),
            self.doc_status.get_docs_by_status(DocStatus.FAILED),
            self.doc_status.get_docs_by_status(DocStatus.PENDING),
        )

        to_process_docs: dict[str, DocProcessingStatus] = {}
        to_process_docs.update(processing_docs)
        to_process_docs.update(failed_docs)
        to_process_docs.update(pending_docs)

        if not to_process_docs:
            logger.info("All documents have been processed or are duplicates")
            return

        # 2. split docs into chunks, insert chunks, update doc status
        docs_batches = [
            list(to_process_docs.items())[i : i + self.max_parallel_insert]
            for i in range(0, len(to_process_docs), self.max_parallel_insert)
        ]

        logger.info(f"Number of batches to process: {len(docs_batches)}.")

        batches: list[Any] = []
        # 3. iterate over batches
        for batch_idx, docs_batch in enumerate(docs_batches):

            async def batch(
                batch_idx: int,
                docs_batch: list[tuple[str, DocProcessingStatus]],
                size_batch: int,
            ) -> None:
                logger.info(f"Start processing batch {batch_idx + 1} of {size_batch}.")
                # 4. iterate over batch
                for doc_id_processing_status in docs_batch:
                    doc_id, status_doc = doc_id_processing_status
                    # Update status in processing
                    doc_status_id = compute_mdhash_id(status_doc.content, prefix="doc-")
                    # Generate chunks from document
                    chunks: dict[str, Any] = {
                        compute_mdhash_id(dp["content"], prefix="chunk-"): {
                            **dp,
                            "full_doc_id": doc_id,
                        }
                        for dp in self.chunking_func(
                            status_doc.content,
                            split_by_character,
                            self.tiktoken_model_name,
                        )
                    }
                    # Process document (text chunks and full docs) in parallel
                    tasks = [
                        self.doc_status.upsert(
                            {
                                doc_status_id: {
                                    "status": DocStatus.PROCESSING,
                                    "updated_at": datetime.now().isoformat(),
                                    "content": status_doc.content,
                                    "content_summary": status_doc.content_summary,
                                    "content_length": status_doc.content_length,
                                    "created_at": status_doc.created_at,
                                }
                            }
                        ),
                        self.chunks_vdb.upsert(chunks),
                        self._process_entity_relation_graph(chunks),
                        self.full_docs.upsert(
                            {doc_id: {"content": status_doc.content}}
                        ),
                        self.text_chunks.upsert(chunks),
                    ]
                    try:
                        await asyncio.gather(*tasks)
                        await self.doc_status.upsert(
                            {
                                doc_status_id: {
                                    "status": DocStatus.PROCESSED,
                                    "chunks_count": len(chunks),
                                    "content": status_doc.content,
                                    "content_summary": status_doc.content_summary,
                                    "content_length": status_doc.content_length,
                                    "created_at": status_doc.created_at,
                                    "updated_at": datetime.now().isoformat(),
                                }
                            }
                        )
                    except Exception as e:
                        logger.error(f"Failed to process document {doc_id}: {str(e)}")
                        await self.doc_status.upsert(
                            {
                                doc_status_id: {
                                    "status": DocStatus.FAILED,
                                    "error": str(e),
                                    "content": status_doc.content,
                                    "content_summary": status_doc.content_summary,
                                    "content_length": status_doc.content_length,
                                    "created_at": status_doc.created_at,
                                    "updated_at": datetime.now().isoformat(),
                                }
                            }
                        )
                        continue
                logger.info(f"Completed batch {batch_idx + 1} of {len(docs_batches)}.")

            batches.append(batch(batch_idx, docs_batch, len(docs_batches)))

        await asyncio.gather(*batches)
        await self._insert_done()

    async def _process_entity_relation_graph(self, chunk: dict[str, Any]) -> None:
        try:
            await extract_entities(
                chunk,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entities_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                llm_response_cache=self.llm_response_cache,
                global_config=asdict(self),
            )
        except Exception as e:
            logger.error("Failed to extract entities and relationships")
            raise e

    async def _insert_done(self) -> None:
        tasks = [
            cast(StorageNameSpace, storage_inst).index_done_callback()
            for storage_inst in [  # type: ignore
                self.full_docs,
                self.text_chunks,
                self.llm_response_cache,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.images_vdb,
                self.multimodal_vdb,
                self.chunk_entity_relation_graph,
            ]
            if storage_inst is not None
        ]
        await asyncio.gather(*tasks)
        logger.info("All Insert done")

    def post_process_vdb(self, graph_ids: list[str], working_dir: str, log_file_path: str, unified_storage: bool) -> None:
        """
        Build image index for the given graph IDs.
        """

        loop = always_get_an_event_loop()
        if unified_storage:
            # create unified vector storages in the working directory
            self.reinit_storages(working_dir, log_file_path, reinit_vdb=True)

            if self.unified_vector_storage is False:
                # If unified_vector_storage is not used when building the knowledge graph,
                # all vector storages need to be merged during post processing
                loop.run_until_complete(
                    self.amerge_vector_db(graph_ids)
                )
        # Build image vector storage
        loop.run_until_complete(
            self.abuild_images_vdb(graph_ids, unified_storage)
        )
        
    async def amerge_vector_db(self, graph_ids: list[str]) -> None:
        await build_unified_vector_db(
            graph_ids,
            self.chunks_vdb,
            self.entities_vdb,
            self.relationships_vdb,
            global_config=asdict(self),
        )
        tasks = [
            cast(StorageNameSpace, storage_inst).index_done_callback()
            for storage_inst in [  # type: ignore
                self.chunks_vdb,
                self.entities_vdb,
                self.relationships_vdb,
            ]
        ]
        await asyncio.gather(*tasks)
        logger.info("Vector databases merged")

    async def abuild_images_vdb(self, graph_ids: list[str], unified_storage: bool) -> None:

        if unified_storage:
            await build_unified_images_vdb(
                graph_ids, 
                self.images_vdb, 
                self.multimodal_vdb, 
                global_config=asdict(self)
            )
            await asyncio.gather(
                self.images_vdb.index_done_callback(),
                self.multimodal_vdb.index_done_callback(),
            )
            logger.info("Unified image vector database built")
        else:
            await build_separate_images_vdb(
                graph_ids,
                global_config=asdict(self)
            )
            logger.info("Separate image vector databases built")

    def mm_query(self, 
                 query: dict[str, Any],
                 param: QueryParam = QueryParam(), 
                 graph_ids: list[str] = []) -> str:
        
        loop = always_get_an_event_loop()
        
        if not param.retrieve_from_dynamic_graph:
            return loop.run_until_complete(self.amm_query(query, param))

        # When querying from multiple graphs, build a dynamic graph by merging the specified graphs            
        if len(graph_ids) == 0:
            raise ValueError("Graph IDs are required when retrieve_from_dynamic_graph is True")

        working_dir = f"./working_dir/mmrag_cache/rag_{os.getpid()}_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
        log_file_path = os.path.join(working_dir, "mmrag.log")

        try:
            os.makedirs(working_dir, exist_ok=True)

            self.reinit_storages(working_dir, log_file_path, reinit_vdb=True)
            loop.run_until_complete(build_dynamic_graph(
                graph_ids,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                text_chunks=self.text_chunks,
                chunks_vdb=self.chunks_vdb,
                entities_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                images_vdb=self.images_vdb,
                multimodal_vdb=self.multimodal_vdb,
                global_config=asdict(self),
            ))
            response = loop.run_until_complete(self.amm_query(query, param))
        finally:
            loop.run_until_complete(self._finalize_storages())
            # Remove the temporary dynamic graph
            if os.path.exists(working_dir):
                shutil.rmtree(working_dir, ignore_errors=True)
            
        return response
    
    async def amm_query(self, query: dict[str, Any], param: QueryParam = QueryParam()) -> str:
        
        if param.strategy == "naive":
            response = await naive_query(query, self.chunks_vdb, self.text_chunks, param)
        else:
            response = await mm_kg_query(
                query,
                self.chunk_entity_relation_graph,
                self.text_chunks,
                self.entities_vdb,
                self.relationships_vdb,
                self.images_vdb,
                self.multimodal_vdb,
                param,
                asdict(self),
            )
        await self._query_done()
        return response
    
    async def _query_done(self):
        await self.llm_response_cache.index_done_callback()

    def is_processed(self) -> DocProcessingStatus:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ais_processed())
    
    async def ais_processed(self) -> DocProcessingStatus:
        """Get current document processing status"""
        doc_status = await self.doc_status.get_docs_by_status(DocStatus.PROCESSED)
        return len(doc_status) > 0

    def _get_number_of_nodes(self) -> int:
        """Get the number of nodes in the knowledge graph."""
        return self.chunk_entity_relation_graph._graph.number_of_nodes()
    
    def _get_number_of_edges(self) -> int:
        """Get the number of edges in the knowledge graph."""
        return self.chunk_entity_relation_graph._graph.number_of_edges()

    def _get_content_summary(self, content: str, max_length: int = 100) -> str:
        """Get summary of document content

        Args:
            content: Original document content
            max_length: Maximum length of summary

        Returns:
            Truncated content with ellipsis if needed
        """
        content = content.strip()
        if len(content) <= max_length:
            return content
        return content[:max_length] + "..."

    async def get_processing_status(self) -> dict[str, int]:
        """Get current document processing status counts

        Returns:
            Dict with counts for each status
        """
        return await self.doc_status.get_status_counts()

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        """Get documents by status

        Returns:
            Dict with document id is keys and document status is values
        """
        return await self.doc_status.get_docs_by_status(status)

    def check_storage_env_vars(self, storage_name: str) -> None:
        """Check if all required environment variables for storage implementation exist

        Args:
            storage_name: Storage implementation name

        Raises:
            ValueError: If required environment variables are missing
        """
        required_vars = STORAGE_ENV_REQUIREMENTS.get(storage_name, [])
        missing_vars = [var for var in required_vars if var not in os.environ]

        if missing_vars:
            raise ValueError(
                f"Storage implementation '{storage_name}' requires the following "
                f"environment variables: {', '.join(missing_vars)}"
            )
