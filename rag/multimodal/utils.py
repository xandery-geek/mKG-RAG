import os
import io
import re
import csv
import html
from typing import Any, Callable, Literal
from dataclasses import dataclass, field


STORAGES_IMPORT_PATH = {
    "NetworkXStorage": "rag.kg.networkx_impl",
    "JsonKVStorage": "rag.kg.json_kv_impl",
    "NanoVectorDBStorage": "rag.kg.nano_vector_db_impl",
    "JsonDocStatusStorage": "rag.kg.json_doc_status_impl",
    "Neo4JStorage": "rag.kg.neo4j_impl",
    "OracleKVStorage": "rag.kg.oracle_impl",
    "OracleGraphStorage": "rag.kg.oracle_impl",
    "OracleVectorDBStorage": "rag.kg.oracle_impl",
    "MilvusVectorDBStorage": "rag.kg.milvus_impl",
    "MongoKVStorage": "rag.kg.mongo_impl",
    "MongoDocStatusStorage": "rag.kg.mongo_impl",
    "MongoGraphStorage": "rag.kg.mongo_impl",
    "MongoVectorDBStorage": "rag.kg.mongo_impl",
    "RedisKVStorage": "rag.kg.redis_impl",
    "ChromaVectorDBStorage": "rag.kg.chroma_impl",
    "TiDBKVStorage": "rag.kg.tidb_impl",
    "TiDBVectorDBStorage": "rag.kg.tidb_impl",
    "TiDBGraphStorage": "rag.kg.tidb_impl",
    "PGKVStorage": "rag.kg.postgres_impl",
    "PGVectorStorage": "rag.kg.postgres_impl",
    "AGEStorage": "rag.kg.age_impl",
    "PGGraphStorage": "rag.kg.postgres_impl",
    "GremlinStorage": "rag.kg.gremlin_impl",
    "PGDocStatusStorage": "rag.kg.postgres_impl",
    "QdrantVectorDBStorage": "rag.kg.qdrant_impl",
    # Custom storage implementations
    "FaissVectorDBStorage": "rag.multimodal.faiss",
}


class NameSpace:
    KV_STORE_FULL_DOCS = "full_docs"
    KV_STORE_TEXT_CHUNKS = "text_chunks"
    KV_STORE_LLM_RESPONSE_CACHE = "llm_response_cache"

    VECTOR_STORE_ENTITIES = "entities"
    VECTOR_STORE_RELATIONSHIPS = "relationships"
    VECTOR_STORE_CHUNKS = "chunks"
    VECTOR_STORE_IMAGES = "images"
    # VECTOR_STORE_MULTIMODAL = "multimodal"
    VECTOR_STORE_MULTIMODAL = "mmv2" # multimodal v2

    GRAPH_STORE_CHUNK_ENTITY_RELATION = "chunk_entity_relation"

    DOC_STATUS = "doc_status"



STORAGE_META_FIELDS = {
    NameSpace.KV_STORE_FULL_DOCS: {},
    NameSpace.KV_STORE_TEXT_CHUNKS: {},
    NameSpace.KV_STORE_LLM_RESPONSE_CACHE: {},
    NameSpace.VECTOR_STORE_ENTITIES: {"entity_name", "graph_id"},
    NameSpace.VECTOR_STORE_RELATIONSHIPS: {"src_id", "tgt_id", "graph_id"},
    NameSpace.VECTOR_STORE_CHUNKS: {"graph_id"},
    NameSpace.VECTOR_STORE_IMAGES: {"image_path", "roi", "entity_name", "src_id", "tgt_id", "graph_id", "weight"},
    NameSpace.VECTOR_STORE_MULTIMODAL: {"image_path", "roi", "entity_name", "src_id", "tgt_id", "graph_id", "weight"},
    NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION: {},
    NameSpace.DOC_STATUS: {},
}


@dataclass
class QueryParam:
    """Configuration parameters for query execution in mKG-RAG."""

    strategy: Literal["naive", "text", "image", "text-image", "multimodal"] = "image"
    """Specifies the retrieval strategy:
    - "naive": Naive chunk-based retrieval.
    - "text": Text-based retrieval.
    - "image": Image-based retrieval.
    - "multimodal": Multimodal retrieval
    """
    mode: Literal["local", "global", "hybrid"] = "hybrid"
    """Specifies the retrieval mode for text-based, image-based, or text-image queries:
    - "local": Focuses on context-dependent information.
    - "global": Utilizes global knowledge.
    - "hybrid": Combines local and global retrieval methods.
    """

    top_k: int = 10
    """Number of top items to retrieve. Represents entities in 'local' mode and relationships in 'global' mode."""

    traverse_hop: int = 1
    """Number of hops to traverse in the knowledge graph for local retrieval."""

    filter_strategy: Literal["none", "quesiton"] = "none"
    """Specifies the filtering strategy:
    - "none": No filtering.
    - "question": Filters based on the relevance of the question.
    """

    filter_param: dict[str, Any] = field(default_factory=dict)
    """Filter parameters for the filtering strategy."""

    max_token_for_text_unit: int = int(os.getenv("MAX_TOKEN_TEXT_CHUNK", "4000"))
    """Maximum number of tokens allowed for each retrieved text chunk."""

    max_token_for_global_context: int = int(
        os.getenv("MAX_TOKEN_RELATION_DESC", "4000")
    )
    """Maximum number of tokens allocated for relationship descriptions in global retrieval."""

    max_token_for_local_context: int = int(os.getenv("MAX_TOKEN_ENTITY_DESC", "4000"))
    """Maximum number of tokens allocated for entity descriptions in local retrieval."""

    hl_keywords: list[str] = field(default_factory=list)
    """List of high-level keywords to prioritize in retrieval."""

    ll_keywords: list[str] = field(default_factory=list)
    """List of low-level keywords to refine retrieval focus."""

    conversation_history: list[dict[str, str]] = field(default_factory=list)
    """Stores past conversation history to maintain context.
    Format: [{"role": "user/assistant", "content": "message"}].
    """

    history_turns: int = 3
    """Number of complete conversation turns (user-assistant pairs) to consider in the response context."""

    retrieve_from_dynamic_graph: bool = False
    """If True, allows querying from multiple knowledge graphs."""

    return_image: bool = False
    """If True, add the image information to the context."""

    context_mode: Literal["graph", "chunk", "hybrid"] = "hybrid"
    """Specifies the context mode for the response:
    - "graph": Uses the entire knowledge graph.
    - "chunk": Uses the chunk-based context.
    - "hybrid": Combines graph and chunk-based contexts.
    """

    
def make_namespace(prefix: str, base_namespace: str):
    return prefix + base_namespace


def lazy_external_import(module_name: str, class_name: str) -> Callable[..., Any]:
    """Lazily import a class from an external module based on the package of the caller."""
    # Get the caller's module and package
    import inspect

    caller_frame = inspect.currentframe().f_back
    module = inspect.getmodule(caller_frame)
    package = module.__package__ if module else None

    def import_class(*args: Any, **kwargs: Any):
        import importlib

        module = importlib.import_module(module_name, package=package)
        cls = getattr(module, class_name)
        return cls(*args, **kwargs)

    return import_class


def split_string_by_multi_markers(content: str, markers: list[str], ignorecase=False) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    # Add flags=re.IGNORECASE to make the split case-insensitive
    flags = re.IGNORECASE if ignorecase else 0
    results = re.split("|".join(re.escape(marker) for marker in markers), content, flags=flags)
    return [r.strip() for r in results if r.strip()]


def list_of_list_to_csv(data: list[list[str]]) -> str:
    output = io.StringIO()
    writer = csv.writer(
        output,
        quoting=csv.QUOTE_ALL,  # Quote all fields
        escapechar="\\",  # Use backslash as escape character
        quotechar='"',  # Use double quotes
        lineterminator="\n",  # Explicit line terminator
    )
    writer.writerows(data)
    return output.getvalue()


def csv_string_to_list(csv_string: str) -> list[list[str]]:
    # Clean the string by removing NUL characters
    cleaned_string = csv_string.replace("\0", "")

    output = io.StringIO(cleaned_string)
    reader = csv.reader(
        output,
        quoting=csv.QUOTE_ALL,  # Match the writer configuration
        escapechar="\\",  # Use backslash as escape character
        quotechar='"',  # Use double quotes
    )

    try:
        return [row for row in reader]
    except csv.Error as e:
        raise ValueError(f"Failed to parse CSV string: {str(e)}")
    finally:
        output.close()

# Refer the utils functions of the official GraphRAG implementation: https://github.com/microsoft/graphrag
def clean_str(text: Any, chars_to_strip=[' ', '"', '\\', '(', ')']) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""

    # If we get non-string input, just give it back
    if not isinstance(text, str):
        return text
    
    # Remove control characters
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    # Strip unwanted characters
    text = text.strip(''.join(chars_to_strip))
    return html.unescape(text).strip()
