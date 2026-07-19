"""Обёртка над ChromaDB: хранение и поиск фрагментов базы знаний."""
import chromadb
from chromadb.utils import embedding_functions

import config

_client = None
_collection = None


def _get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL
    )


def get_collection():
    """Возвращает (создавая при необходимости) коллекцию Chroma."""
    global _client, _collection
    if _collection is not None:
        return _collection

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    _collection = _client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=_get_embedding_function(),
    )
    return _collection


def reset_collection():
    """Удаляет и заново создаёт коллекцию — используется перед полной переиндексацией."""
    global _client, _collection
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    if _client is None:
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    _client.delete_collection(name=config.COLLECTION_NAME)
    _collection = _client.create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=_get_embedding_function(),
    )
    return _collection


def add_chunks(ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
    if not ids:
        return
    collection = get_collection()
    collection.add(ids=ids, documents=texts, metadatas=metadatas)


def query(question: str, top_k: int = config.TOP_K) -> list[dict]:
    """Ищет top_k ближайших по смыслу фрагментов. Возвращает список dict с
    ключами text, metadata, distance, отсортированный от самого релевантного."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    result = collection.query(
        query_texts=[question],
        n_results=min(top_k, collection.count()),
    )

    chunks = []
    for text, metadata, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        chunks.append({"text": text, "metadata": metadata, "distance": distance})
    return chunks
