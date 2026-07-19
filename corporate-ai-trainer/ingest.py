"""Загрузка документов из knowledge_base/ и индексация их в ChromaDB.

Запуск: python ingest.py
Поддерживаемые форматы: .txt, .md, .pdf, .docx
"""
import sys
from pathlib import Path

import docx
from pypdf import PdfReader

import config
import vectorstore


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_docx(path: Path) -> str:
    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


READERS = {
    ".txt": read_txt,
    ".md": read_txt,
    ".pdf": read_pdf,
    ".docx": read_docx,
}


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Режет текст на перекрывающиеся фрагменты по количеству символов,
    стараясь не рвать текст посреди слова."""
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space
        chunks.append(text[start:end].strip())
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def load_documents(knowledge_dir: Path) -> list[tuple[str, str]]:
    """Возвращает список (имя_файла, текст) для всех поддерживаемых файлов."""
    documents = []
    for path in sorted(knowledge_dir.rglob("*")):
        if not path.is_file():
            continue
        reader = READERS.get(path.suffix.lower())
        if reader is None:
            continue
        try:
            text = reader(path)
        except Exception as exc:
            print(f"[!] Не удалось прочитать {path}: {exc}", file=sys.stderr)
            continue
        if text.strip():
            documents.append((str(path.relative_to(knowledge_dir)), text))
    return documents


def main() -> None:
    if not config.KNOWLEDGE_DIR.exists():
        print(f"Папка {config.KNOWLEDGE_DIR} не найдена.", file=sys.stderr)
        sys.exit(1)

    documents = load_documents(config.KNOWLEDGE_DIR)
    if not documents:
        print(
            f"В {config.KNOWLEDGE_DIR} не найдено ни одного .txt/.md/.pdf/.docx файла."
        )
        sys.exit(0)

    vectorstore.reset_collection()

    total_chunks = 0
    for source_name, text in documents:
        chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        ids = [f"{source_name}::{i}" for i in range(len(chunks))]
        metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]
        vectorstore.add_chunks(ids=ids, texts=chunks, metadatas=metadatas)
        total_chunks += len(chunks)
        print(f"[+] {source_name}: {len(chunks)} фрагментов")

    print(f"\nГотово. Проиндексировано документов: {len(documents)}, фрагментов: {total_chunks}")


if __name__ == "__main__":
    main()
