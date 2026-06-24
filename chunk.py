"""
chunk.py
---------
Contains the core text-chunking logic using LangChain's RecursiveCharacterTextSplitter.
Kept separate from main.py as required by the assignment, so the chunking
logic can be reused/tested independently of the API layer.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 50, chunk_overlap: int = 10) -> list[str]:
    """
    Splits a large block of text into smaller chunks using RecursiveCharacterTextSplitter.

    Args:
        text: The raw input text to split.
        chunk_size: Maximum number of characters per chunk (default 50).
        chunk_overlap: Number of overlapping characters between consecutive
                       chunks, used to preserve context across chunk boundaries
                       (default 10).

    Returns:
        A list of text chunks (strings).

    Raises:
        ValueError: if chunk_overlap is greater than or equal to chunk_size,
                    since that configuration is invalid (it would either loop
                    forever or produce nonsensical chunks).
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than "
            f"chunk_size ({chunk_size})."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_text(text)
    return chunks
