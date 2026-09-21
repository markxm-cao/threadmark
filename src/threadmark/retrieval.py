import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from threadmark.chunking import CodeChunk


MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "NamanAgnih0tri/code-reranker-miniLM-staqc"


@dataclass
class RetrievalResult:
    chunk: CodeChunk
    score: float


def tokenize_basic(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric terms."""
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def split_identifier(identifier: str) -> list[str]:
    """Split snake_case and camelCase identifiers into lowercase components."""
    parts = []

    for snake_part in identifier.split("_"):
        camel_parts = re.findall(
            r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+",
            snake_part,
        )

        parts.extend(
            part.lower()
            for part in camel_parts
        )

    return parts


def tokenize_code_aware(text: str) -> list[str]:
    """Preserve complete identifiers while also indexing their components."""
    raw_tokens = re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*|\d+",
        text,
    )

    tokens = []

    for raw_token in raw_tokens:
        normalized = raw_token.lower()
        tokens.append(normalized)

        for part in split_identifier(raw_token):
            if part != normalized:
                tokens.append(part)

    return tokens


def chunk_to_text(chunk: CodeChunk) -> str:
    """Convert a code chunk into retrieval text."""
    parts = [
        f"File: {chunk.file_path}",
        f"Lines: {chunk.start_line}-{chunk.end_line}",
    ]

    if chunk.symbol_name is not None:
        parts.append(
            f"Symbol: {chunk.symbol_name}"
        )

    parts.append(chunk.content)

    return "\n".join(parts)


def embed_chunks(
    chunks: list[CodeChunk],
    model: SentenceTransformer,
):
    """Encode code chunks as normalized embedding vectors."""
    texts = [
        chunk_to_text(chunk)
        for chunk in chunks
    ]

    return model.encode(
        texts,
        normalize_embeddings=True,
    )


def search_chunks_semantic(
    query: str,
    chunks: list[CodeChunk],
    embeddings,
    model: SentenceTransformer,
    top_k: int = 5,
) -> list[RetrievalResult]:
    """Rank code chunks by cosine similarity to the query embedding."""
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    scores = embeddings @ query_embedding
    ranked_indices = scores.argsort()[::-1][:top_k]

    return [
        RetrievalResult(
            chunk=chunks[index],
            score=float(scores[index]),
        )
        for index in ranked_indices
    ]


def search_chunks_bm25(
    query: str,
    chunks: list[CodeChunk],
    top_k: int = 5,
) -> list[RetrievalResult]:
    """Rank code chunks by BM25 lexical relevance to the query."""
    corpus = [
        tokenize_basic(chunk_to_text(chunk))
        for chunk in chunks
    ]

    bm25 = BM25Okapi(corpus)

    query_tokens = tokenize_basic(query)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = scores.argsort()[::-1][:top_k]

    return [
        RetrievalResult(
            chunk=chunks[index],
            score=float(scores[index]),
        )
        for index in ranked_indices
    ]


def search_chunks_hybrid(
    query: str,
    chunks: list[CodeChunk],
    embeddings,
    model: SentenceTransformer,
    top_k: int = 5,
) -> list[RetrievalResult]:
    """Combine semantic and BM25 rankings using reciprocal rank fusion."""
    semantic_results = search_chunks_semantic(
        query,
        chunks,
        embeddings,
        model,
        top_k=len(chunks),
    )

    lexical_results = search_chunks_bm25(
        query,
        chunks,
        top_k=len(chunks),
    )

    rrf_scores = {}
    k = 60

    for rank, result in enumerate(semantic_results, start=1):
        chunk = result.chunk
        key = (
            chunk.file_path,
            chunk.start_line,
            chunk.end_line,
        )

        rrf_scores[key] = (
            rrf_scores.get(key, 0)
            + 1 / (k + rank)
        )

    for rank, result in enumerate(lexical_results, start=1):
        chunk = result.chunk
        key = (
            chunk.file_path,
            chunk.start_line,
            chunk.end_line,
        )

        rrf_scores[key] = (
            rrf_scores.get(key, 0)
            + 1 / (k + rank)
        )

    chunk_lookup = {
        (
            chunk.file_path,
            chunk.start_line,
            chunk.end_line,
        ): chunk
        for chunk in chunks
    }

    ranked = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        RetrievalResult(
            chunk=chunk_lookup[key],
            score=score,
        )
        for key, score in ranked[:top_k]
    ]


def rerank_results(
    query: str,
    results: list[RetrievalResult],
    reranker: CrossEncoder,
    top_k: int = 5,
) -> list[RetrievalResult]:
    """Rerank retrieved chunks using a cross-encoder relevance model."""
    if not results:
        return []

    pairs = [
        (query, chunk_to_text(result.chunk))
        for result in results
    ]

    scores = reranker.predict(pairs)

    reranked = [
        RetrievalResult(
            chunk=result.chunk,
            score=float(score),
        )
        for result, score in zip(results, scores)
    ]

    reranked.sort(
        key=lambda result: result.score,
        reverse=True,
    )

    return reranked[:top_k]