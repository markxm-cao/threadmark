import math
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from threadmark.chunking import CodeChunk
from threadmark.behavior import BehaviorFact
from threadmark.control_flow import (
    CodeMotif,
    extract_chunk_motifs,
)
from threadmark.repository import (
    find_source_files,
    read_source_file,
)


CONCEPT_MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class StructuralResult:
    chunk: CodeChunk
    score: float
    matched_behaviors: list[str]
    behavior_score: float
    relation_score: float
    concept_score: float
    

@dataclass(frozen=True)
class BehaviorRelation:
    source: str
    target: str


@dataclass(frozen=True)
class RelationEvidence:
    source: str
    target: str

    source_line: int
    target_line: int

    source_detail: str
    target_detail: str
    
    
def score_chunk_behaviors(
    chunk: CodeChunk,
    facts: list[BehaviorFact],
    query_behaviors: list[str],
    behavior_weights: dict[str, float],
) -> StructuralResult:
    """Score a chunk by weighted coverage of predicted behaviors."""

    chunk_behaviors = get_chunk_behaviors(
        chunk,
        facts,
    )

    wanted_behaviors = set(query_behaviors)

    matched = wanted_behaviors & chunk_behaviors

    if not wanted_behaviors:
        score = 0.0
    else:
        matched_weight = sum(
            behavior_weights.get(behavior, 1.0)
            for behavior in matched
        )

        total_weight = sum(
            behavior_weights.get(behavior, 1.0)
            for behavior in wanted_behaviors
        )

        score = matched_weight / total_weight

    return StructuralResult(
        chunk=chunk,
        score=score,
        matched_behaviors=sorted(matched),
        behavior_score=score,
        relation_score=0.0,
        concept_score=0.0,
    )
    
    
def score_chunk_relations(
    chunk: CodeChunk,
    facts: list[BehaviorFact],
    query_relations,
) -> tuple[float, list[BehaviorRelation]]:
    """Score a chunk by coverage of predicted behavior relations."""

    chunk_relations = get_chunk_relations(
        chunk,
        facts,
    )

    wanted_relations = {
        BehaviorRelation(
            source=relation.source,
            target=relation.target,
        )
        for relation in query_relations
    }

    matched = wanted_relations & chunk_relations

    if not wanted_relations:
        return 0.0, []

    score = (
        len(matched)
        / len(wanted_relations)
    )

    return score, sorted(
        matched,
        key=lambda relation: (
            relation.source,
            relation.target,
        ),
    )
    
    
def search_chunks_structural(
    chunks: list[CodeChunk],
    facts: list[BehaviorFact],
    query_behaviors: list[str],
    behavior_weights: dict[str, float],
    top_k: int = 5,
) -> list[StructuralResult]:
    """Rank chunks by predicted-behavior coverage."""

    results = [
        score_chunk_behaviors(
            chunk,
            facts,
            query_behaviors,
            behavior_weights,
        )
        for chunk in chunks
    ]

    results.sort(
        key=lambda result: result.score,
        reverse=True,
    )

    return results[:top_k]


def get_chunk_behaviors(
    chunk: CodeChunk,
    facts: list[BehaviorFact],
) -> set[str]:
    """Return the abstract behaviors present inside a code chunk."""

    return {
        fact.behavior
        for fact in facts
        if (
            fact.behavior is not None
            and fact.file_path == chunk.file_path
            and chunk.start_line <= fact.line <= chunk.end_line
        )
    }
    
    
def get_chunk_relations(
    chunk: CodeChunk,
    facts: list[BehaviorFact],
) -> set[BehaviorRelation]:
    """Extract condition-to-action behavior relations inside a chunk."""

    relevant_facts = [
        fact
        for fact in facts
        if (
            fact.file_path == chunk.file_path
            and chunk.start_line <= fact.line <= chunk.end_line
        )
    ]

    conditions_by_detail = {
        fact.detail: fact
        for fact in relevant_facts
        if (
            fact.kind == "condition"
            and fact.behavior is not None
        )
    }

    components_by_line: dict[int, set[str]] = {}

    for fact in relevant_facts:
        if (
            fact.kind == "condition_component"
            and fact.behavior is not None
        ):
            components_by_line.setdefault(
                fact.line,
                set(),
            ).add(fact.behavior)

    relations = set()

    for fact in relevant_facts:
        if (
            fact.condition is None
            or fact.behavior is None
        ):
            continue

        controller = conditions_by_detail.get(
            fact.condition
        )

        if controller is None:
            continue

        source_behaviors = {
            controller.behavior
        }

        if controller.behavior == "compound_condition":
            source_behaviors.update(
                components_by_line.get(
                    controller.line,
                    set(),
                )
            )

        for source_behavior in source_behaviors:
            if source_behavior in {
                "generic_condition",
                "compound_condition",
            }:
                continue

            relations.add(
                BehaviorRelation(
                    source=source_behavior,
                    target=fact.behavior,
                )
            )

    return relations


def get_chunk_relation_evidence(
    chunk: CodeChunk,
    facts: list[BehaviorFact],
) -> list[RelationEvidence]:
    """Extract condition-to-action relations together with source evidence."""

    relevant_facts = [
        fact
        for fact in facts
        if (
            fact.file_path == chunk.file_path
            and chunk.start_line <= fact.line <= chunk.end_line
        )
    ]

    conditions_by_detail = {
        fact.detail: fact
        for fact in relevant_facts
        if (
            fact.kind == "condition"
            and fact.behavior is not None
        )
    }

    components_by_line: dict[int, list[BehaviorFact]] = {}

    for fact in relevant_facts:
        if (
            fact.kind == "condition_component"
            and fact.behavior is not None
        ):
            components_by_line.setdefault(
                fact.line,
                [],
            ).append(fact)

    evidence = []

    for fact in relevant_facts:
        if (
            fact.condition is None
            or fact.behavior is None
        ):
            continue

        controller = conditions_by_detail.get(
            fact.condition
        )

        if controller is None:
            continue

        source_facts = [controller]

        if controller.behavior == "compound_condition":
            source_facts = components_by_line.get(
                controller.line,
                [],
            )

        for source_fact in source_facts:
            if source_fact.behavior in {
                "generic_condition",
                "compound_condition",
            }:
                continue

            evidence.append(
                RelationEvidence(
                   source=source_fact.behavior,
                    target=fact.behavior,
                    source_line=source_fact.line,
                    target_line=fact.line,
                    source_detail=source_fact.detail,
                    target_detail=fact.detail,
                )
            )

    return evidence


def compute_behavior_weights(
    chunks: list[CodeChunk],
    facts: list[BehaviorFact],
) -> dict[str, float]:
    """Compute IDF-like weights for behaviors based on chunk rarity."""

    behavior_counts: dict[str, int] = {}

    for chunk in chunks:
        chunk_behaviors = get_chunk_behaviors(
            chunk,
            facts,
        )

        for behavior in chunk_behaviors:
            behavior_counts[behavior] = (
                behavior_counts.get(behavior, 0) + 1
            )

    total_chunks = len(chunks)

    weights = {}

    for behavior, count in behavior_counts.items():
        weights[behavior] = (
            math.log(
                (total_chunks + 1)
                / (count + 1)
            )
            + 1
        )

    return weights


def build_chunk_behavior_text(
    chunk: CodeChunk,
    facts: list[BehaviorFact],
) -> str:
    """Build searchable text from the behaviors and identifiers inside a chunk."""

    relevant_facts = [
        fact
        for fact in facts
        if (
            fact.file_path == chunk.file_path
            and chunk.start_line <= fact.line <= chunk.end_line
        )
    ]

    parts = [
        f"File: {chunk.file_path}",
    ]

    if chunk.symbol_name is not None:
        parts.append(
            f"Symbol: {chunk.symbol_name}"
        )

    for fact in relevant_facts:
        parts.append(
            f"{fact.operation}: {fact.detail}"
        )

    return "\n".join(parts)


def build_concept_text(concepts: list[str]) -> str:
    """Combine query-plan concepts into one semantic search string."""

    return "\n".join(concepts)


def embed_chunk_behavior_texts(
    chunks: list[CodeChunk],
    facts: list[BehaviorFact],
    model: SentenceTransformer,
):
    """Embed behavior-focused representations of all code chunks."""

    texts = [
        build_chunk_behavior_text(chunk, facts)
        for chunk in chunks
    ]

    return model.encode(
        texts,
        normalize_embeddings=True,
    )
    
    
def compute_concept_scores(
    concepts: list[str],
    chunk_behavior_embeddings,
    model: SentenceTransformer,
):
    """Compute semantic similarity between query concepts and chunk behavior text."""

    concept_text = build_concept_text(concepts)

    concept_embedding = model.encode(
        concept_text,
        normalize_embeddings=True,
    )

    return chunk_behavior_embeddings @ concept_embedding


def build_source_cache(
    repo_path: str,
) -> dict[str, str]:
    """Load Python source files once for motif extraction."""

    source_cache = {}

    for file_path in find_source_files(repo_path):
        if not file_path.endswith(".py"):
            continue

        lines = read_source_file(
            repo_path,
            file_path,
        )

        source_cache[file_path] = "\n".join(
            line
            for _, line in lines
        )

    return source_cache


def extract_repository_chunk_motifs(
    chunks: list[CodeChunk],
    source_cache: dict[str, str],
) -> dict[tuple[str, int, int], list[CodeMotif]]:
    """Extract supported motifs for every candidate chunk."""

    motifs_by_chunk = {}

    for chunk in chunks:
        source = source_cache.get(
            chunk.file_path
        )

        if source is None:
            motifs = []
        else:
            motifs = extract_chunk_motifs(
                chunk,
                source,
            )

        key = (
            chunk.file_path,
            chunk.start_line,
            chunk.end_line,
        )

        motifs_by_chunk[key] = motifs

    return motifs_by_chunk


def build_motif_text(
    motif: CodeMotif,
) -> str:
    """Build a semantic representation of a code motif."""

    readable_kind = motif.kind.replace(
        "_",
        " ",
    )

    return (
        f"Behavior pattern: {readable_kind}\n"
        f"{motif.detail}"
    )


def build_motif_query_text(
    query: str,
    plan: QueryPlan,
) -> str:
    """Build semantic query text for motif matching."""

    concepts = "\n".join(
        plan.concepts
    )

    return (
        f"Question: {query}\n"
        f"Intent: {plan.intent}\n"
        f"Concepts:\n{concepts}"
    )


def embed_repository_motifs(
    motifs_by_chunk,
    model: SentenceTransformer,
):
    """Embed canonical motif descriptions for each chunk."""

    embeddings_by_chunk = {}

    for key, motifs in motifs_by_chunk.items():
        if not motifs:
            continue

        texts = [
            build_motif_text(motif)
            for motif in motifs
        ]

        embeddings_by_chunk[key] = model.encode(
            texts,
            normalize_embeddings=True,
        )

    return embeddings_by_chunk


def compute_motif_score(
    query: str,
    plan: QueryPlan,
    chunk_key,
    motif_embeddings_by_chunk,
    model: SentenceTransformer,
) -> float:
    """Return the best semantic match between the query and a chunk motif."""

    motif_embeddings = (
        motif_embeddings_by_chunk.get(
            chunk_key
        )
    )

    if motif_embeddings is None:
        return 0.0

    query_text = build_motif_query_text(
        query,
        plan,
    )

    query_embedding = model.encode(
        query_text,
        normalize_embeddings=True,
    )

    scores = (
        motif_embeddings
        @ query_embedding
    )

    return float(scores.max())





# PYTHONPATH=src python -m threadmark.structural_retrieval
from threadmark.repository import clone_repository
from threadmark.chunking import chunk_repository_ast
from threadmark.behavior import extract_repository_behaviors
from threadmark.query_planner import (
    QueryPlan,
    PlannedRelation,
    load_frozen_query_plans,
    query_plan_from_dict,
)



if __name__ == "__main__":
    frozen_plans = load_frozen_query_plans()
    repo_path = clone_repository(
        "https://github.com/nartnek/RiftPredict",
        "data/repos",
    )

    chunks = chunk_repository_ast(repo_path)
    facts = extract_repository_behaviors(repo_path)
    source_cache = build_source_cache(
        repo_path   
    )

    concept_model = SentenceTransformer(
        CONCEPT_MODEL_NAME
    )
    
    motifs_by_chunk = (
        extract_repository_chunk_motifs(
            chunks,
            source_cache,
        )
    )
    
    motif_embeddings_by_chunk = (
        embed_repository_motifs(
            motifs_by_chunk,
            concept_model,
        )
    )
        
    chunk_behavior_embeddings = embed_chunk_behavior_texts(
        chunks,
        facts,
        concept_model,
    )
    
    behavior_weights = compute_behavior_weights(
        chunks,
        facts,
    )

    queries = [
        "How does the crawler avoid collecting duplicate match IDs?",
        "How does the crawler discover new players from downloaded matches?",
        "How does the crawler periodically checkpoint its progress?",
    ]

    for query in queries:
        print(f"\n\nQUESTION: {query}")

        plan = query_plan_from_dict(
            frozen_plans[query]
        )
        concept_scores = compute_concept_scores(
            plan.concepts,
            chunk_behavior_embeddings,
            concept_model,
        )

        print(
            "Behaviors:",
            plan.likely_behaviors,
        )
        
        print("Behavior weights:")

        for behavior in plan.likely_behaviors:
            weight = behavior_weights.get(
                behavior,
                1.0,
            )

            print(
                f"  {behavior:<24} {weight:.3f}"
            )

        results = search_chunks_structural(
            chunks=chunks,
            facts=facts,
            query_behaviors=plan.likely_behaviors,
            behavior_weights=behavior_weights,
            top_k=10,
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):
            chunk = result.chunk
            
            chunk_key = (
                chunk.file_path,
                chunk.start_line,
                chunk.end_line,
            )
            
            motifs = motifs_by_chunk.get(
                            chunk_key,
                            [],
                        )
            
            relation_score, matched_relations = (
                            score_chunk_relations(
                                chunk,
                                facts,
                                plan.likely_relations,
                            )
                        )
            
            chunk_index = chunks.index(chunk)
                        
            concept_score = float(
                concept_scores[chunk_index]
            )
            
            motif_score = compute_motif_score(
                query,
                plan,
                chunk_key,
                motif_embeddings_by_chunk,
                concept_model,
            )
        
            print(
                f"\nRank {rank} | "
                f"Score {result.score:.3f}"
            )

            print(
                f"Behavior: {result.behavior_score:.3f} | "
                f"Relation: {relation_score:.3f} | "
                f"Concept: {concept_score:.3f} | "
                f"Motif: {motif_score:.3f}"
            )

            print(
                f"{chunk.file_path}:"
                f"{chunk.start_line}-{chunk.end_line}"
            )

            print(
                "Matched:",
                result.matched_behaviors,
            )

            if matched_relations:
                print("Matched relations:")

                for relation in matched_relations:
                    print(
                        f"  {relation.source}"
                        f" -> "
                        f"{relation.target}"
                    )
                    
            if motifs:
                print("Motifs:")

                for motif in motifs:
                    print(
                        f"  {motif.kind}: "
                        f"{motif.detail}"
                    )
            
    interesting_ranges = [
        (599, 638),
        (659, 698),
        (689, 728),
    ]

    for chunk in chunks:
        if (
            chunk.file_path
            == "src/data_collection/collect_matches.py"
            and (chunk.start_line, chunk.end_line)
            in interesting_ranges
        ):
            print(
                f"\n{chunk.file_path}:"
                f"{chunk.start_line}-{chunk.end_line}"
            )

            relations = get_chunk_relations(
                chunk,
                facts,
            )

            for relation in sorted(
                relations,
                key=lambda r: (r.source, r.target),
            ):
                print(
                    f"  {relation.source}"
                    f" -> "
                    f"{relation.target}"
                )
                
    debug_ranges = [
        (539, 578),
        (599, 638),
    ]

    for chunk in chunks:
        if (
            chunk.file_path
            == "src/data_collection/collect_matches.py"
            and (chunk.start_line, chunk.end_line)
            in debug_ranges
        ):
            print(
                f"\n=== {chunk.start_line}-{chunk.end_line} ==="
            )

            evidence = get_chunk_relation_evidence(
                chunk,
                facts,
            )

            for relation in evidence:
                if (
                    relation.source == "membership_test"
                    and relation.target
                    in {"skip_processing", "record_item"}
                ):
                    print(
                        f"{relation.source_line}: "
                        f"{relation.source_detail}"
                    )
                    print(
                        f"  -> {relation.target}"
                        f" at {relation.target_line}: "
                        f"{relation.target_detail}"
                    )