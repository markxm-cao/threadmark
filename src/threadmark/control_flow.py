import ast
from dataclasses import dataclass


COLLECTION_MUTATIONS = {
    "add",
    "append",
    "appendleft",
    "extend",
    "update",
}

QUEUE_METHODS = {
    "append",
    "appendleft",
}

TRACKING_METHODS = {
    "add",
}


@dataclass(frozen=True)
class MembershipGuard:
    file_path: str
    symbol_name: str

    condition_line: int
    item: str
    collection: str
    operator: str

    terminator_line: int
    terminator: str

    followup_line: int | None = None
    followup_operation: str | None = None
    followup_item: str | None = None
    followup_collection: str | None = None
    
    
@dataclass(frozen=True)
class DeduplicationGuard:
    file_path: str
    symbol_name: str

    condition_line: int
    item: str
    collection: str

    terminator_line: int
    terminator: str

    mutation_line: int
    mutation_method: str
    

@dataclass(frozen=True)
class CodeMotif:
    file_path: str
    symbol_name: str
    start_line: int
    end_line: int
    kind: str
    detail: str
    

@dataclass(frozen=True)
class NovelItemDiscovery:
    file_path: str
    symbol_name: str

    condition_line: int
    item: str

    checked_collections: tuple[str, ...]

    queue_line: int
    queue_collection: str
    queue_method: str

    record_line: int
    record_collection: str
    record_method: str

    
    
def parse_membership_condition(
    node: ast.AST,
) -> tuple[str, str, str] | None:
    """Extract item, collection, and operator from a simple membership test."""

    if not isinstance(node, ast.Compare):
        return None

    if len(node.ops) != 1:
        return None

    if len(node.comparators) != 1:
        return None

    operator = node.ops[0]

    if isinstance(operator, ast.In):
        operator_name = "in"
    elif isinstance(operator, ast.NotIn):
        operator_name = "not_in"
    else:
        return None

    item = ast.unparse(node.left)
    collection = ast.unparse(
        node.comparators[0]
    )

    return (
        item,
        collection,
        operator_name,
    )
    
    
def parse_non_membership_checks(
    node: ast.AST,
) -> list[tuple[str, str]]:
    """
    Extract item/collection pairs from one or more
    `item not in collection` conditions joined by AND.
    """

    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, ast.And):
            return []

        checks = []

        for value in node.values:
            checks.extend(
                parse_non_membership_checks(value)
            )

        return checks

    if not isinstance(node, ast.Compare):
        return []

    if (
        len(node.ops) != 1
        or len(node.comparators) != 1
    ):
        return []

    if not isinstance(node.ops[0], ast.NotIn):
        return []

    item = ast.unparse(node.left)

    collection = ast.unparse(
        node.comparators[0]
    )

    return [
        (
            item,
            collection,
        )
    ]
    
    
def find_terminator(
    statements: list[ast.stmt],
) -> ast.stmt | None:
    """Return a path-terminating statement in an if body."""

    for statement in statements:
        if isinstance(
            statement,
            (
                ast.Continue,
                ast.Break,
                ast.Return,
                ast.Raise,
            ),
        ):
            return statement

    return None


def find_novel_item_discoveries(
    statements: list[ast.stmt],
    file_path: str,
    symbol_name: str,
    start_line: int,
    end_line: int,
) -> list[NovelItemDiscovery]:
    """Find guarded insertion of previously unseen items."""

    discoveries = []

    for statement in statements:
        if not isinstance(statement, ast.If):
            continue

        if not (
            start_line
            <= statement.lineno
            <= end_line
        ):
            continue

        checks = parse_non_membership_checks(
            statement.test
        )

        if not checks:
            continue

        items = {
            item
            for item, _ in checks
        }

        # All membership checks must concern the same item.
        if len(items) != 1:
            continue

        item = next(iter(items))

        checked_collections = tuple(
            collection
            for _, collection in checks
        )

        queue_mutation = None
        record_mutation = None

        for body_statement in statement.body:
            for node in ast.walk(body_statement):
                mutation = parse_collection_mutation(
                    node
                )

                if mutation is None:
                    continue

                collection, method, mutated_item = (
                    mutation
                )

                if mutated_item != item:
                    continue

                if (
                    method in QUEUE_METHODS
                    and queue_mutation is None
                ):
                    queue_mutation = (
                        node.lineno,
                        collection,
                        method,
                    )

                if (
                    method in TRACKING_METHODS
                    and record_mutation is None
                ):
                    record_mutation = (
                        node.lineno,
                        collection,
                        method,
                    )

        if (
            queue_mutation is None
            or record_mutation is None
        ):
            continue

        queue_line, queue_collection, queue_method = (
            queue_mutation
        )

        record_line, record_collection, record_method = (
            record_mutation
        )

        # Keep all evidence inside this candidate chunk.
        if (
            queue_line > end_line
            or record_line > end_line
        ):
            continue

        # Strong extra constraint:
        # one of the collections checked for absence
        # should also be updated to record the item.
        if record_collection not in checked_collections:
            continue

        discoveries.append(
            NovelItemDiscovery(
                file_path=file_path,
                symbol_name=symbol_name,
                condition_line=statement.lineno,
                item=item,
                checked_collections=checked_collections,
                queue_line=queue_line,
                queue_collection=queue_collection,
                queue_method=queue_method,
                record_line=record_line,
                record_collection=record_collection,
                record_method=record_method,
            )
        )

    return discoveries


def novel_item_discovery_to_motif(
    discovery: NovelItemDiscovery,
) -> CodeMotif:
    checked = ", ".join(
        discovery.checked_collections
    )

    return CodeMotif(
        file_path=discovery.file_path,
        symbol_name=discovery.symbol_name,
        start_line=discovery.condition_line,
        end_line=max(
            discovery.queue_line,
            discovery.record_line,
        ),
        kind="novel_item_discovery",
        detail=(
            f"When {discovery.item} is absent from "
            f"{checked}, add {discovery.item} to "
            f"{discovery.queue_collection} and record it "
            f"in {discovery.record_collection}."
        ),
    )


def parse_collection_mutation(
    node: ast.AST,
) -> tuple[str, str, str] | None:
    """Extract collection, mutation method, and item from a method call."""

    if not isinstance(node, ast.Expr):
        return None

    call = node.value

    if not isinstance(call, ast.Call):
        return None

    if not isinstance(call.func, ast.Attribute):
        return None

    method = call.func.attr

    if method not in COLLECTION_MUTATIONS:
        return None

    if not call.args:
        return None

    collection = ast.unparse(
        call.func.value
    )

    item = ast.unparse(
        call.args[0]
    )

    return (
        collection,
        method,
        item,
    )
    
    
def extract_guards_from_block(
    statements: list[ast.stmt],
    file_path: str,
    symbol_name: str,
) -> list[MembershipGuard]:
    guards = []

    for index, statement in enumerate(statements):
        if not isinstance(statement, ast.If):
            continue

        membership = parse_membership_condition(
            statement.test
        )

        if membership is None:
            continue

        terminator = find_terminator(
            statement.body
        )

        if terminator is None:
            continue

        item, collection, operator = membership

        guard = MembershipGuard(
            file_path=file_path,
            symbol_name=symbol_name,
            condition_line=statement.lineno,
            item=item,
            collection=collection,
            operator=operator,
            terminator_line=terminator.lineno,
            terminator=type(terminator).__name__.lower(),
        )

        guards.append(guard)

    return guards


def find_followup_mutations(
    statements: list[ast.stmt],
    start_index: int,
    item: str,
) -> list[tuple[int, str, str, str]]:
    """
    Find later collection mutations involving the same item.

    Returns:
        (line, collection, method, item)
    """

    mutations = []

    for statement in statements[start_index + 1:]:
        for node in ast.walk(statement):
            mutation = parse_collection_mutation(node)

            if mutation is None:
                continue

            collection, method, mutated_item = mutation

            if mutated_item != item:
                continue

            mutations.append(
                (
                    node.lineno,
                    collection,
                    method,
                    mutated_item,
                )
            )

    return mutations


def analyze_block(
    statements: list[ast.stmt],
) -> None:
    """Print terminating membership guards and their fall-through mutations."""

    for index, statement in enumerate(statements):
        if isinstance(statement, ast.If):
            membership = parse_membership_condition(
                statement.test
            )

            terminator = find_terminator(
                statement.body
            )

            if (
                membership is not None
                and terminator is not None
            ):
                item, collection, operator = membership

                print(
                    f"\n{statement.lineno}: "
                    f"{item} {operator} {collection}"
                )

                print(
                    f"    TRUE -> "
                    f"{type(terminator).__name__.lower()} "
                    f"at {terminator.lineno}"
                )

                followups = find_followup_mutations(
                    statements,
                    index,
                    item,
                )

                for (
                    line,
                    followup_collection,
                    method,
                    followup_item,
                ) in followups:
                    same_collection = (
                        followup_collection == collection
                    )

                    print(
                        f"    FALSE/FALLTHROUGH -> "
                        f"{followup_collection}.{method}"
                        f"({followup_item}) "
                        f"at {line}"
                    )

                    print(
                        f"        same item: yes"
                    )

                    print(
                        f"        same collection: "
                        f"{same_collection}"
                    )

        # Recursively analyze nested statement blocks.
        for _, value in ast.iter_fields(statement):
            if not isinstance(value, list):
                continue

            child_statements = [
                child
                for child in value
                if isinstance(child, ast.stmt)
            ]

            if child_statements:
                analyze_block(child_statements)
                
                
def find_deduplication_guards(
    statements: list[ast.stmt],
    file_path: str,
    symbol_name: str,
    start_line: int,
    end_line: int,
) -> list[DeduplicationGuard]:
    """Find skip-if-seen then record-if-new motifs inside a line range."""

    guards = []

    for index, statement in enumerate(statements):
        if not isinstance(statement, ast.If):
            continue

        # The guard itself must belong to this chunk.
        if not (
            start_line
            <= statement.lineno
            <= end_line
        ):
            continue

        membership = parse_membership_condition(
            statement.test
        )

        if membership is None:
            continue

        item, collection, operator = membership

        # For this first motif, we're looking specifically for:
        #
        # if item in collection:
        #     continue / return / break / raise
        if operator != "in":
            continue

        terminator = find_terminator(
            statement.body
        )

        if terminator is None:
            continue

        followups = find_followup_mutations(
            statements,
            index,
            item,
        )

        for (
            line,
            followup_collection,
            method,
            followup_item,
        ) in followups:

            # Keep the entire motif inside this chunk.
            if line > end_line:
                continue

            # The important part:
            # same item goes into the SAME collection
            # that was checked by the guard.
            if followup_collection != collection:
                continue

            guards.append(
                DeduplicationGuard(
                    file_path=file_path,
                    symbol_name=symbol_name,
                    condition_line=statement.lineno,
                    item=item,
                    collection=collection,
                    terminator_line=terminator.lineno,
                    terminator=type(
                        terminator
                    ).__name__.lower(),
                    mutation_line=line,
                    mutation_method=method,
                )
            )

    return guards


def deduplication_guard_to_motif(
    guard: DeduplicationGuard,
) -> CodeMotif:
    return CodeMotif(
        file_path=guard.file_path,
        symbol_name=guard.symbol_name,
        start_line=guard.condition_line,
        end_line=guard.mutation_line,
        kind="deduplication_guard",
        detail=(
            f"Skip {guard.item} when it already exists in "
            f"{guard.collection}; otherwise add {guard.item} "
            f"to {guard.collection}."
        ),
    )


def extract_chunk_motifs(
    chunk: CodeChunk,
    source: str,
) -> list[CodeMotif]:
    """Extract supported control-flow motifs contained within one code chunk."""

    tree = ast.parse(source)

    motifs = []

    def iter_statement_blocks(
        statements: list[ast.stmt],
    ):
        yield statements

        for statement in statements:
            for _, value in ast.iter_fields(statement):
                if not isinstance(value, list):
                    continue

                child_statements = [
                    child
                    for child in value
                    if isinstance(child, ast.stmt)
                ]

                if child_statements:
                    yield from iter_statement_blocks(
                        child_statements
                    )

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue

        for block in iter_statement_blocks(node.body):
            dedup_guards = find_deduplication_guards(
                statements=block,
                file_path=chunk.file_path,
                symbol_name=node.name,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
            )
            
            for guard in dedup_guards:
                motifs.append(
                    deduplication_guard_to_motif(
                        guard
                    )
                )

            discoveries = find_novel_item_discoveries(
                statements=block,
                file_path=chunk.file_path,
                symbol_name=node.name,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
            )

            for discovery in discoveries:
                motifs.append(
                    novel_item_discovery_to_motif(
                        discovery
                    )
                )

    return list(dict.fromkeys(motifs))




# PYTHONPATH=src python -m threadmark.control_flow
if __name__ == "__main__":
    from threadmark.repository import (
        clone_repository,
        read_source_file,
    )

    repo_path = clone_repository(
        "https://github.com/nartnek/RiftPredict",
        "data/repos",
    )

    file_path = "src/data_collection/collect_matches.py"

    lines = read_source_file(
        repo_path,
        file_path,
    )

    source = "\n".join(
        line
        for _, line in lines
    )

    tree = ast.parse(source)

    # Find collect_matches()
    collect_matches_node = None

    for node in tree.body:
        if (
            isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            )
            and node.name == "collect_matches"
        ):
            collect_matches_node = node
            break

    if collect_matches_node is None:
        raise ValueError(
            "collect_matches function not found"
        )

    # Recursively yield every block of sibling statements.
    def iter_statement_blocks(
        statements: list[ast.stmt],
    ):
        yield statements

        for statement in statements:
            for _, value in ast.iter_fields(statement):
                if not isinstance(value, list):
                    continue

                child_statements = [
                    child
                    for child in value
                    if isinstance(child, ast.stmt)
                ]

                if child_statements:
                    yield from iter_statement_blocks(
                        child_statements
                    )

    test_ranges = [
        (629, 668),
        (659, 698),
    ]

    for start_line, end_line in test_ranges:
        print(
            f"\n=== {start_line}-{end_line} ==="
        )

        discoveries = set()

        for block in iter_statement_blocks(
            collect_matches_node.body
        ):
            found = find_novel_item_discoveries(
                statements=block,
                file_path=file_path,
                symbol_name="collect_matches",
                start_line=start_line,
                end_line=end_line,
            )

            discoveries.update(found)

        if not discoveries:
            print(
                "No novel-item discovery found."
            )
            continue

        for discovery in sorted(
            discoveries,
            key=lambda item: item.condition_line,
        ):
            motif = novel_item_discovery_to_motif(
                discovery
            )

            print(
                f"Kind: {motif.kind}"
            )

            print(
                f"Location: "
                f"{motif.file_path}:"
                f"{motif.start_line}-"
                f"{motif.end_line}"
            )

            print(
                f"Symbol: "
                f"{motif.symbol_name}"
            )

            print(
                f"Detail: "
                f"{motif.detail}"
            )

            print(
                f"Checked collections: "
                f"{discovery.checked_collections}"
            )

            print(
                f"Queue mutation: "
                f"{discovery.queue_collection}."
                f"{discovery.queue_method}"
                f"({discovery.item}) "
                f"at {discovery.queue_line}"
            )

            print(
                f"Tracking mutation: "
                f"{discovery.record_collection}."
                f"{discovery.record_method}"
                f"({discovery.item}) "
                f"at {discovery.record_line}"
            )