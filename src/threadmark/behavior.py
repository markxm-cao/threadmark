import ast
from dataclasses import dataclass
from threadmark.behavior_categories import operation_to_behavior
from threadmark.repository import (
    find_source_files,
    read_source_file,
)


@dataclass
class BehaviorFact:
    file_path: str
    symbol_name: str
    line: int
    kind: str
    operation: str
    behavior: str | None
    detail: str
    condition: str | None = None
    
    
class BehaviorExtractor(ast.NodeVisitor):
    def __init__(
        self,
        file_path: str,
        symbol_name: str,
    ):
        self.file_path = file_path
        self.symbol_name = symbol_name
        self.facts: list[BehaviorFact] = []
        self.condition_stack: list[str] = []

    def current_condition(self) -> str | None:
        if not self.condition_stack:
            return None

        return " AND ".join(self.condition_stack)

    def add_fact(
        self,
        node: ast.AST,
        kind: str,
        operation: str,
        detail: str,
    ) -> None:
        
        behavior = operation_to_behavior(operation)
        
        self.facts.append(
            BehaviorFact(
                file_path=self.file_path,
                symbol_name=self.symbol_name,
                line=node.lineno,
                kind=kind,
                operation=operation,
                behavior=behavior,
                detail=detail,
                condition=self.current_condition(),
            )
        )
        
    def visit_If(self, node: ast.If) -> None:
        condition = ast.unparse(node.test)

        if isinstance(node.test, ast.BoolOp):
            if isinstance(node.test.op, ast.And):
                operation = "boolean_and"
            else:
                operation = "boolean_or"
        else:
            operation = classify_condition(node.test)

        self.add_fact(
            node,
            kind="condition",
            operation=operation,
            detail=condition,
        )
        
        if isinstance(node.test, ast.BoolOp):
            components = extract_condition_components(
                node.test
            )

            for component_operation, component_detail in components:
                self.add_fact(
                    node,
                    kind="condition_component",
                    operation=component_operation,
                    detail=component_detail,
                )
                
        self.condition_stack.append(condition)

        for statement in node.body:
            self.visit(statement)

        self.condition_stack.pop()

        if node.orelse:
            else_condition = f"NOT ({condition})"
            self.condition_stack.append(else_condition)

            for statement in node.orelse:
                self.visit(statement)

            self.condition_stack.pop()
    
    def visit_Continue(self, node: ast.Continue) -> None:
        self.add_fact(
            node,
            kind="control",
            operation="continue",
            detail="continue",
        )


    def visit_Break(self, node: ast.Break) -> None:
        self.add_fact(
            node,
            kind="control",
            operation="break",
            detail="break",
        )

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is None:
            detail = "return"
        else:
            detail = f"return {ast.unparse(node.value)}"

        self.add_fact(
            node,
            kind="control",
            operation="return",
            detail=detail,
        )

    def visit_Raise(self, node: ast.Raise) -> None:
        if node.exc is None:
            detail = "raise"
        else:
            detail = f"raise {ast.unparse(node.exc)}"

        self.add_fact(
            node,
            kind="control",
            operation="raise",
            detail=detail,
        )
        
    def visit_Call(self, node: ast.Call) -> None:
        detail = ast.unparse(node)

        mutation_operations = {
            "add": "collection_add",
            "append": "collection_append",
            "extend": "collection_extend",
            "update": "collection_update",
            "remove": "collection_remove",
            "discard": "collection_discard",
            "pop": "collection_pop",
            "clear": "collection_clear",
        }

        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr

            if method_name in mutation_operations:
                kind = "mutation"
                operation = mutation_operations[method_name]
            else:
                kind = "call"
                operation = "function_call"
        else:
            kind = "call"
            operation = "function_call"

        self.add_fact(
            node,
            kind=kind,
            operation=operation,
            detail=detail,
        )

        self.generic_visit(node)
        
    def visit_Assign(self, node: ast.Assign) -> None:
        self.add_fact(
            node,
            kind="assignment",
            operation="assignment",
            detail=ast.unparse(node),
        )

        self.generic_visit(node)
        
def extract_function_behavior(
    file_path: str,
    lines: list[tuple[int, str]],
    function_name: str,
) -> list[BehaviorFact]:
    """Extract structural behavior facts from a Python function."""

    source = "\n".join(
        line
        for _, line in lines
    )

    tree = ast.parse(source)

    for node in tree.body:
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            if node.name == function_name:
                extractor = BehaviorExtractor(
                    file_path=file_path,
                    symbol_name=function_name,
                )

                for statement in node.body:
                    extractor.visit(statement)

                return extractor.facts

    return []


def extract_file_behaviors(
    file_path: str,
    lines: list[tuple[int, str]],
) -> list[BehaviorFact]:
    """Extract behavior facts from functions and methods in a Python file."""

    source = "\n".join(
        line
        for _, line in lines
    )

    tree = ast.parse(source)

    all_facts = []

    for node in tree.body:
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            extractor = BehaviorExtractor(
                file_path=file_path,
                symbol_name=node.name,
            )

            for statement in node.body:
                extractor.visit(statement)

            all_facts.extend(extractor.facts)

        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(
                    child,
                    (ast.FunctionDef, ast.AsyncFunctionDef),
                ):
                    extractor = BehaviorExtractor(
                        file_path=file_path,
                        symbol_name=f"{node.name}.{child.name}",
                    )

                    for statement in child.body:
                        extractor.visit(statement)

                    all_facts.extend(extractor.facts)

    return all_facts


def extract_repository_behaviors(
    repo_path: str,
) -> list[BehaviorFact]:
    """Extract behavior facts from all Python source files in a repository."""

    all_facts = []

    for file_path in find_source_files(repo_path):
        if not file_path.endswith(".py"):
            continue

        lines = read_source_file(
            repo_path,
            file_path,
        )

        try:
            facts = extract_file_behaviors(
                file_path,
                lines,
            )
        except SyntaxError:
            continue

        all_facts.extend(facts)

    return all_facts


def classify_condition(node: ast.expr) -> str:
    """Normalize common conditional expressions into behavioral operations."""

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return "negated_predicate"

    if isinstance(node, ast.Compare):
        if len(node.ops) == 1:
            operator = node.ops[0]

            if isinstance(operator, ast.In):
                return "membership_check"

            if isinstance(operator, ast.NotIn):
                return "non_membership_check"

            if isinstance(operator, ast.Eq):
                if (
                    isinstance(node.left, ast.BinOp)
                    and isinstance(node.left.op, ast.Mod)
                ):
                    return "periodic_modulo_check"

                return "equality_check"

            if isinstance(operator, ast.NotEq):
                return "inequality_check"

    return "condition"


def extract_condition_components(
    node: ast.expr,
) -> list[tuple[str, str]]:
    """Extract normalized atomic operations from a condition."""

    components = []

    if isinstance(node, ast.BoolOp):
        for value in node.values:
            components.extend(
                extract_condition_components(value)
            )

        return components

    operation = classify_condition(node)

    components.append(
        (
            operation,
            ast.unparse(node),
        )
    )

    return components




if __name__ == "__main__":
    from threadmark.repository import clone_repository

    repo_path = clone_repository(
        "https://github.com/nartnek/RiftPredict",
        "data/repos",
    )

    facts = extract_repository_behaviors(repo_path)

    print(f"Behavior facts: {len(facts)}")