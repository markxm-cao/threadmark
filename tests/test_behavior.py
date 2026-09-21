import ast

from threadmark.behavior import classify_condition


def parse_expression(source: str):
    return ast.parse(
        source,
        mode="eval",
    ).body


def test_membership_condition():
    node = parse_expression("item in seen")

    assert classify_condition(node) == "membership_check"


def test_non_membership_condition():
    node = parse_expression("item not in seen")

    assert classify_condition(node) == "non_membership_check"


def test_periodic_modulo_condition():
    node = parse_expression("counter % interval == 0")

    assert classify_condition(node) == "periodic_modulo_check"


def test_negated_predicate():
    node = parse_expression("not ready")

    assert classify_condition(node) == "negated_predicate"