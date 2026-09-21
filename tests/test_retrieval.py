from threadmark.retrieval import (
    split_identifier,
    tokenize_code_aware,
)


def test_split_identifier_handles_camel_case():
    assert split_identifier("collectMatchIds") == [
        "collect",
        "match",
        "ids",
    ]


def test_split_identifier_handles_snake_case():
    assert split_identifier("match_id_cache") == [
        "match",
        "id",
        "cache",
    ]


def test_tokenize_code_aware_preserves_and_splits_identifiers():
    tokens = tokenize_code_aware(
        "collectMatchIds match_id 42"
    )

    assert tokens == [
        "collectmatchids",
        "collect",
        "match",
        "ids",
        "match_id",
        "match",
        "id",
        "42",
    ]