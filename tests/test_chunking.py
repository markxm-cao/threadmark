import pytest

from threadmark.chunking import (
    chunk_lines_fixed,
    chunk_python_file_ast,
)


def test_fixed_chunking_uses_overlap():
    lines = [
        (1, "line 1"),
        (2, "line 2"),
        (3, "line 3"),
        (4, "line 4"),
        (5, "line 5"),
    ]

    chunks = chunk_lines_fixed(
        file_path="example.py",
        lines=lines,
        chunk_size=3,
        overlap=1,
    )

    assert len(chunks) == 2

    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 3
    assert chunks[0].content == "line 1\nline 2\nline 3"

    assert chunks[1].start_line == 3
    assert chunks[1].end_line == 5
    assert chunks[1].content == "line 3\nline 4\nline 5"


def test_fixed_chunking_rejects_invalid_overlap():
    with pytest.raises(
        ValueError,
        match="overlap must be smaller than chunk_size",
    ):
        chunk_lines_fixed(
            file_path="example.py",
            lines=[(1, "test")],
            chunk_size=3,
            overlap=3,
        )


def test_ast_chunking_preserves_top_level_symbols():
    lines = [
        (1, "def foo():"),
        (2, "    return 1"),
        (3, ""),
        (4, "class Bar:"),
        (5, "    def baz(self):"),
        (6, "        return 2"),
    ]

    chunks = chunk_python_file_ast(
        file_path="example.py",
        lines=lines,
    )

    assert len(chunks) == 2

    assert chunks[0].symbol_name == "foo"
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 2

    assert chunks[1].symbol_name == "class Bar"
    assert chunks[1].start_line == 4
    assert chunks[1].end_line == 6