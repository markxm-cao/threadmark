from textwrap import dedent

from threadmark.chunking import CodeChunk
from threadmark.control_flow import extract_chunk_motifs


def test_detects_deduplication_guard():
    source = dedent(
        """\
        def collect_matches(match_ids, seen_ids):
            for match_id in match_ids:
                if match_id in seen_ids:
                    continue

                seen_ids.add(match_id)
        """
    )

    chunk = CodeChunk(
        file_path="example.py",
        start_line=1,
        end_line=6,
        content=source,
        symbol_name="collect_matches",
    )

    motifs = extract_chunk_motifs(
        chunk=chunk,
        source=source,
    )

    dedup_motifs = [
        motif
        for motif in motifs
        if motif.kind == "deduplication_guard"
    ]

    assert len(dedup_motifs) == 1

    motif = dedup_motifs[0]

    assert motif.file_path == "example.py"
    assert motif.symbol_name == "collect_matches"
    assert motif.start_line == 3
    assert motif.end_line == 6

    assert (
        motif.detail
        == "Skip match_id when it already exists in seen_ids; "
        "otherwise add match_id to seen_ids."
    )


def test_does_not_detect_deduplication_when_collection_differs():
    source = dedent(
        """\
        def collect_matches(match_ids, seen_ids, processed_ids):
            for match_id in match_ids:
                if match_id in seen_ids:
                    continue

                processed_ids.add(match_id)
        """
    )

    chunk = CodeChunk(
        file_path="example.py",
        start_line=1,
        end_line=6,
        content=source,
        symbol_name="collect_matches",
    )

    motifs = extract_chunk_motifs(
        chunk=chunk,
        source=source,
    )

    dedup_motifs = [
        motif
        for motif in motifs
        if motif.kind == "deduplication_guard"
    ]

    assert dedup_motifs == []