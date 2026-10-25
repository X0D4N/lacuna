"""BPE tokenizer tests: machinery only, with a toy vocab.

Real-model token ids are verified separately against llama-server's
/tokenize endpoint (manual check, 2026-10-25: exact match on three
prompts for Qwen2.5-0.5B-Instruct). These tests cover the merge loop,
the byte mapping and the encode/decode round-trip without a model file.
"""

from lacuna.bpe import BPETokenizer, bytes_to_unicode


def _toy():
    # Vocab covers single bytes a/b/space plus the merged forms.
    tokens = ["a", "b", "Ġ", "ab", "Ġa"]
    merges = ["a b", "Ġ a"]
    return BPETokenizer(tokens, merges)


def test_byte_mapping_covers_all_bytes():
    m = bytes_to_unicode()
    assert len(m) == 256
    assert len(set(m.values())) == 256
    # Printable ASCII maps to itself.
    assert m[ord("A")] == "A"
    assert m[ord(" ")] == "Ġ"


def test_merge_applies_in_rank_order():
    tok = _toy()
    assert tok.encode("ab") == [3]  # "a"+"b" -> "ab"


def test_space_merge():
    tok = _toy()
    assert tok.encode(" a") == [4]  # "Ġ"+"a" -> "Ġa"


def test_decode_round_trip():
    tok = _toy()
    assert tok.decode(tok.encode("ab a")) == "ab a"


def test_unknown_piece_raises_loud():
    tok = BPETokenizer(["a"], [])
    try:
        tok.encode("b")
    except ValueError:
        return
    raise AssertionError("expected ValueError for a byte outside the vocab")
