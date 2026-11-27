"""Tree speculation tests: exact by construction.

The proof: accepted tokens must EQUAL plain target-only greedy decoding for
the same seed (the deterministic mock target makes this checkable).
"""


from lacuna.engine import _mock_target_step
from lacuna.speculate import speculative_tree_generate


def _greedy(seed, n):
    out, p = [], list(seed)
    for _ in range(n):
        t = _mock_target_step(p)
        out.append(t)
        p = p + [t]
    return out


def _draft_tree(prefix, depth, branching):
    """Tree containing the true greedy chain plus distractor branches."""
    tree = {}
    # true greedy chain up to depth
    p, chain = list(prefix), []
    for _ in range(depth):
        t = _mock_target_step(p)
        chain.append(t)
        p = p + [t]
    for i in range(1, depth + 1):
        tree[tuple(chain[:i])] = 1.0
    # distractors: wrong tokens branching off each chain node
    for i in range(depth):
        node = tuple(chain[:i])
        for b in range(1, branching):
            wrong = (_mock_target_step(list(prefix) + list(node)) + b * 7919) % 4096
            if wrong == chain[i]:
                wrong = (wrong + 1) % 4096
            tree[node + (wrong,)] = 0.1
    return tree


def _verify_tree(prefix, tree):
    """ONE scan of the tree building a children index, then a greedy walk:
    at each node take the child the target itself would emit. Returns the
    longest prefix path the target agrees with."""
    children = {}
    for path in tree:
        for i in range(len(path)):
            children.setdefault(path[:i], set()).add(path[i])
    out, node = [], ()
    while True:
        want = _mock_target_step(list(prefix) + out)
        if want in children.get(node, ()):
            out.append(want)
            node = node + (want,)
        else:
            break
    return tuple(out)


def test_tree_matches_greedy_decoding():
    seed = [11, 22]
    res = speculative_tree_generate(
        _draft_tree, _verify_tree, max_tokens=10, max_depth=2, branching=2,
        prefix=seed,
    )
    # THE exactness proof: identical to target-only greedy decoding.
    assert res["tokens"] == _greedy(seed, 10)
    assert res["mean_acceptance_length"] >= 1.0
    # full acceptance every round -> depth adapted upward from 2
    assert res["final_depth"] > 2
    assert res["rounds"] >= 1


def test_tree_single_arg_draft_still_works():
    """A draft_tree_fn(prefix) with no depth/branching args must also work."""
    seed = [5]

    def simple_draft(prefix):
        return _draft_tree(prefix, 3, 2)

    res = speculative_tree_generate(
        simple_draft, _verify_tree, max_tokens=8, max_depth=3, prefix=seed
    )
    assert res["tokens"] == _greedy(seed, 8)
    assert res["mean_acceptance_length"] >= 1.0


def test_tree_empty_tree_terminates():
    res = speculative_tree_generate(
        lambda prefix: {}, _verify_tree, max_tokens=8, prefix=[1]
    )
    assert res["tokens"] == []
    assert res["rounds"] == 0
