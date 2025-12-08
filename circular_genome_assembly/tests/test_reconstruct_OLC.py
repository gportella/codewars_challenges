import pytest

from  genome_assembly_olc import reconstruct_genome
from test_solutions import test_solutions


def _is_rotation(a: str, b: str) -> bool:
    return len(a) == len(b) and b in (a + a)




@pytest.mark.parametrize(
    "reads,expected,has_errors",
    [
        (case["reads"], case["solution"], case.get("has_errors", False))
        for case in test_solutions
    ],
)
def test_reconstruct_genome_matches_known_solutions(reads, expected, has_errors):
    result = reconstruct_genome(reads, has_errors=has_errors)
    assert len(result) == len(expected)
    assert _is_rotation(result, expected)
