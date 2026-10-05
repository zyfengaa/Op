"""Edit this file to repair the exercises; keep broken.py as the original evidence.

Each TODO currently delegates to the intentionally broken implementation.
Use: python 09-practice/check.py grade VA01
"""
from . import broken


def solve(case_id, payload):
    # TODO VA01: repair launch coverage; support empty input and tails.
    # TODO VA02: respect the logical view's element stride.
    # TODO VA03: follow per-column broadcasting, including non-square shapes.
    # TODO VA04: cover the vectorized remainder without an out-of-bounds read.
    # TODO RD01: let threads cover the whole input, not just its first segment.
    # TODO RD02: choose the correct neutral value for max padding.
    # TODO RD03: separate physical padding width from mathematical element count.
    # TODO RD04: support arbitrary positive input lengths.
    # TODO SM01: preserve numerical stability for very negative finite logits.
    # TODO LN01: merge unequal-size statistics with their counts.
    # TODO GM01: distinguish output validity from cooperative-load responsibility.
    # TODO AT01: rescale both online accumulators when the maximum changes.
    #
    # Add branches here, e.g. if case_id == "VA01": ...; do not import solutions.
    return broken.solve(case_id, payload)
