"""Instructor implementations. Read after attempting student.py."""
import math


def solve(case_id, p):
    if case_id in ("VA01", "VA04"):
        return [a + b for a, b in zip(p["a"], p["b"])]
    if case_id == "VA02":
        return [p["a"][i * p["stride"]] + p["b"][i] for i in range(len(p["b"]))]
    if case_id == "VA03":
        return [[value + p["bias"][c] for c, value in enumerate(row)] for row in p["x"]]
    if case_id == "RD01" or case_id == "RD04":
        return math.fsum(p["x"])
    if case_id == "RD02":
        return max(p["x"])
    if case_id == "RD03" or case_id == "LN01":
        return math.fsum(p["x"]) / len(p["x"])
    if case_id == "SM01":
        m = max(p["x"])
        e = [math.exp(x - m) for x in p["x"]]
        return [x / math.fsum(e) for x in e]
    if case_id == "GM01":
        # Output C[2,2] still needs the A[2,1] load owned by lane (0,1).
        return sum(a * b for a, b in zip(p["a_row"], p["b_col"]))
    if case_id == "AT01":
        m = max(p["scores"])
        e = [math.exp(x - m) for x in p["scores"]]
        return sum(w * v for w, v in zip(e, p["values"])) / sum(e)
    raise KeyError(case_id)
