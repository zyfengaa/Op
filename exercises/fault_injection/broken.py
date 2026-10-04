"""INTENTIONALLY WRONG kernels/dataflow models. No GPU hangs or unsafe memory access."""
import math


def va01(p):
    n, block = len(p["a"]), p["block"]
    out = [0.0] * n
    for i in range((n // block) * block):
        out[i] = p["a"][i] + p["b"][i]
    return out


def va02(p):
    return [p["a"][i] + p["b"][i] for i in range(len(p["b"]))]


def va03(p):
    return [[value + p["bias"][r] for value in row] for r, row in enumerate(p["x"])]


def va04(p):
    out = [0.0] * len(p["a"])
    for start in range(0, len(out) - 3, 4):
        for j in range(4):
            out[start + j] = p["a"][start + j] + p["b"][start + j]
    return out


def rd01(p):
    return sum(p["x"][:p["threads"]])


def rd02(p):
    width = 1 << (len(p["x"]) - 1).bit_length()
    return max(p["x"] + [0.0] * (width - len(p["x"])))


def rd03(p):
    padded = 1 << (len(p["x"]) - 1).bit_length()
    return sum(p["x"]) / padded


def rd04(p):
    partial = p["x"][:]
    step = len(partial) // 2
    while step:
        for i in range(step):
            partial[i] += partial[i + step]
        step //= 2
    return partial[0]


def sm01(p):
    m = max([0.0] + p["x"])
    e = [math.exp(x - m) for x in p["x"]]
    total = sum(e)
    # Return NaN instead of triggering division by zero, to expose the symptom.
    return [x / total if total else math.nan for x in e]


def ln01(p):
    lanes = [p["x"][t::p["threads"]] for t in range(p["threads"])]
    active = [sum(lane) / len(lane) for lane in lanes if lane]
    return sum(active) / len(active)


def gm01(p):
    # Safe model of an output-tail early return. A[2,1] was assigned to lane (0,1),
    # whose output column is 3 (outside N=3). That lane exits before the load.
    a_tile_row = [p["a_row"][0], 0.0, p["a_row"][2]]
    return sum(a * b for a, b in zip(a_tile_row, p["b_col"]))


def at01(p):
    m, total, acc = -math.inf, 0.0, 0.0
    for score, value in zip(p["scores"], p["values"]):
        new_m = max(m, score)
        alpha = math.exp(m - new_m)
        weight = math.exp(score - new_m)
        total = alpha * total + weight
        acc = acc + weight * value
        m = new_m
    return acc / total


VARIANTS = {key.upper(): value for key, value in list(globals().items())
            if key[:2] in ("va", "rd", "sm", "ln", "gm", "at") and callable(value)}


def solve(case_id, payload):
    return VARIANTS[case_id](payload)
