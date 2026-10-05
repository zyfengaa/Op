"""Fault lab CLI. No GPU or third-party package required."""
import argparse
import copy
import math
import sys

from fault_injection import broken, solutions, student
from fault_injection.cases import CASES

# Hand-computed anchors: selftest must not only compare a solution with itself.
GOLDEN = {
    "VA01": [11, 12, 13, 14, 15], "VA02": [11, 22, 33],
    "VA03": [[11, 22, 33], [14, 25, 36]], "VA04": list(range(1, 11)),
    "RD01": 45, "RD02": -2, "RD03": 2, "RD04": 15,
    "SM01": [0.7310585786300049, 0.2689414213699951],
    "LN01": 22, "GM01": 58, "AT01": 18.807970779778824,
}


def close(actual, expected):
    if isinstance(expected, (list, tuple)):
        return isinstance(actual, (list, tuple)) and len(actual) == len(expected) and all(
            close(a, e) for a, e in zip(actual, expected))
    return isinstance(actual, (float, int)) and math.isfinite(actual) and math.isclose(
        actual, expected, rel_tol=1e-9, abs_tol=1e-9)


def evaluate(case_id, impl, verbose=True):
    case = CASES[case_id]
    ok = True
    for index, payload in enumerate(case.variants):
        expected = solutions.solve(case_id, copy.deepcopy(payload))
        try:
            actual = impl.solve(case_id, copy.deepcopy(payload))
            passed = close(actual, expected)
        except Exception as error:
            actual, passed = f"{type(error).__name__}: {error}", False
        if verbose:
            print(f"  case {index + 1}: {'PASS' if passed else 'FAIL'}")
            if not passed:
                print(f"    input={payload}\n    actual={actual}\n    expected={expected}")
        ok = ok and passed
    return ok


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "show", "grade", "selftest"])
    parser.add_argument("case", nargs="?", default="all")
    parser.add_argument("--impl", choices=["student", "broken", "solution"], default="student")
    parser.add_argument("--hint", action="store_true")
    args = parser.parse_args()
    if args.case != "all" and args.case not in CASES:
        parser.error("unknown case id")
    selected = list(CASES) if args.case == "all" else [args.case]
    if args.command == "selftest":
        # Each defect must be exposed by at least one regression input, while the
        # reference passes the same contract. Expected failures are not CI failures.
        all_ok = True
        for key in selected:
            exposed = not evaluate(key, broken, False)
            correct = close(solutions.solve(key, copy.deepcopy(CASES[key].variants[0])), GOLDEN[key])
            correct = correct and evaluate(key, solutions, False)
            print(f"{key}: defect_exposed={exposed}, solution_passed={correct}")
            all_ok &= exposed and correct
        return 0 if all_ok else 1
    all_ok = True
    for key in selected:
        case = CASES[key]
        print(f"{key} | {case.title}")
        if args.command == "show":
            print(f"  症状：{case.symptom}\n  复现输入：{case.variants[0]}")
            print(f"  待审代码：09-practice/fault_injection/broken.py::{key.lower()}")
            if args.hint:
                print(f"  提示：{case.hint}")
        elif args.command == "grade":
            impl = {"student": student, "broken": broken, "solution": solutions}[args.impl]
            all_ok &= evaluate(key, impl)
    if args.command == "grade":
        print("ALL PASS" if all_ok else "NOT YET: fix student.py and rerun")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
