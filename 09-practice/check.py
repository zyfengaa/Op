"""Fault lab CLI. No GPU or third-party package required."""
# 判题 CLI 的入口，四种命令：
#   list     列出题库（题号 + 标题）
#   show     只给症状、复现输入和 broken.py 的函数名；加 --hint 才额外给提示
#   grade    把某个实现跑一遍全部回归输入并逐条报 PASS/FAIL
#   selftest 仓库自检：每个故意做坏的版本都要被至少一个输入击中，参考解必须全过
#
# 判题的关键设计是"双向隔离"：学生实现和参考解各自拿到 payload 的深拷贝，
# 任何一方在原地改写输入都不会污染另一方，比较结果才可信。
import argparse
import copy
import math
import sys

# 导入面刻意划清：判题器需要 broken（自检）、solutions（算期望值）和 student（默认被评者）。
# CASES 只提供题面与输入，绝不从 cases.py 读取解法；否则题面和答案会落在同一个文件里。
from fault_injection import broken, solutions, student
from fault_injection.cases import CASES

# Hand-computed anchors: selftest must not only compare a solution with itself.
# 手算锚点：若只用"参考解 vs 参考解"自检，两边一起写错也会被判通过。
# 这些常数从题面输入手工推导，只锚定每题的第一组输入（variants[0]）；
# 其余输入交给"参考解 vs 被评实现"的差分比较，新增回归输入时不必再手工维护常数。
GOLDEN = {
    "VA01": [11, 12, 13, 14, 15], "VA02": [11, 22, 33],
    "VA03": [[11, 22, 33], [14, 25, 36]], "VA04": list(range(1, 11)),
    "RD01": 45, "RD02": -2, "RD03": 2, "RD04": 15,
    "SM01": [0.7310585786300049, 0.2689414213699951],
    "LN01": 22, "GM01": 58, "AT01": 18.807970779778824,
}


# 递归比较。结构由 expected 的形状决定：标量题（RD/SM/AT）比单个浮点数，
# VA03/VA04 这类逐层下钻，比较嵌套的 list/tuple。
def close(actual, expected):
    if isinstance(expected, (list, tuple)):
        # 长度先对齐再逐元素递归；长度不等直接判否，不让 zip 静默截断掩盖差异。
        return isinstance(actual, (list, tuple)) and len(actual) == len(expected) and all(
            close(a, e) for a, e in zip(actual, expected))
    # isfinite 必须显式判：NaN 与任何值的比较都是 False，只靠 isclose 时，
    # 一个返回 nan 的实现会因为"不相等"而"碰巧"判错——我们要的是主动识别而非碰巧。
    # rel_tol 管大数、abs_tol 管零点附近；两者都给是因为 rel_tol 在 0 附近会退化。
    return isinstance(actual, (float, int)) and math.isfinite(actual) and math.isclose(
        actual, expected, rel_tol=1e-9, abs_tol=1e-9)


# 逐 variant 跑被评实现，并与参考解对拍。
# 任一组输入没过就返回 False：修复只对第一组输入生效不算完成
# （例如 VA01 还要覆盖空输入、单元素和不能整除的尾部）。
def evaluate(case_id, impl, verbose=True):
    case = CASES[case_id]
    ok = True
    for index, payload in enumerate(case.variants):
        # 期望值来自参考解而非手算锚点：GOLDEN 只覆盖第一组输入，其余输入需要现算。
        expected = solutions.solve(case_id, copy.deepcopy(payload))
        try:
            # 深拷贝是必需的：题面和实现都可能原地改 payload；不拷贝会让 expected
            # 与 actual 共享同一份可变对象，先算的一方会篡改后算的一方的输入。
            actual = impl.solve(case_id, copy.deepcopy(payload))
            passed = close(actual, expected)
        except Exception as error:
            # 让抛异常的实现判 FAIL，而不是把整个判题器一起带崩；
            # 错误类型与信息原样留下，便于区分"没实现"和"实现崩了"。
            actual, passed = f"{type(error).__name__}: {error}", False
        if verbose:
            print(f"  case {index + 1}: {'PASS' if passed else 'FAIL'}")
            if not passed:
                print(f"    input={payload}\n    actual={actual}\n    expected={expected}")
        ok = ok and passed
    return ok


def main():
    if hasattr(sys.stdout, "reconfigure"):
        # Windows 控制台默认不是 UTF-8，症状串含中文，不重设编码会直接抛 WriteError。
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    # 四个子命令串起"读题 → 定位实现 → 判分 → 仓库自检"的完整流程。
    parser.add_argument("command", choices=["list", "show", "grade", "selftest"])
    parser.add_argument("case", nargs="?", default="all")
    # 默认判学生实现；--impl broken/solution 用于自检与演示，不改变判题逻辑。
    parser.add_argument("--impl", choices=["student", "broken", "solution"], default="student")
    # 提示默认关闭：题目刻意只给症状，--hint 是主动求助时才展开的。
    parser.add_argument("--hint", action="store_true")
    args = parser.parse_args()
    # 提前挡掉拼错的题号，避免后面在字典查找处抛 KeyError。
    if args.case != "all" and args.case not in CASES:
        parser.error("unknown case id")
    selected = list(CASES) if args.case == "all" else [args.case]
    if args.command == "selftest":
        # Each defect must be exposed by at least one regression input, while the
        # reference passes the same contract. Expected failures are not CI failures.
        # 两个方向都要成立：故意错误版本必须失败（这道题才有区分度），
        # 手算锚点与参考解必须全过。学生尚未填的 TODO 不计入仓库回归失败。
        all_ok = True
        for key in selected:
            # 故意错误版本必须被至少一组输入判错，否则该题无法暴露缺陷。
            exposed = not evaluate(key, broken, False)
            # 参考解先对手算锚点，再对全部回归输入。
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
            # show 只给症状、复现输入和函数位置，不打印 broken 的函数体：
            # 让学习者在动代码之前先自己预测错在哪、哪些输入会掩盖错误。
            print(f"  症状：{case.symptom}\n  复现输入：{case.variants[0]}")
            print(f"  待审代码：09-practice/fault_injection/broken.py::{key.lower()}")
            if args.hint:
                print(f"  提示：{case.hint}")
        elif args.command == "grade":
            # 同一套输入换不同的被评实现，判题口径完全一致。
            impl = {"student": student, "broken": broken, "solution": solutions}[args.impl]
            all_ok &= evaluate(key, impl)
    if args.command == "grade":
        print("ALL PASS" if all_ok else "NOT YET: fix student.py and rerun")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
