"""Deterministic quiz grading; no third-party dependencies."""
# 阅读题判分器：答案键与提交都来自 JSON 文件，纯标准库、输出确定，便于 CI 直接看返回码。
#
# 判分口径：
#   * 期望答案的顺序冻结在 solutions.json，提交按同一顺序逐题比较；
#   * 比较前统一 strip() 并转大写，避免大小写/空白造成的假失败；
#   * 提交缺失、类型不对或长度不足一律当作"没答"，而不是报错——判分器不应该
#     因为一份格式不全的答卷就崩掉；
#   * 只有满分才返回 0，其余返回 1。
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # 默认读取同目录的 student.json；--answers 可指向别处的答卷。
    parser.add_argument("--answers", type=Path, default=Path(__file__).with_name("student.json"))
    args = parser.parse_args()
    # 答案键是仓库自带的，读取失败属于环境错误，直接抛出即可。
    expected = json.loads(Path(__file__).with_name("solutions.json").read_text(encoding="utf-8"))
    try:
        submitted = json.loads(args.answers.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        # 文件缺失（OSError）或 JSON 语法错误（ValueError）都走这里：转成 parser.error
        # 给出清晰信息与退出码 2，而不是抛一长串 traceback。
        parser.error(str(error))
    score = 0
    for task, answers in expected.items():
        # 层层兜底：提交整体不是 dict、或该题答案不是 list，都退化为空答案。
        values = submitted.get(task, []) if isinstance(submitted, dict) else []
        if not isinstance(values, list):
            values = []
        for index, correct in enumerate(answers):
            # 缺项按空串处理；比较前 strip + 转大写，让答案格式宽容一些。
            actual = str(values[index]).strip().upper() if index < len(values) else ""
            passed = actual == correct
            print(f"{task}.{index+1}: {'PASS' if passed else 'TRY AGAIN'}")
            # True 参与算术即 1，这里用它做简短计分。
            score += passed
    print(f"score: {score}/15")
    # 总分 15 与题库题数绑定；若增删题目，这里的分母要同步改。
    # 只有满分才返回 0，任何失分都返回 1。
    return 0 if score == 15 else 1


if __name__ == "__main__":
    raise SystemExit(main())
