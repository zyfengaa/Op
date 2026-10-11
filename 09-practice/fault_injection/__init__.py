"""Deterministic, CPU-only debugging exercises. Broken variants are intentional."""
# 包说明：这组练习是纯 CPU、纯 Python 的数据流模拟，不依赖 GPU 或第三方库，
# 因此任何 Python 环境都能直接跑判题。
#
# __init__ 故意保持"空"（只有模块 docstring）：`import fault_injection` 本身不导入
# broken / solutions / student 等子模块，子模块按需显式导入。这样一次误导入
# 不会把参考答案或错误实现带进内存，也不会在导入期产生任何副作用。
