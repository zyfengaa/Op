"""Public symptoms and independent regression inputs; solutions are separate."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    title: str
    symptom: str
    hint: str
    variants: tuple


CASES = {
    "VA01": Case("尾部几个输出一直为零", "整除长度通过，N=5/block=4 时末项错。",
                 "列出所有被启动线程的全局索引。", (
        dict(a=[1, 2, 3, 4, 5], b=[10]*5, block=4),
        dict(a=[], b=[], block=4),
        dict(a=[1], b=[2], block=8),
        dict(a=list(range(17)), b=[1]*17, block=8))),
    "VA02": Case("同形输入换成切片后结果改变", "连续向量通过，步长 2 的视图失败。",
                 "shape 相同不等于物理地址相同。", (
        dict(a=[1, 99, 2, 99, 3, 99], b=[10, 20, 30], stride=2),
        dict(a=[4, 5, 6], b=[1, 1, 1], stride=1),
        dict(a=[7, 0, 0, 8], b=[2, 3], stride=3))),
    "VA03": Case("bias 加到了不该加的列", "方阵看起来像正常输出，非方阵逐列对照失败。",
                 "bias 是每行一份，还是每列一份？", (
        dict(x=[[1, 2, 3], [4, 5, 6]], bias=[10, 20, 30]),
        dict(x=[[0, 0, 0]], bias=[1, -1, 2]),
        dict(x=[[1, 2], [3, 4]], bias=[-2, 8]))),
    "VA04": Case("四元素加载升级后尾部丢失", "N=8 通过，N=9/10/11 失败。",
                 "完整向量片之外还剩几个元素？", (
        dict(a=list(range(10)), b=[1]*10),
        dict(a=[3], b=[4]),
        dict(a=list(range(8)), b=[2]*8),
        dict(a=list(range(11)), b=[3]*11))),
    "RD01": Case("大向量结果像只算了一截", "N<=线程数通过，输入增长后输出不再增长。",
                 "线程 t 是否只读取了一个元素？", (
        dict(x=list(range(1, 10)), threads=4),
        dict(x=[2, -3, 7], threads=8),
        dict(x=[1]*33, threads=8))),
    "RD02": Case("全负数的最大值变成零", "非 2 的幂长度更容易失败。",
                 "padding 必须不影响归约结果。", (
        dict(x=[-5, -2, -9]), dict(x=[-7]), dict(x=[-3, -8, -1, -4, -2]))),
    "RD03": Case("平均值随补齐宽度变化", "5 个相同数的 mean 反而小于这个数。",
                 "参与公式的有效元素数是哪一个？", (
        dict(x=[2]*5), dict(x=[1, 3, 5]), dict(x=[-4, 2, 8, 6]))),
    "RD04": Case("改线程数后最后几个数消失", "二分树在长度 8 正常，长度 5 不对。",
                 "每轮有奇数个局部值时，哪个值没被合并？", (
        dict(x=[1, 2, 3, 4, 5]), dict(x=[1]), dict(x=list(range(1, 8))),
        dict(x=[-1, 2, 3, 4, 5, 6, 7, 8]))),
    "SM01": Case("普通全负输入通过，极负输入 NaN", "[-5,-4,-3] 可通过，[-1000,-1001] 失败。",
                 "分别检查平移等价性与浮点下溢。", (
        dict(x=[-1000, -1001]), dict(x=[-5, -4, -3]), dict(x=[1000, 1001, 999]))),
    "LN01": Case("线程数一变，均值就变了", "分给各线程的元素数量不一样时失败。",
                 "局部 mean 能否不带 count 直接平均？", (
        dict(x=[1, 2, 3, 4, 100], threads=4),
        dict(x=[1, 2, 3, 4], threads=2),
        dict(x=[3, -2, 9, 7, 12, 6, 1], threads=3))),
    "GM01": Case("右下输出 tile 少了一项", "输出越界线程提前退出后，合法 C[2,2] 也错。",
                 "线程的输出坐标与它负责的 load 是不同职责。", (
        dict(a_row=[1, 2, 3], b_col=[7, 9, 11]),
        dict(a_row=[3, -4, 5], b_col=[2, 6, 8]),
        dict(a_row=[1, 0, 1], b_col=[3, 4, 5]))),
    "AT01": Case("Attention 输出跑到 V 范围外", "新块最大值升高后，概率加权值异常。",
                 "m 改变时，需要换基准的状态有几个？", (
        dict(scores=[0, 2], values=[10, 20]),
        dict(scores=[3, 2, 1], values=[1, 7, 3]),
        dict(scores=[-2, 0, 4], values=[-1, 3, 6]))),
}
