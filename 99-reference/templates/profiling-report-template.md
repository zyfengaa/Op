# Profiling 报告模板

## 目标

- Kernel / operator:
- 为什么选择它（端到端占比）:
- 设备 / 软件版本:
- 输入 shape / dtype / layout:
- 测量场景：isolated kernel / graph / end-to-end

## 计时可信度

- release build 与 line info:
- warmup 和采样次数:
- profiler replay/cache/clock 设置:
- 与普通运行时间是否一致:
- 是否有其他进程或 stream 干扰:

## 数据

- Kernel duration:
- SM / compute pipeline:
- HBM / L2 / shared traffic:
- occupancy / registers / shared memory:
- warp stall / issue information:
- 报告文件路径:

## 推理链

- 观察到的事实:
- 初始瓶颈分类:
- 根因假设:
- 可证伪预测:
- 下一次只改的变量:

## 结论

- 假设成立/不成立/证据不足:
- 绝对时间变化:
- E2E 变化:
- 适用 shape:
- 已知风险:
