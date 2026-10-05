# 性能实验记录模板

复制本文件，为每一个重要变更创建独立记录。

## Workload

- Operator:
- Model / call site:
- Shape:
- Dtype / accumulation dtype:
- Layout / stride:
- Device:
- Driver / toolkit / compiler:
- Git commit:
- Build flags:

## Baseline

- Correctness result:
- Warmup / iterations:
- Kernel latency (P50/P90/mean):
- End-to-end latency:
- FLOPs:
- Estimated bytes:
- Profiler report:

## Observation

只记录可以从测量复现的事实，不要把解释写成事实。

## Hypothesis

说明哪个机制造成当前瓶颈，以及如果判断正确，修改后应观察到什么变化。

## One change

记录只改动的主要变量、具体参数和代码位置。

## Result

- Correctness:
- Kernel latency:
- E2E latency:
- Relevant metrics:
- Unexpected changes:

## Decision

- Hypothesis: supported / rejected / unclear
- Keep or revert:
- Applicability by shape:
- Follow-up:
