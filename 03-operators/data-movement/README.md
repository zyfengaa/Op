# 数据搬运算子

不改数值，只改布局或位置。看起来简单，实际是访存模式问题最集中的一类。

## 包含什么

| 算子 | 做什么 | 难点 |
|---|---|---|
| transpose | 交换两个维度 | 读合并则写不合并（或反之） |
| pad | 补边界 | 边界判断 + 对齐 |
| gather | 按索引取值 | 索引随机，访存不可预测 |
| scatter | 按索引写值 | 写入冲突 |
| reshape / view | 改形状 | 连续时是零拷贝，否则要真搬 |
| permute | 换维度顺序 | 只改 stride，但要真算时仍是 transpose |
| cat / split | 拼接 / 切分 | 多输入的边界与对齐 |
| slice | 切片 | stride 不为 1，破坏合并假设 |

## 核心矛盾

**读和写的顺序不能同时最优。**

transpose 是最典型的例子：读的时候按行（合并），写的时候按列（不合并）；或者反过来。总有一边要付代价。

~~~text
选项 A：读合并、写不合并
选项 B：读不合并、写合并
选项 C：shared memory 中转，读写都合并，代价是一次片上往返 + 一次屏障
~~~

**通常选 C。** 全局访存从 12.5% 利用率升到接近 100%，代价是片上往返，而片上带宽远高于全局。

## shared memory 重排

~~~text
1. 按合并方式把全局数据读入 shared tile
2. __syncthreads()
3. 按目标顺序从 shared tile 写出到全局
~~~

**注意 bank conflict**：tile 的列跨度是 32 的整数倍时，同一 warp 会全部撞到同一个 bank。

~~~text
tile[32][32]  → 按列访问时 bank = t mod 32 → 全撞 bank 0
tile[32][33]  → bank = t mod 33 → 分散，冲突消除
~~~

**padding 一列**是标准做法。多出来的那一列不参与计算，只改变地址。

## gather / scatter 的特殊性

这两个算子的访存模式**由索引数据决定**，编译期不可知：

- 索引集中 → 访存相对合并
- 索引分散 → 每个线程一次独立访存，带宽利用率极低

**没有通用的优化手段**。可行方向：

- 如果索引有局部性，按索引排序或分桶后再处理
- 如果是 embedding 类查表，考虑把表的行对齐到 cache line
- 如果 GPU 资源允许，用纹理内存/只读缓存

**契约要点**：索引越界的行为（报错 / 钳制 / 未定义）、重复索引在 scatter 下的语义（累加 / 覆盖）必须写清。scatter 的累加语义会引入非确定性（见 [determinism](../../02-foundations/contracts/determinism.md)）。

## reshape 什么时候是真的搬

~~~text
连续张量 reshape  → 只改元数据，零拷贝
非连续 reshape    → 需要先 contiguous，产生一次真实拷贝
~~~

**这是性能陷阱**：一个看起来只是「改形状」的操作，可能触发全量拷贝。

契约里要明确：接受非连续输入并内部拷贝，还是要求调用方保证连续。前者对调用方友好但有隐藏成本，后者把成本显式化。

## 对齐

向量化的前提是 16 字节对齐和长度整除。pad 类算子要特别小心：**补出来的边界区域**往往破坏了对齐假设。

契约要写清：输出是否保证对齐；如果有 padding，padding 区域填什么值（0 / `-inf` / 不定义）。

## 检查清单

- [ ] 读写顺序冲突的取舍已说明（或用了 shared memory 中转）
- [ ] bank conflict 已用 padding 或其他方式消除
- [ ] gather/scatter 的越界行为已定义
- [ ] scatter 的重复索引语义已定义，非确定性已登记
- [ ] reshape 是否触发真实拷贝已说明
- [ ] 输出对齐保证已写明
- [ ] padding 区域的填充值已定义

## 常见误区

**以为 transpose 只是改 stride。** 只有在**不真正访存**时才是。一旦要读数据，它就是访存模式问题。

**忘记 bank conflict。** 用了 shared memory 却选了个撞 bank 的 tile 形状。

**假设 reshape 零成本。**

**gather 用随机索引做基准测试。** 结果不具代表性——真实索引的局部性决定性能。

## 相关页面

- [手法 · 访存模式](../../04-performance/techniques/memory-access.md)：合并访存与向量化
- [memory-hierarchy](../../02-foundations/memory-hierarchy.md)：访存效率的计算
- [indexing-and-layout](../../02-foundations/indexing-and-layout.md)：stride 与地址推导
- [常见错误清单](../../99-reference/common-pitfalls.md)：VA02（切片）、VA04（向量化尾块）
