# Semi-Prefill Overhead 分析报告（bfcl_long_context，Prefix Caching 场景）

## 1 实验概述

| 参数 | 值 |
|------|-----|
| 模型 | Llama-3.3-70B-Instruct |
| 上下文窗口 (cw) | 32,000 tokens |
| 压缩阈值 (threshold) | 30,000 tokens |
| 保留最近 (keep_recent) | 2,600 tokens |
| 摘要上限 (summary_max) | 1,024 tokens |
| 预留 (reserve) | 2,000 tokens |
| 有效样本数 | 10 / 10 |
| 配置来源 | run_config.json |

**硬件参数（理论模型，与 cw8000 参考报告保持一致）**

| 符号 | 含义 | 值 | 来源 |
|------|------|-----|------|
| $r_{pf}$ | Prefill 速率 | 0.15 ms/tok | cw8000 参考报告 |
| $r_{dec}$ | Decode 速率 | 12.0 ms/tok | cw8000 参考报告 |
| $c_{fix}$ | 请求固定开销 | 20 ms | cw8000 参考报告 |

**异步压缩 vs 同步压缩**

| 模式 | 含义 | 关键路径上的压缩成本 |
|------|------|-------------------|
| **Async（异步摘要）** | 摘要在后台/空闲时预生成，不阻塞用户请求 | 仅 Semi-Prefill：$(B_2+C_1) \times r_{pf} + c_{fix}$ |
| **Sync（同步摘要）** | 摘要在压缩触发时同步生成，用户必须等待 | Summary Prefill $B_1 \times r_{pf}$ + Summary Decode $B_2 \times r_{dec}$ + Semi-Prefill |

> validate 样本在 score summary 中为 invalid；本报告只分析压缩/prefill 工作负载，不报告最终准确率。

---

## 2 Agent 工作负载统计

### 2.1 基本统计

| 指标 | 值 |
|------|-----|
| 总轮数 | 137 |
| LLM calls | 217 |
| 压缩次数 | 20 |
| 压缩率 | 14.6% |
| 平均压缩/请求 | 2.00 |
| 平均 rounds/请求 | 13.70 |
| 平均累计上下文 | 423,920 tokens |
| 累计/首轮范围 | 13.5×–19.0× |

| 样本 | 总轮数 | 压缩次数 | 压缩率 | δ 中位数 | ctx max | 累计 ctx | 累计/首轮 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | 14 | 2 | 14.3% | 2,717 | 29,561 | 480,727 | 18.23 |
| C1 | 14 | 2 | 14.3% | 2,715 | 29,317 | 425,108 | 16.21 |
| C197 | 14 | 2 | 14.3% | 2,709 | 29,428 | 355,396 | 13.49 |
| C198 | 14 | 2 | 14.3% | 2,716 | 29,158 | 412,130 | 15.63 |
| C199 | 14 | 2 | 14.3% | 2,719 | 28,934 | 426,076 | 16.18 |
| C2 | 11 | 2 | 18.2% | 2,752 | 29,366 | 394,545 | 15.02 |
| V125 | 14 | 2 | 14.3% | 2,717 | 30,411 | 363,145 | 13.76 |
| V174 | 14 | 2 | 14.3% | 2,704 | 28,421 | 500,044 | 18.98 |
| V184 | 14 | 2 | 14.3% | 2,716 | 28,576 | 414,147 | 15.72 |
| V196 | 14 | 2 | 14.3% | 2,694 | 29,383 | 467,882 | 17.79 |


### 2.2 每轮新增 tokens（δ）

跨样本 δ 中位数约 **2,716 tokens/轮**。该值用于估计普通增量 prefill 基线：

$$\delta_{med} \times r_{pf} + c_{fix} \approx 2,716 \times 0.15 + 20 = 427\text{ ms/轮}$$

### 2.3 上下文长度分布

![Context Length Over Turns](charts/context_length_over_turns.png)

**特征**：图中蓝色为首轮 Full Prefill，绿色为 Prefix Cache 命中的增量轮，橙色为压缩后的 Semi-Prefill 轮。多轮 agentic request 平均跨越 **13.70 rounds**，平均累计上下文为 **423.9K tokens**，是单轮请求的 **13.5×–19.0×**。

### 2.4 Context Churn Ratio

![Context Churn Ratio](charts/context_churn_ratio.png)

增量轮 churn 由新增 token 占当前上下文比例估计；压缩轮 churn 由 $(B_2+C_1)/L_{post}$ 估计。压缩轮通常出现明显尖峰，因为 B₂ 重写导致 C₁ 也需要重新 prefill。

---

## 3 Semi-Prefill 触发统计

### 3.1 每次压缩的 token 段分布

![Semi-Prefill Composition](charts/semi_prefill_composition.png)

| 样本 | 事件 | Turn | A | B2 | C1 | B2+C1 | Async SP ms | Sync event ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | 1 | 2 | 0 | 698 | 2,925 | 3,623 | 563 | 12,431 |
| C0 | 2 | 10 | 0 | 1,347 | 5,388 | 6,735 | 1,030 | 20,222 |
| C1 | 1 | 2 | 0 | 497 | 2,928 | 3,425 | 534 | 10,352 |
| C1 | 2 | 11 | 0 | 968 | 5,386 | 6,354 | 973 | 16,010 |
| C197 | 1 | 2 | 0 | 907 | 2,744 | 3,651 | 568 | 14,991 |
| C197 | 2 | 10 | 0 | 1,088 | 5,380 | 6,468 | 990 | 17,040 |
| C198 | 1 | 1 | 0 | 374 | 2,821 | 3,195 | 499 | 8,097 |
| C198 | 2 | 9 | 0 | 329 | 5,389 | 5,718 | 878 | 7,755 |
| C199 | 1 | 1 | 0 | 305 | 2,759 | 3,064 | 480 | 7,572 |
| C199 | 2 | 9 | 0 | 1,343 | 5,392 | 6,735 | 1,030 | 20,211 |
| C2 | 1 | 2 | 0 | 447 | 5,462 | 5,909 | 906 | 9,502 |
| C2 | 2 | 9 | 0 | 1,037 | 5,383 | 6,420 | 983 | 16,429 |
| V125 | 1 | 1 | 0 | 350 | 2,705 | 3,055 | 478 | 8,481 |
| V125 | 2 | 9 | 0 | 346 | 5,390 | 5,736 | 880 | 8,243 |
| V174 | 1 | 1 | 0 | 391 | 2,832 | 3,223 | 503 | 8,286 |
| V174 | 2 | 8 | 0 | 1,296 | 5,378 | 6,674 | 1,021 | 19,392 |
| V184 | 1 | 1 | 0 | 315 | 2,732 | 3,047 | 477 | 7,669 |
| V184 | 2 | 9 | 0 | 1,314 | 5,389 | 6,703 | 1,025 | 19,792 |
| V196 | 1 | 1 | 0 | 548 | 3,700 | 4,248 | 657 | 10,334 |
| V196 | 2 | 8 | 0 | 1,258 | 5,369 | 6,627 | 1,014 | 19,046 |


**统计汇总**：

| 指标 | 值 |
|------|-----|
| B₂+C₁ 均值 | 5,030 tokens |
| B₂+C₁ 范围 | 3,047 – 6,735 tokens |
| C₁ 均值 | 4,273 tokens |
| B₂ 均值 | 758 tokens |
| Semi-prefill 均值 | 775 ms/event |
| Sync 单事件均值 | 13,093 ms/event |

### 3.2 Semi-Prefill 尖峰 vs 增量基线

![Semi-Prefill Spikes](charts/semi_prefill_spikes.png)

Async 下每次压缩事件的 semi-prefill 平均为 **775 ms**，约为普通增量 prefill 基线的 **1.8×**。

---

## 4 执行时间分解（Prefix Caching 理论模型）

### 4.1 公式

| 轮类型 | Prefill 成本 | Decode 成本 | 备注 |
|--------|-------------|-------------|------|
| T0（Full Prefill） | $L_0 \times r_{pf} + c_{fix}$ | $O \times r_{dec}$ | 冷启动 |
| 增量轮（Cached） | $\delta \times r_{pf} + c_{fix}$ | $O \times r_{dec}$ | Prefix Cache 命中 |
| 压缩轮（Async） | $(B_2 + C_1) \times r_{pf} + c_{fix}$ | $O \times r_{dec}$ | 仅 Semi-Prefill |
| 压缩轮（Sync） | $B_1 \times r_{pf} + B_2 \times r_{dec} + (B_2+C_1) \times r_{pf} + c_{fix}$ | $O \times r_{dec}$ | 含摘要生成 |

### 4.2 总时间分解

![Phase Breakdown](charts/phase_breakdown_stacked.png)

| 样本 | Async s | Sync s | Full ms | Incr ms | SP ms | Decode ms | SumDec ms | Sync overhead |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | 22.02 | 53.08 | 3,976 | 4,752 | 1,594 | 11,700 | 24,540 | 159.8% |
| C1 | 22.04 | 46.89 | 3,954 | 4,742 | 1,507 | 11,832 | 17,580 | 128.4% |
| C197 | 19.49 | 49.97 | 3,971 | 4,711 | 1,558 | 9,252 | 23,940 | 178.6% |
| C198 | 20.94 | 35.41 | 3,975 | 4,700 | 1,377 | 10,884 | 8,436 | 81.0% |
| C199 | 29.34 | 55.61 | 3,969 | 4,789 | 1,510 | 19,068 | 19,776 | 99.8% |
| C2 | 21.88 | 45.92 | 3,961 | 3,488 | 1,889 | 12,540 | 17,808 | 129.7% |
| V125 | 21.12 | 36.48 | 3,979 | 5,003 | 1,359 | 10,776 | 8,352 | 84.6% |
| V174 | 25.05 | 51.21 | 3,972 | 4,736 | 1,525 | 14,820 | 20,244 | 117.6% |
| V184 | 24.07 | 50.03 | 3,973 | 4,752 | 1,502 | 13,848 | 19,548 | 121.7% |
| V196 | 22.29 | 50.00 | 3,966 | 4,301 | 1,671 | 12,348 | 21,672 | 142.5% |


![Phase Pie Charts](charts/phase_pie_charts.png)

### 4.3 Async vs Sync 对比

![Phase Breakdown Sync vs Async](charts/phase_breakdown_sync_vs_async.png)

同步摘要使理论总延迟从 **228.2s** 增加到 **474.6s**，增幅 **107.9%**。新增成本主要来自 Summary Decode。

### 4.4 Prefill-Only 分解（排除 Decode）

![Async Prefill-Only Bar](charts/async_prefill_only_bar.png)

![Async Prefill-Only Pie](charts/async_prefill_only_pie.png)

**关键发现（Async）**：Semi-Prefill 只发生在 **14.6%** 的 rounds，却消耗了 **15.3%** 的 Prefill-only 计算量；Incremental Prefill 占 **45.4%**。

![Prefill-Only Sync vs Async](charts/prefill_only_sync_vs_async_bar.png)

Sync 下 Prefill-only 总量从 **101.2s** 增至 **347.5s**，倍率 **3.4×**，其中 Summary Decode 占 Sync Prefill-only 的 **52.3%**。

### 4.5 Per-Turn Latency Breakdown

前 3 个有效样本的逐轮图如下，文件命名与参考报告保持一致：

![Per-Turn Latency C0](charts/per_turn_latency_C0.png)
![Per-Turn Latency C1](charts/per_turn_latency_C1.png)
![Per-Turn Latency C197](charts/per_turn_latency_C197.png)

![Per-Turn Sync vs Async C0](charts/per_turn_latency_sync_async_C0.png)
![Per-Turn Sync vs Async C1](charts/per_turn_latency_sync_async_C1.png)
![Per-Turn Sync vs Async C197](charts/per_turn_latency_sync_async_C197.png)

---

## 5 同步 vs 异步摘要：时间占比对比

### 5.1 摘要生成的成本分解

![Per-Event Sync Breakdown](charts/per_event_sync_breakdown.png)

![Per-Event Sync vs Async](charts/per_event_sync_vs_async.png)

Async 下单次压缩事件平均只承担 **0.77s** 的 Semi-Prefill；Sync 下平均膨胀至 **13.09s**。

### 5.2 总延迟与占比

![Sync vs Async Stacked](charts/sync_vs_async_stacked.png)

![Sync vs Async Pie](charts/sync_vs_async_pie.png)

![Prefill-Only Breakdown](charts/sync_vs_async_prefill_only.png)

### 5.3 Overhead 对比

![Overhead Async vs Sync](charts/overhead_async_vs_sync.png)

| 场景 | 压缩 Overhead |
|------|-------------|
| Async | **7.28%** |
| Sync | **123.09%** |

---

## 6 核心结论

| 结论 | 数据支撑 |
|------|----------|
| 单个 agentic request 平均触发 2.00 次压缩 | 20 events / 10 active samples |
| 平均跨越 13.70 rounds | timing/summary 统计 |
| 平均累计上下文 423.9K tokens | sum(prompt_tokens) |
| 累计上下文是单轮的 13.5×–19.0× | accumulated / first prompt |
| Async 压缩 overhead 为 7.28% | 仅 Semi-Prefill 进关键路径 |
| Sync 压缩 overhead 为 123.09% | Summary Decode 进入关键路径 |

## 附录

- 生成脚本：`code/generate_cw8000_reference_analysis.py`
- JSON 结果：`analysis_results.json`
- 表格目录：`tables/`
- 图表目录：`charts/`
