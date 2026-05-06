# BFCL Long-Context Semi-Prefill Bench

这个目录用于测试 BFCL `multi_turn_long_context` 在 `cw=32000, threshold=30000, C1>2000` 下的压缩触发频率和 semi-prefill overhead。

## 目标

- 对 BFCL long-context 样本做可控增强，使单样本内部压缩触发概率满足 `P <= 1/5`，并尽量达到 `P <= 1/6`。
- `P = compression_count / inference_step_count`。分析里同时保留 turn 数、step 数，避免把短样本的一次压缩误读为完整 benchmark 结论。
- 每次压缩记录完整 ABC message：压缩前 `[A - System Prefix] + [B1 - 完整历史] + [C1 - 最近保留段]`，压缩后 `[A - System Prefix] + [B2 - 压缩摘要] + [C1 - 最近保留段]`。
- 每个 ABC 段同时记录总 token 和逐 message token：`A_message_tokens`、`B1_message_tokens`、`C1_message_tokens`、`B2_message_tokens`。
- 支持样本之间跳过已完成样本，样本内部 turn/step checkpoint 恢复。
- 记录 TTFT、decode、压缩摘要生成耗时，用于 prefix caching 下的 prefill / semi-prefill / decode 占比分析。

## 目录

```text
bfcl_long_context_sp_bench/
├── sp_config.py          # 三模型、cw32k preset、数据路径、目标阈值
├── dataset_rewrite.py    # BFCL long-context 数据增强脚本
├── bench_handler.py      # 工具调用 runner、压缩、ABC、trace、checkpoint、timing
├── run.py                # search/full 执行入口
├── analyze.py            # workload + latency breakdown
├── param_search.py       # 搜索结果汇总
├── start_proxy.sh        # vllm_tool_proxy 启动脚本
├── sample_ids.json       # 默认 smoke/search 样本
└── results/              # 新实验输出根目录
```

输出结构参考 `/root/agentbench_semi_prefill_bench`：

```text
results/<run_name>/
├── run_config.json
├── sample_ids.json
├── summary.jsonl
├── prompt_logs/<sample>.jsonl
├── traces/<sample>.json
├── abc_segments/<sample>.json
├── timing/<sample>.json
└── checkpoints/<sample>.json
```

## 服务启动

vLLM split 服务参考 `/root/vllm/split_serve.sh`：

```bash
cd /root/vllm
./split_serve.sh --all
```

代理服务：

```bash
cd /root/bfcl_long_context_sp_bench
BACKEND_URL=http://10.10.111.43:8005 PORT=6003 ./start_proxy.sh
```

支持的模型 key：

```text
GLM-4-9B-0414
Llama-3.3-70B-Instruct
Qwen3-235B-A22B
```

对应模型路径在 `sp_config.py` 的 `MODEL_REGISTRY` 中配置。

## 数据集改造

生成增强数据集：

```bash
cd /root/bfcl_long_context_sp_bench
/usr/bin/python3 dataset_rewrite.py \
  --model Llama-3.3-70B-Instruct \
  --initial-target-tokens 26000 \
  --turn-growth-tokens 2600 \
  --min-turns 6
```

默认输出：

```text
data/BFCL_v4_multi_turn_long_context_sp.json
data/dataset_summary.json
sample_ids.json
```

改造逻辑：

- 保留原始 BFCL task、tools、initial_config。
- 在第一轮 user message 后追加 varied audit/context block，使 message + tool schema 初始估算接近 26k，低于 30k threshold。
- 在后续 turn 中追加约 2600 token 的 context block，使 boundary compression 的 C1 更容易超过 2000 token。
- 对原始 turn 数小于 6 的样本，追加 no-op context-maintenance turn，用于稳定 `P <= 1/5` 或 `P <= 1/6` 的分母。
- 该增强数据用于 overhead/trigger 研究，不等同于官方 BFCL accuracy 数据。

## 运行

短搜索：

```bash
cd /root/bfcl_long_context_sp_bench
/usr/bin/python3 run.py search \
  --model Llama-3.3-70B-Instruct \
  --prepare-data \
  --prepare-max-samples 0 \
  --presets cw32k_thr30000_c12600 \
  --limit 3
```

完整 smoke/full：

```bash
/usr/bin/python3 run.py full \
  --model Llama-3.3-70B-Instruct \
  --preset cw32k_thr30000_c12600 \
  --limit 6
```

切换模型示例：

```bash
/usr/bin/python3 run.py full --model GLM-4-9B-0414 --preset cw32k_thr30000_c12600 --limit 3
/usr/bin/python3 run.py full --model Qwen3-235B-A22B --preset cw32k_thr30000_c12600 --limit 3
```

所有运行都会写入新的 `results/<run_name>/`。不要删除旧结果；如果要做新一轮，使用新的 `--run-name` 或 `--results-root`。

## 分析

```bash
/usr/bin/python3 analyze.py results/full_Llama-3.3-70B-Instruct_cw32k_thr30000_c12600
/usr/bin/python3 param_search.py --presets cw32k_thr30000_c12600
```

`analysis.json` 包含：

- 每个任务的 turn 数、step 数、工具调用次数。
- 每步新增 token 数：`new_tokens_per_step`。
- 总上下文长度分布：`context_length_distribution`。
- 上下文变化比例：压缩事件的 `(pre_prompt_tokens - post_prompt_tokens) / pre_prompt_tokens`。
- semi-prefill 次数和每次长度：`semi_prefill_count`、`semi_prefill_lengths`、`semi_prefill_events`。
- 时延拆分：`prefill_ms`、`semi_prefill_ms`、`summary_gen_ms`、`incremental_ms`、`decode_ms` 和百分比。

## 判定口径

一次短跑只能证明该样本/该模型/该 preset 的触发行为，不能证明全量 benchmark 结论。合格检查建议：

```text
P <= 1/5: compression_count / inference_step_count <= 0.2
P <= 1/6: compression_count / inference_step_count <= 0.1667
C1 lower bound: C1_tokens_after > 2000
```

如果 `P=0`，说明没有触发压缩，不适合用于 semi-prefill overhead 结论；如果 `P>1/5`，需要降低初始 target/growth 或提高分母 turn/step；如果 `C1<=2000`，需要提高对应 preset 的 `keep_recent_tokens_budget` 或增加最近 turn 的 context block。
