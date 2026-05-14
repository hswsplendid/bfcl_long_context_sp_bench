# 通信量化消融实验方案

日期：2026-05-14

## 1. 目标

本实验只测 BFCL long-context 样本 `multi_turn_long_context_125` 的首轮表现，用同一条请求比较通信量化方案对以下指标的影响：

- 通信量：从 vLLM worker 日志中汇总每次 hidden-state 传输的 payload / serialized bytes。
- TTFT：BFCL runner 记录的首个输出 chunk 延迟。
- TPOT：`decode_ms / max(output_tokens - 1, 1)`。
- task quality：主指标是首轮工具调用是否通过 BFCL 官方 multi-turn checker，指标名为 `first_turn_accuracy`，单样本时为 `0/1`；辅助指标包括 function-name F1、call-exact F1、argument-key F1、argument-value F1、调用数量是否一致和调用顺序是否一致。

这个结论只代表样本 125 的首轮质量，不代表完整 14 轮或完整 benchmark accuracy。

## 2. 版本定义

split serving 有两跳通信：

- hop1：EH -> CB。
- hop2：CB -> ET。

实验编号 1/2/3/5/6 对应如下：

| 编号 | 名称 | `--comm-ablation` | hop1 | hop2 |
|---:|---|---|---|---|
| 1 | Full pClaw | `full_pclaw` | FV3 pClaw similarity-aware INT4/INT8 | Int4Ref pClaw，使用 EH ref |
| 2 | Direct-INT8 | `direct_int8` | 直接 INT8，无 similarity ref | 直接 INT8，无 EH ref |
| 3 | Direct-INT4 | `direct_int4` | 直接 INT4，无 similarity ref | 直接 INT4，无 EH ref |
| 5 | w/o Inter-Similarity-INT4 | `hop1_raw_hop2_pclaw` | raw hidden，完全关闭第 1 跳优化 | Int4Ref pClaw |
| 6 | w/o Intra-Similarity-INT4 | `hop1_pclaw_hop2_raw` | FV3 pClaw | raw hidden，完全关闭第 2 跳优化 |

注意：这里的 5/6 按“两跳通信”解释，不是在同一跳内部只关某个小模块。编号 5 是关第 1 跳、开第 2 跳；编号 6 是开第 1 跳、关第 2 跳。

## 3. 代码开关

新增/使用的环境变量：

| 变量 | 值 | 含义 |
|---|---|---|
| `VLLM_COMM_ABLATION_MODE` | `full_pclaw/direct_int8/direct_int4/hop1_raw_hop2_pclaw/hop1_pclaw_hop2_raw/raw` | 实验版本名 |
| `VLLM_COMM_HOP1_MODE` | `pclaw/direct_int8/direct_int4/raw` | EH -> CB 模式 |
| `VLLM_COMM_HOP2_MODE` | `pclaw/direct_int8/direct_int4/raw` | CB -> ET 模式 |
| `VLLM_FORWARD_V3_ENABLE` | `1/0` | hop1 非 raw 时开启 FV3 |
| `VLLM_COMM_SHM` | `1/0` | hop2 pClaw 需要 EH -> ET SHM ref |
| `VLLM_COMM_INT4REF` | `1/0` | hop2 pClaw 的 Int4Ref codec |
| `VLLM_FV3_TIMING` | `1` | 打印 FV3 payload、encode/decode 时序 |
| `VLLM_SPLIT_TIMING` | `1` | 打印 raw / queued 传输 bytes |
| `VLLM_PROFILE_LATENCY` | `1` | 打印通信 profiler 汇总 |

`split_serve.sh` 已经把 `--comm-ablation` 自动映射成这些环境变量；正常实验不需要手动设置 hop1/hop2。

## 4. 启动服务

每个版本都要重启两台机器上的服务。建议每次使用新的 log 目录，避免覆盖旧日志。

### 4.1 SSH 拓扑和指令正确性

已知 SSH config：

- `ssh a800-1`：连接 A800-1 公网入口，实际 `HostName 211.81.55.182`、`User jumper`、`Port 1194`。
- `ssh 153`：通过 `ProxyJump a800-1` 连接 153，实际内网地址 `10.10.111.101`、`User hsw`。
- `ssh 211.81.55.181`：同样连到 `211.81.55.182:1194`，但用户是 `hsw`。除非明确要使用该用户，否则文档中的 A800-1 示例统一使用 `a800-1`。

结论：五个消融模式的 `--comm-ablation ${MODE}` 指令是正确的；需要修正的是远程执行和日志同步方式。不要写裸 `211.81.55.182:/path`，因为这会绕过 SSH config 里的 `Port 1194`、`User` 和 `ProxyJump`。日志同步应使用 `a800-1:/path` 或显式 `rsync -e 'ssh -p 1194' user@211.81.55.182:/path`。

A800-1 的 CB 启动命令仍然应使用 `--ip 10.10.111.101`。这里的 `--ip` 不是 SSH 地址，而是 CB 连接 controller 的内网地址。

开始前先做轻量预检：

```bash
ssh 153 'hostname; test -d /root/vllm && echo /root/vllm-ok'
ssh a800-1 'hostname; test -d /root/vllm && echo /root/vllm-ok'
```

如果登录用户不能访问 `/root/vllm`，需要换成实际部署 vLLM 的账号或路径；下面命令默认两台机器上的源码路径都是 `/root/vllm`。

### 4.2 交互式启动

153 服务器，也就是 controller + EH + ET：

```bash
cd /root/vllm

MODE=full_pclaw
RUN=bfcl125_${MODE}_turn1_20260514_001
export VLLM_LOG_DIR=/root/vllm/logs/${RUN}/153
export VLLM_FV3_TIMING=1
export VLLM_SPLIT_TIMING=1
export VLLM_PROFILE_LATENCY=1

./split-serve.sh --ctrl --eh --et --comm-ablation ${MODE}
```

A800-1，公网标识 `211.81.55.182`，CB 连接 controller 内网 IP `10.10.111.101`：

```bash
cd /root/vllm

MODE=full_pclaw
RUN=bfcl125_${MODE}_turn1_20260514_001
export VLLM_LOG_DIR=/root/vllm/logs/${RUN}/a800_1
export VLLM_FV3_TIMING=1
export VLLM_SPLIT_TIMING=1
export VLLM_PROFILE_LATENCY=1

./split-serve.sh --cb --ip 10.10.111.101 --comm-ablation ${MODE}
```

也可以从有上述 SSH config 的控制机直接启动。以下命令是长运行命令，建议分别放在两个终端或 tmux session 中：

```bash
MODE=full_pclaw
RUN=bfcl125_${MODE}_turn1_20260514_001

ssh -t 153 "cd /root/vllm && \
  VLLM_LOG_DIR=/root/vllm/logs/${RUN}/153 \
  VLLM_FV3_TIMING=1 \
  VLLM_SPLIT_TIMING=1 \
  VLLM_PROFILE_LATENCY=1 \
  ./split-serve.sh --ctrl --eh --et --comm-ablation ${MODE}"
```

```bash
MODE=full_pclaw
RUN=bfcl125_${MODE}_turn1_20260514_001

ssh -t a800-1 "cd /root/vllm && \
  VLLM_LOG_DIR=/root/vllm/logs/${RUN}/a800_1 \
  VLLM_FV3_TIMING=1 \
  VLLM_SPLIT_TIMING=1 \
  VLLM_PROFILE_LATENCY=1 \
  ./split-serve.sh --cb --ip 10.10.111.101 --comm-ablation ${MODE}"
```

把 `MODE` 依次替换为：

```text
full_pclaw
direct_int8
direct_int4
hop1_raw_hop2_pclaw
hop1_pclaw_hop2_raw
```

服务就绪检查：

```bash
tail -80 /root/vllm/logs/${RUN}/153/vllm_server.log
tail -80 /root/vllm/logs/${RUN}/153/worker_eh.log
tail -80 /root/vllm/logs/${RUN}/153/worker_et.log
tail -80 /root/vllm/logs/${RUN}/a800_1/worker_cb.log
```

期望能看到：

- Full pClaw：`ForwardV3Encoder mode=pclaw`、`Int4ref warmup done`、`Int4ref encoded/decoded`。
- Direct-INT8：`ForwardV3Encoder mode=direct_int8`、`DirectQuant warmup done`、`DirectQuant encoded/decoded bits=int8`。
- Direct-INT4：`ForwardV3Encoder mode=direct_int4`、`DirectQuant encoded/decoded bits=int4`。
- w/o hop1：EH -> CB 日志走 `SenderProc BASE ... bytes=...MB`，CB -> ET 有 `Int4ref encoded`。
- w/o hop2：EH -> CB 有 FV3 payload，CB -> ET 走 `SenderProc BASE ... bytes=...MB`。

## 5. 启动 tool proxy

BFCL runner 默认请求 `http://localhost:6003/v1`，需要本地 proxy 指向 vLLM controller。

如果 runner 在 153 上运行：

```bash
cd /root/bfcl_long_context_sp_bench
BACKEND_URL=http://127.0.0.1:8055 PORT=6003 ./start_proxy.sh
```

如果 runner 不在 153 上运行，把 backend 改为 controller 内网地址：

```bash
cd /root/bfcl_long_context_sp_bench
BACKEND_URL=http://10.10.111.101:8055 PORT=6003 ./start_proxy.sh
```

## 6. 运行样本 125 首轮

每个版本使用独立 `RUN` 名称。不要复用旧结果目录。

```bash
cd /root/bfcl_long_context_sp_bench

MODE=full_pclaw
RUN=bfcl125_${MODE}_turn1_20260514_001

/usr/bin/python3 run.py full \
  --model Llama-3.3-70B-Instruct \
  --preset cw32k_thr30000_c12600 \
  --ids multi_turn_long_context_125 \
  --max-turns 1 \
  --run-name ${RUN} \
  --disable-compression \
  --force
```

说明：

- `--max-turns 1` 只跑首轮。
- `--disable-compression` 隔离 context compression，首轮实验不需要它。
- `--force` 只覆盖当前 `RUN` 目录里的同名样本；不要对旧实验目录使用同一个 `RUN`。

## 7. 评分和汇总

先做首轮质量评分：

```bash
cd /root/bfcl_long_context_sp_bench

MODE=full_pclaw
RUN=bfcl125_${MODE}_turn1_20260514_001

/usr/bin/python3 score_runs.py results/${RUN} \
  --ids multi_turn_long_context_125 \
  --first-turn-only \
  --output results/${RUN}/scores/first_turn_score.json
```

`score_runs.py` 输出两层质量指标：

- `accuracy` / `first_turn_accuracy`：官方 BFCL multi-turn checker 的首轮 0/1 结果，是主指标。
- `quality_breakdown` 和 `entries[].fine_grained_quality`：辅助诊断指标，用来解释错误来源，不替代官方分数。

首轮 task quality 建议报告以下字段：

| 指标 | 含义 | 用途 |
|---|---|---|
| `first_turn_accuracy` | BFCL 官方 checker 是否通过 | 主指标，能验证工具调用执行语义是否满足官方答案 |
| `function_name_f1_mean` | 预测工具名集合和答案工具名集合的 micro-F1 | 判断是否选对工具 |
| `call_exact_f1_mean` | 工具名和完整参数签名都相同的调用 F1 | 判断整条调用是否精确匹配 |
| `argument_key_f1_mean` | 参数键集合 F1 | 判断是否漏传或多传参数 |
| `argument_value_f1_mean` | 参数键值对 F1 | 判断参数值是否正确 |
| `call_count_exact_rate` | 预测调用数是否等于答案调用数 | 识别漏调用、重复调用或额外调用 |
| `function_order_exact_rate` | 每轮工具名顺序是否一致 | 多工具调用时判断顺序错误 |

F1 的定义是：precision = 预测中正确项 / 预测项，recall = 预测中正确项 / 标准答案项，F1 是二者调和平均。这里的“项”不建议用自然语言 token，而建议用结构化工具调用项，例如函数名、完整调用签名、参数键、参数键值对。

对样本 `multi_turn_long_context_125` 的首轮，官方答案是：

```text
get_watchlist()
```

所以首轮最有解释力的是 `first_turn_accuracy`、`function_name_f1_mean`、`call_exact_f1_mean`、`call_count_exact_rate` 和 `function_order_exact_rate`。因为 `get_watchlist()` 没有参数，`argument_key_f1_mean` 和 `argument_value_f1_mean` 在首轮基本只是“没有参数是否一致”的诊断，信息量有限。若以后把评估扩展到第 2 轮，答案包含 `get_stock_info(symbol='ALPH')` 和 `place_order(order_type='Buy',symbol='ALPH',price=1320.45,amount=100)`，参数级 F1 才会变得非常有用。

还可以作为辅助但不建议作为主指标的质量项：

- JSON / tool-call schema validity：模型输出是否能被 decoder 解析，参数是否符合工具 schema。
- required argument recall：必填参数是否全部出现。
- type correctness：数值、字符串、枚举、列表、字典类型是否正确。
- numeric tolerance score：浮点数价格等字段是否在容差内。
- execution success rate：工具调用能否实际执行，不抛异常。
- state-change correctness：多轮任务中执行后的状态是否和官方模拟环境一致。
- no-call correctness：应答轮如果不该调用工具，模型是否没有误调用。
- repetition consistency / pass@N：同一模式重复 N 次时首轮是否稳定通过。

需要注意：自然语言 token-level F1 对 BFCL function calling 不太合适。模型可以用不同文本包裹同一个工具调用，或者输出看起来相似但无法执行的调用；反过来，文本差异很大也可能工具调用完全正确。因此本实验推荐结构化 F1，只作为官方 checker 之外的诊断。

保存两台机器的 vLLM 日志到结果目录。153 本机日志：

```bash
cd /root/bfcl_long_context_sp_bench
mkdir -p results/${RUN}/vllm_logs/153
cp -a /root/vllm/logs/${RUN}/153/*.log results/${RUN}/vllm_logs/153/
```

从 A800-1 拉取 CB 日志：

```bash
cd /root/bfcl_long_context_sp_bench
mkdir -p results/${RUN}/vllm_logs/a800_1
rsync -av a800-1:/root/vllm/logs/${RUN}/a800_1/*.log results/${RUN}/vllm_logs/a800_1/
```

如果汇总机器不在 153 上，也用 SSH alias 拉取 153 日志：

```bash
cd /root/bfcl_long_context_sp_bench
mkdir -p results/${RUN}/vllm_logs/153
rsync -av 153:/root/vllm/logs/${RUN}/153/*.log results/${RUN}/vllm_logs/153/
```

汇总 TTFT、TPOT、质量和通信量：

```bash
cd /root/bfcl_long_context_sp_bench

/usr/bin/python3 collect_ablation_metrics.py results/${RUN} \
  --first-turn-only \
  --score-file results/${RUN}/scores/first_turn_score.json \
  --vllm-log-dir results/${RUN}/vllm_logs \
  --output results/${RUN}/metrics/ablation_metrics.json
```

最终重点看：

- `metrics/ablation_metrics.json -> timing.ttft_ms_mean`
- `metrics/ablation_metrics.json -> timing.tpot_ms_mean`
- `metrics/ablation_metrics.json -> quality.accuracy`
- `metrics/ablation_metrics.json -> quality.quality_breakdown.function_name_f1_mean`
- `metrics/ablation_metrics.json -> quality.quality_breakdown.call_exact_f1_mean`
- `metrics/ablation_metrics.json -> communication.wire_or_direct_payload_total_mb`
- `metrics/ablation_metrics.json -> communication.codec_payload_total_mb` 作为 codec 侧估计，不能和 wire 字段相加
- `communication.by_file` 区分 EH/CB/ET 日志来源。

## 8. 推荐批量执行顺序

先跑一个轻量 sanity check，再跑五个版本：

```bash
MODES=(
  full_pclaw
  direct_int8
  direct_int4
  hop1_raw_hop2_pclaw
  hop1_pclaw_hop2_raw
)
```

每个 `MODE` 的顺序：

1. 两台机器停止上一轮服务。
2. 两台机器设置同一个 `RUN` 和各自 `VLLM_LOG_DIR`。
3. 153 启动 `./split-serve.sh --ctrl --eh --et --comm-ablation ${MODE}`。
4. A800-1 启动 `./split-serve.sh --cb --ip 10.10.111.101 --comm-ablation ${MODE}`。
5. 确认 logs 中对应模式和 warmup 成功。
6. 启动/确认 `vllm_tool_proxy`。
7. 运行 BFCL 样本 125 首轮。
8. 跑 `score_runs.py --first-turn-only`。
9. 保存两台机器日志并跑 `collect_ablation_metrics.py`。

## 9. 是否需要重新编译

当前改动都在 Python / Triton JIT 路径：

- `vllm/v1/engine/comm/forward_v3_comm.py`
- `vllm/v1/engine/comm/manager.py`
- `vllm/v1/engine/comm/direct_quant_codec.py`
- `vllm/v1/engine/comm/tcp_transport.py`
- `vllm/v1/engine/comm/forward_v3_decode_kernel.py` 中的 fused kernels 是 Triton Python JIT kernel。

因此如果运行时导入的是 `/root/vllm/vllm` 源码目录，不需要 CMake / CUDA 扩展重新编译；需要做的是重启所有 vLLM 进程，让 Python 代码和环境变量重新加载。

必须确认两件事：

```bash
/usr/bin/python3 - <<'PY'
import vllm
print(vllm.__file__)
PY
```

期望路径指向：

```text
/root/vllm/vllm/__init__.py
```

然后确认 fused path 没有回退：

```bash
grep -R "Triton fused kernels loaded\|Triton kernels pre-compiled\|DirectQuant warmup done\|Int4ref warmup done" /root/vllm/logs/${RUN} -n
```

如果看到 `Triton unavailable`、`Triton warmup failed`、`Direct INT8 fused decode unavailable` 或 `Direct INT4 fused decode unavailable`，这次结果可能回退到了较慢的 PyTorch 解码路径，不能作为性能结论。

只有当修改了 `csrc/`、`CMakeLists.txt`、`setup.py` 或 vLLM 已经不是从 `/root/vllm` 源码目录导入时，才需要重新安装/编译 vLLM。这个实验的新增消融代码本身不需要重新编译。

## 10. 已验证与未验证边界

这份流程能验证：

- 样本 125 首轮的 TTFT / TPOT / first-turn accuracy。
- 每个版本在两跳通信上的 payload / serialized bytes 日志。
- fused Triton kernel 是否加载和预热。

这份流程不能单独证明：

- 完整 14 轮任务质量。
- 多样本平均准确率。
- 长时间服务稳定性。
- 首个请求冷启动之外的 P50/P99，需要每个版本增加 warmup + 多次重复后再统计。