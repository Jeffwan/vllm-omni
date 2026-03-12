# TensorRT-LLM Disaggregated Serving 深度分析报告

> 调研日期: 2026-03-12
> 覆盖范围: ai-dynamo 项目架构、TRT-LLM 集成点（编排/路由/KV Events/KV Cache Offloading）、AIBrix 集成方案

---

## 目录

1. [ai-dynamo 项目全景](#1-ai-dynamo-项目全景)
2. [TRT-LLM 编排机制](#2-trt-llm-编排机制)
3. [路由支持](#3-路由支持)
4. [KV Events 机制](#4-kv-events-机制)
5. [KV Cache Offloading](#5-kv-cache-offloading)
6. [在 AIBrix 中支持 TensorRT-LLM](#6-在-aibrix-中支持-tensorrt-llm)
7. [关键结论与建议](#7-关键结论与建议)

---

## 1. ai-dynamo 项目全景

### 1.1 项目定位

NVIDIA Dynamo（ai-dynamo）是 NVIDIA 在 GTC 2025 发布的**开源数据中心级分布式推理服务框架**，是 Triton Inference Server 的下一代演进，专为 Reasoning 和生成式 AI 模型（特别是 LLM）设计。Apache 2.0 协议开源。

- **仓库**: https://github.com/ai-dynamo/dynamo
- **性能**: DeepSeek-R1 671B 在 GB200 NVL72 上吞吐提升 **30x**；Llama 70B 在 Hopper GPU 上提升 **2x+**

### 1.2 技术栈

| 层次 | 技术 |
|------|------|
| 核心运行时 | **Rust**（高性能、低延迟） |
| 用户扩展层 | **Python**（通过 PyO3/Maturin 绑定） |
| 推理引擎 | TensorRT-LLM、vLLM、SGLang、Mocker（测试用） |
| 服务发现 | etcd / Kubernetes CRD / 文件系统（单机） |
| 事件总线 | NATS JetStream / ZMQ（v0.9.0+ 默认） |
| KV Cache 传输 | NIXL（NVIDIA Inference Xfer Library） |

### 1.3 三层命名抽象

```
Namespace > Component > Endpoint
例: dynamo > VllmWorker > generate
```

### 1.4 核心组件架构

```
Client HTTP Request
  │
  ▼
┌────────────────────────────────┐
│  Frontend (HttpService)        │  Rust HTTP 服务器, OpenAI 兼容 API
│  - /v1/chat/completions        │
│  - /v1/completions             │
└────────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────┐
│  OpenAIPreprocessor            │  分词、应用 Chat Template
└────────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────┐
│  Smart Router (KvRouter)       │  KV Cache 感知路由
│  ├── KvIndexer (RadixTree)     │  前缀匹配查找
│  └── KvScheduler               │  负载+overlap 成本计算
└────────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────┐
│  Worker (TRT-LLM/vLLM/SGLang)  │  推理执行
│  - 执行推理                     │
│  - 发布 KvCacheEvent 到 NATS    │
└────────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────┐
│  KvSubscriber                   │  更新 Router 的 RadixTree 索引
└────────────────────────────────┘
```

### 1.5 关键组件说明

**A. GPU Resource Planner（GPU 资源规划器）**
- 持续监控 GPU 容量、请求速率、序列长度、队列等待时间
- 动态调整 Prefill/Decode 的 GPU 分配比例
- 可在 Disaggregated 和 Aggregated 模式间动态切换
- 集成 SLO 目标（TTFT、ITL）驱动分配决策
- Grace period 防止抖动（新 decode worker 加入后 3 个间隔内不缩减）

**B. KVBM（分布式 KV Cache Block Manager）**
三层架构：
1. **模型集成层**: 与推理引擎连接，无需引擎特定定制
2. **内存管理层**: 分配、组织、复用追踪、自定义驱逐/卸载策略
3. **存储与传输层**: 通过 NIXL 连接 GPU HBM → CPU RAM → 本地 SSD → 远程/云存储

**C. NIXL（NVIDIA Inference Xfer Library）**
- 统一 API 覆盖: NVLink、InfiniBand/RoCE、PCIe/Ethernet、GPUDirect Storage、UCX、S3
- "通用内存段"抽象: HBM、DRAM、本地 SSD、网络存储统一为通用传输基质
- 非阻塞、非连续内存区域传输

### 1.6 部署模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| **单机 Docker** | Docker Compose 部署 NATS + etcd + Workers | 开发测试 |
| **文件系统发现** | `--discovery-backend file://`，无需 etcd/NATS | 单节点快速验证 |
| **Kubernetes + Grove** | CRD 描述整个推理系统，统一管理 | 生产环境 |
| **Slurm** | `srun` 启动多节点 worker | HPC 集群 |

**Grove**（https://github.com/ai-dynamo/grove）是 NVIDIA 的开源 K8s 编排层：
- 单个 Custom Resource (CR) 描述完整推理系统（prefill pool、decode pool、router、frontend）
- 层级 gang scheduling、拓扑感知 GPU 放置、多级自动扩缩容
- 支持显式启动顺序

### 1.7 生态与采纳

已宣布的采纳者: AWS、Google Cloud、Microsoft Azure、CoreWeave、Fireworks、Meta、Perplexity、Together AI、Cohere 等。

---

## 2. TRT-LLM 编排机制

### 2.1 Disaggregated Serving 核心原理

LLM 推理的两个阶段有截然不同的资源需求：

| 阶段 | 资源特征 | 描述 |
|------|----------|------|
| **Prefill（预填充）** | **计算密集** | 并行处理整个用户 prompt，密集矩阵乘法 |
| **Decode（解码）** | **内存带宽密集** | 自回归地逐 token 生成，每步读取整个 KV Cache |

Aggregated serving 中两阶段争夺同一 GPU 资源。Disaggregated serving 将它们分到**不同 GPU 池**，可使用不同的并行策略：
- Prefill workers: **较低 TP**（减少通信开销，适合计算密集型）
- Decode workers: **较高 TP**（提升内存带宽利用率）

### 2.2 Worker 架构

**Prefill Workers（Context Workers）**:
- 处理用户输入 prompt，计算所有 prompt token 的 KV Cache 并产出第一个 token
- 不支持请求迁移（`--migration-limit=0`）
- 启动参数: `--disaggregation-mode prefill --disaggregation-strategy prefill_first`
- 可使用低 TP（如 EP4DP16）

**Decode Workers（Generation Workers）**:
- 使用缓存的 KV 值逐 token 生成
- 支持可配置迁移限制的请求迁移
- 启动参数: `--disaggregation-mode decode --disaggregation-strategy prefill_first`
- 可使用高 TP（如 EP64DP3）

### 2.3 请求处理完整流程

```
Client Request (OpenAI 兼容 API)
        │
        ▼
  HTTP Frontend (Rust, 高性能)
        │
        ▼
  Pre-processing Worker (分词)
        │
        ▼
  Smart Router (KV-cache 感知路由)
        │
        ▼  [路由到最优 decode worker]
  Decode Worker
        │
        ├─→ [检查: 本地是否有 KV cache blocks?]
        │         │
        │    是: 跳过 prefill，直接开始解码
        │    否: 转发请求到 prefill worker
        │                    │
        │                    ▼
        │            Prefill Worker
        │              - 处理 prompt
        │              - 生成 KV cache blocks
        │              - 返回 ctx_params (decode worker 获取 KV blocks 的元数据)
        │                    │
        │                    ▼
        │         KV Cache Transfer (via NIXL/UCX/RDMA)
        │              - Prefill GPU ──→ Decode GPU
        │              - 可与其他请求的计算重叠
        │                    │
        ├←───────────────────┘
        │
        ▼
  Decode Worker (开始 token 生成)
        │
        ▼
  Post-processing Worker (去分词)
        │
        ▼
  Streaming Response 返回给 Client
```

### 2.4 关键编排细节

1. **Smart Router** 使用 **Radix Tree** 计算新请求与已缓存 KV blocks 的重叠分数
2. **KV cache bypass**: 若 decode worker 已有相关 KV cache（如多轮对话），可**完全跳过 prefill**
3. **Prefill worker 返回 `ctx_params`**: 元数据，使 decode worker 能从 prefill worker GPU 内存中检索 KV cache blocks
4. **传输计算重叠**: 一个请求的 KV cache 传输期间，其他请求继续计算；多 GPU 实例间不同 GPU 组的 cache 传输并行执行
5. **全局 Prefill Queue**: 基于 NATS stream 的 consumer group 模式。Decode worker push 请求到队列，多个 prefill worker pull 请求。两种 worker 完全解耦

### 2.5 多模态支持

Dynamo 支持 **Encode-Prefill-Decode (EPD)** 三阶段流水线，视觉编码分离为独立 worker 类型，通过 `epd_disagg.sh` 启动。

### 2.6 部署配置示例

**单节点 8 GPU 部署**:

```bash
# Prefill worker (GPUs 0-3)
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m dynamo.trtllm \
    --disaggregation-mode prefill \
    --disaggregation-strategy prefill_first \
    --publish-events-and-metrics \
    --engine-config prefill_engine.yaml

# Decode worker (GPUs 4-7)
CUDA_VISIBLE_DEVICES=4,7 python3 -m dynamo.trtllm \
    --disaggregation-mode decode \
    --disaggregation-strategy prefill_first \
    --publish-events-and-metrics \
    --engine-config decode_engine.yaml

# Frontend
python3 -m dynamo.frontend --router-mode round-robin --http-port 8000
```

**Kubernetes 部署** 使用 `DynamoGraphDeployment` 自定义资源。

**配置表示法**: `CTX=1xTEP-4|GEN=2xDEP-8` 表示 1 个 context 实例用 Tensor-Expert Parallelism 4，加 2 个 generation 实例用 Data-Expert Parallelism 8。

### 2.7 GPU Resource Planner 调度逻辑

- 每 `metric-pulling-interval` 从所有 worker 收集指标
- 每 `adjustment-interval` 与预设阈值比较
- 每次调整间隔最多改变 **1 个 worker** 数量（防止过度补偿）
- Grace period:
  - `NEW_DECODE_WORKER_GRACE_PERIOD=3`: 新增 decode worker 后 3 个间隔内不缩减
  - `NEW_PREFILL_WORKER_QUEUE_BUFFER_PERIOD=3`: 若预测 queue size 在 3 个间隔内降至阈值以下，不扩增 prefill
- **AI Configurator (v0.4+)**: 预测不同调度技术的性能影响，前瞻性扩缩容

---

## 3. 路由支持

### 3.1 Router 组件架构

Router 用 **Rust** 实现，源码位于 `lib/llm/src/kv_router/`：

| 模块 | 职责 |
|------|------|
| `kv_router.rs` | 顶层路由模块 |
| `kv_router/indexer.rs` | KvIndexer：RadixTree + 异步事件处理 |
| `kv_router/scheduler.rs` | 基于成本的 worker 选择（核心调度逻辑在 ~269-300 行） |
| `kv_router/approx.rs` | 近似 KV 索引器（TTL 过期，无需事件） |
| `kv_router/publisher.rs` | KV 事件发布到事件面板 |
| `kv_router/subscriber.rs` | 从 worker 消费 KV 事件 |

### 3.2 四种路由策略

通过 `--router-mode` 选择：

| 策略 | 说明 |
|------|------|
| **Random** | 均匀随机选择 worker，用作 baseline |
| **Round-Robin** | 顺序轮询可用 worker |
| **KV-Aware (Exact)** | 主策略。通过真实 KV cache 事件填充 RadixTree 计算精确 overlap 分数。需要 NATS/ZMQ 事件面板 |
| **KV-Aware (Approximate)** | 从路由历史推断 cache 状态，TTL 过期（默认 120s）+ 大小剪枝。不需要 worker KV 事件 |

### 3.3 KV-Aware 成本函数（核心）

```
cost = kv_overlap_score_weight × prefill_blocks + decode_blocks
```

其中：
- **prefill_blocks** = 请求需要的总 blocks - 该 worker 上已缓存的 blocks（即需要新计算的量）
- **decode_blocks** = 该 worker 上当前活跃生成使用的 blocks（负载指标）
- **kv_overlap_score_weight**（默认 1.0）= 调节 TTFT 优化 vs ITL 优化的权重

Router 选择**成本最低**的 worker。当 `router_temperature > 0` 时，对归一化成本 logits 做 softmax 采样，引入概率性选择以更好地分布负载。

**示例**（overlap_score_weight = 1.0，请求需要 10 个 blocks）：
- Worker A: 缓存 2 blocks → 1.0 × (10-2) + 10 active = **18**
- Worker B: 缓存 5 blocks → 1.0 × (10-5) + 5 active = **10** ← 被选中
- Worker C: 缓存 8 blocks → 1.0 × (10-8) + 9 active = **11**

### 3.4 RadixTree 前缀匹配

`KvIndexer` 维护 **Radix Tree**（前缀树）追踪各 worker 持有的 KV cache blocks：

- **Block 哈希**: Token 序列确定性哈希为 block hash。共享相同前缀的两个请求在共享 blocks 上产生相同哈希。Block size 可配（典型 16-32 tokens）
- **哈希链**: 每个 block 的哈希包含前面 blocks 的信息，确保完整性
- **多模态 & LoRA 感知**: 多模态输入和 LoRA adapter ID 纳入 block 哈希计算，防止错误 cache 匹配
- **查找复杂度**: O(sequence_length) 前缀匹配
- **并发**: 支持单线程 (`KvIndexer`) 和并发 (`ThreadPoolIndexer<ConcurrentRadixTree>`) 变体
- **剪枝**: 超过 `router_max_tree_size`（默认 1,048,576）时剪枝至 `router_prune_target_ratio`（默认 0.8）

### 3.5 Router 状态管理

**前缀 blocks（持久化）**:
- 存储在 RadixTree 中
- 通过 NATS JetStream 事件和 Object Store 快照持久化
- 新 Router 副本自动同步：下载快照 + 从上次确认的 JetStream 位置回放事件
- `--router-snapshot-threshold`（默认 1,000,000）控制快照触发时机

**活跃 blocks（临时）**:
- 基于请求生命周期本地追踪
- Router 重启后重置，但随新请求处理收敛
- Prefill-only router 禁用此追踪（`track_active_blocks=false`）

### 3.6 Disaggregated 模式下的完整路由流程

```
1. HTTP Request 到达 Frontend
2. Decode Router (KV-aware) 使用成本函数选择 Decode Worker
3. Decode Worker 接收请求，做出 "disagg 决策":
   - 基于 prefill 长度和本地队列大小
   - 如果足够短，可能在本地做 prefill
4. 如果需要远程 prefill:
   a. Decode Worker 在本地 GPU 内存预分配 KV cache blocks
   b. 推送 RemotePrefillRequest（带 block ID）到全局 Prefill Queue (NATS stream)
5. Prefill Worker 从队列拉取请求（NATS consumer group 负载均衡）
6. Prefill Worker 从 etcd 加载 NIXL 元数据建立 GPU 通信
7. Prefill Worker 执行 prefill 计算
8. NIXL Transfer: 直接 GPU-to-GPU 写入 KV cache 到 Decode Worker 预分配的 blocks
9. Decode Worker 将请求加入 in-flight decode batch
10. Streaming token 生成返回给 client
```

### 3.7 多 Router 副本同步

Router 副本间通过三种事件类型同步：
- **AddRequest**: 通知请求分配给 worker（含 request ID、worker ID、token sequence blocks、overlap score）
- **MarkPrefillCompleted**: 信号从 prefill 到 decode 阶段的转换
- **Free**: 指示请求完成和资源释放

### 3.8 关键配置参数

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `--router-mode` | `kv` | 策略: `random`, `round-robin`, `kv` |
| `--kv-overlap-score-weight` | 1.0 | 平衡 prefill 成本 vs 负载 |
| `--router-temperature` | 0.0 | Softmax 采样温度（0 = 确定性） |
| `--router-ttl` | 120s | 近似模式中缓存 blocks 的 TTL |
| `--router-max-tree-size` | 1,048,576 | RadixTree 剪枝前最大 blocks |
| `--no-kv-events` | false | 禁用 KV 事件（回退到近似索引器） |
| `--router-replica-sync` | false | 启用跨副本活跃 block 共享 |
| `--router-snapshot-threshold` | 1,000,000 | 触发快照的 JetStream 消息数 |

---

## 4. KV Events 机制

### 4.1 TensorRT-LLM KV Cache 事件类型

TRT-LLM 的 KV cache 事件 API 定义了四种事件：

| 事件类型 | 内容 | 用途 |
|----------|------|------|
| `KVCacheCreatedData` | Block 元数据 | 新 cache blocks 被创建 |
| `KVCacheStoredData` | Block 哈希、序列信息 | 报告存储的 block 序列 |
| `KVCacheRemovedData` | Block 哈希 | 报告被驱逐/移除的 blocks |
| `KVCacheUpdatedData` | 更新的 block 元数据 | 捕获已有 blocks 的更新 |

事件通过**异步可迭代接口**获取，可配超时（默认 2 秒）。提供 cache 状态的最终一致性视图。

### 4.2 事件发布（Worker 侧）

```
TRT-LLM 引擎产生内部 cache 状态变化
        │
        ▼
KvEventPublisher 包装为 KvCacheEvent 消息（Stored/Removed 变体）
        │
        ▼
MessagePack 序列化（紧凑二进制编码）
        │
        ▼
发布到事件面板（NATS JetStream / ZMQ）
```

**TRT-LLM 特定配置**:
- KV 事件**默认禁用**（不同于 vLLM 默认启用）
- 引擎端启用: `kv_cache_connector_config.enable_kv_events`
- `--kv-event-mode` 控制传输: ZMQ 或 NATS
- ZMQ 模式下，publisher 发送 msgpack 序列化事件到 `tcp://*:5557`
- Workers 也暴露 `/kv_cache_events` HTTP 端点用于监控

### 4.3 事件消费（Router 侧）

1. `start_subscriber()` 启动后台任务消费事件面板中的事件
2. 每个事件调用 `apply_event(RouterEvent)` 更新 RadixTree
3. `Stored` 事件: 将 block 哈希插入 tree，关联到发布 worker ID
4. `Removed` 事件: 从 tree 删除对应条目
5. 更新的 tree 状态立即反映在后续 `OverlapScores` 计算中

### 4.4 三种事件传输机制

通过 `DYN_EVENT_PLANE` 环境变量控制：

**1. NATS JetStream（v0.9.0 前默认）**:
- 持久化事件存储，1 小时保留期
- Stream 命名: `namespace-{namespace}-component-{component}-kv-events`
- Router 持久化 consumer 位置，支持重启后完全恢复
- 快照保存到 NATS Object Store

**2. ZMQ Transport（v0.9.0+ 默认）**:
- 高性能基于 socket 的事件分发
- 无需集中式消息 broker
- MessagePack 序列化

**3. NATS Core + Local Indexer**:
- 临时（fire-and-forget）发布
- Workers 维护最近事件的本地循环缓冲区（默认 1024 事件）
- Router 通过 NATS RPC 直接查询 worker 恢复遗漏事件

### 4.5 KV Event 完整生命周期

```
1. BLOCK CREATION（块创建）
   Prefill worker 计算 prompt tokens 的 KV cache
   → 发射 KVCacheStoredData 事件（含 block 哈希）

2. ROUTER UPDATE（路由器更新）
   Router subscriber 接收 Stored 事件
   → 将 block 哈希插入 RadixTree，以 worker ID 为键

3. ROUTING DECISION（路由决策）
   新请求到达 frontend
   → Router 查询 RadixTree 获取所有 worker 的 overlap 分数
   → 成本函数选择最佳 cache-hit / 最低负载组合的 worker

4. KV TRANSFER（KV 传输 - disaggregated 模式）
   Decode worker 预分配 GPU 内存 blocks
   → Prefill worker 计算完成
   → NIXL 直接 GPU-to-GPU 写入 KV cache 到 Decode Worker 预分配 blocks
   → MarkPrefillCompleted 事件传播到所有 router 副本

5. BLOCK REMOVAL（块移除）
   内存压力或请求完成触发驱逐
   → 发射 KVCacheRemovedData 事件
   → Router 从 RadixTree 移除对应条目

6. INTER-ROUTER SYNC（跨 Router 同步）
   AddRequest、MarkPrefillCompleted、Free 事件
   在 router 副本间传播以维护一致负载视图
```

---

## 5. KV Cache Offloading

### 5.1 分层存储架构

```
GPU HBM（最快，最贵）
    │
    ▼
CPU Host RAM
    │
    ▼
本地 SSD
    │
    ▼
远程/云对象存储（最便宜，最慢）
```

KVBM 的驱逐策略将旧的或访问频率低的 blocks 逐层下移，可在整个数据中心管理潜在的 PB 级 KV cache。

### 5.2 NIXL 传输层

NIXL 提供统一传输抽象：

| 传输方式 | 说明 |
|----------|------|
| NVLink (C2C, NVSwitch) | 同节点 GPU 间直连 |
| InfiniBand / RoCE | 跨节点 RDMA |
| PCIe / Ethernet | 通用网络 |
| GPUDirect Storage | GPU 直接访问存储 |
| UCX | 统一通信 |
| S3 | 云对象存储 |

自动通过 "通用内存段" 选择最优数据路径。

### 5.3 KV Cache Transfer 后端对比

| 后端 | 状态 | 说明 |
|------|------|------|
| **NIXL** | 默认（v0.21.0+） | 推荐；支持动态节点加入/离开 |
| **UCX** | 支持 | 推荐；支持动态扩缩容 |
| **MPI** | 支持 | 适合静态部署 |

配置方式（引擎 YAML）：
```yaml
cache_transceiver_config:
  backend: NIXL  # 或 UCX, MPI, DEFAULT
```

### 5.4 Disaggregated KV Cache 传输流程

1. Worker 初始化后，将 **NIXL 元数据**（所有 KV cache blocks 的内存描述符）存储到 etcd
2. Prefill worker 首次使用时加载并缓存这些描述符；后续请求只需 block ID
3. 传输**非阻塞**: GPU forward pass 在传输期间继续服务其他请求
4. 支持 prefill 和 decode 间**不同 TP 策略**，高性能 kernel 将 KV blocks 转置为匹配布局
5. Dynamo 内存分配器合并连续 blocks 为更大 blocks 减少总传输次数

### 5.5 KVBM Connector API

Dynamo 提供自定义 connector:
- `DynamoKVBMConnectorLeader`（调度器侧）
- `DynamoKVBMConnectorWorker`（worker 侧）
- 模块路径: `kvbm.trtllm_integration.connector`（v0.9.0+）

支持层级 KV cache offloading:
```bash
# CPU 缓存
DYN_KVBM_CPU_CACHE_GB=60

# 磁盘缓存（需先启用 CPU offloading）
DYN_KVBM_DISK_CACHE_GB=20
```

构建: `./container/build.sh --framework trtllm --enable-kvbm`

### 5.6 KV Cache Connector API（TRT-LLM 内部）

TRT-LLM 的 KV Cache Connector API 提供状态传输的抽象层：
- KV cache 交换模块与 KV cache manager 和底层通信库**模块化解耦**
- 职责: 高效收发 cache、及时释放 cache 空间、交换过程中 cache 布局转换
- 重叠优化: 一个请求发送/接收 KV blocks 时，其他请求继续计算
- 多 GPU 并行: 不同 GPU 组间的 KV cache 传输并行执行

### 5.7 计划中的增强：层级异步传输（GitHub issue #9212）

不再等待完整 prefill 完成后才传输 KV cache，而是**每层 KV cache 计算完成后立即传输**。这将把有效 TTFT 降至仅 prefill 计算时间加上最后一层的传输时间。

### 5.8 性能基准（DeepSeek R1 on GB200）

| 配置 | ISL/OSL | 加速比 |
|------|---------|--------|
| 标准 disagg | 4400/1200 | 1.4x-1.8x |
| + MTP | 4400/1200 | 1.6x-2.5x |
| 标准 disagg | 8192/256 | 2x（8-GPU） |
| 标准 disagg | 4096/1024 | 1.7x @ 50 tokens/sec/user |

---

## 6. 在 AIBrix 中支持 TensorRT-LLM

### 6.1 AIBrix 现状

AIBrix 是 ByteDance 开发的**开源 Kubernetes 原生 GenAI 推理基础设施平台**，现位于 vLLM 项目组织下（`github.com/vllm-project/aibrix`）。最新版本 v0.6.0（2026-03-05）。

**当前支持的引擎**:
- **vLLM**: 主引擎，最深集成
- **SGLang**: v0.4.0 起支持
- **Dynamo/xLLM**: Helm values 和测试中引用

### 6.2 AIBrix 引擎集成机制

AIBrix 通过三个机制集成引擎：

**A. Kubernetes Deployment Labels**:
```yaml
labels:
  model.aibrix.ai/engine: "vllm"     # 引擎类型
  model.aibrix.ai/name: "llama-70b"  # 模型标识
  model.aibrix.ai/port: "8000"       # 服务端口
  model.aibrix.ai/metric-port: "8081" # Prometheus 端口
```

**B. Metrics Name Mapping（Go 代码）**:
`pkg/metrics/metrics.go` 中的 `Metric` struct 包含 `EngineMetricsNameMapping map[string]string`。每个 AIBrix 标准指标映射到引擎特定的 Prometheus 指标名：

| AIBrix 标准指标 | TRT-LLM/Triton 等价 |
|---|---|
| `num_requests_running` | `inflight_batcher_requests_active` |
| `gpu_cache_usage` | `kv_cache_block_usage` |
| `time_to_first_token` | `first_token_latency_ms` |
| `e2e_request_latency` | `request_latency_ms` |

**C. AI Runtime Sidecar**:
FastAPI 边车容器，处理模型下载、LoRA adapter 动态加载/卸载、指标归一化、健康检查。

### 6.3 AIBrix 的 Disaggregated Serving 支持

**StormService CRD**:
三层 CRD 层次结构：

```
StormService（顶层）── 定义整体服务、更新策略、副本数
    │
    ├── RoleSet[prefill] ── prefill 角色集合
    │       └── Pod ── 执行推理
    │
    └── RoleSet[decode] ── decode 角色集合
            └── Pod ── 执行推理
```

两种部署模式：
- **Replica Mode**（replicas > 1）: 每个 RoleSet 固定 P/D 比例
- **Pooled Mode**（replicas = 1）: 每个角色独立扩缩容

### 6.4 集成 TensorRT-LLM 的具体步骤

#### 步骤 1: Metrics Adapter（主要阻塞点）

TRT-LLM 的 Prometheus metrics 是主要挑战。`trtllm-serve` 提供 `/metrics` 端点，但存在稳定性问题（GitHub issue #9678 报告启用 Prometheus metrics 时 AttributeError 崩溃）。

**建议方案**: 使用 **Triton Inference Server 作为 TRT-LLM 的 serving frontend**，Triton 有稳定、文档完善的 Prometheus metrics。

在 `pkg/metrics/metrics.go` 中添加 TRT-LLM 引擎的 metrics name mapping:

```go
// 伪代码示例
engineMetrics["tensorrt-llm"] = map[string]string{
    "num_requests_running":    "inflight_batcher_requests_active",
    "gpu_cache_usage":         "kv_cache_block_usage",
    "time_to_first_token":     "first_token_latency_ms",
    "e2e_request_latency":     "request_latency_ms",
}
```

#### 步骤 2: OpenAI 兼容 API（基本就绪）

`trtllm-serve` 已提供 OpenAI 兼容的 chat/completions 端点。AIBrix Gateway 直接路由即可。

#### 步骤 3: AI Runtime Sidecar 扩展

需要 TRT-LLM 特定逻辑：

- **引擎编译**: TRT-LLM 需从 checkpoint 构建优化引擎（大模型 10-30 分钟）。这是与 vLLM/SGLang 最大的操作差异。方案: init containers / 预构建引擎镜像 / 独立 Builder Job
- **健康检查映射**: TRT-LLM 的 readiness probes 与 vLLM 不同
- **配置转换**: 量化、max batch size、并行策略等引擎特定设置

#### 步骤 4: Helm Chart 和部署模板

```yaml
# 新增 Helm values
labels:
  model.aibrix.ai/engine: "tensorrt-llm"
# 容器镜像: trtllm-serve 或 Triton + TRT-LLM backend
# GPU 资源请求
# RDMA 设备请求（disaggregated 场景）
```

#### 步骤 5: StormService 适配 TRT-LLM Disaggregation

```
StormService (trt-llm-model)
  ├── RoleSet[prefill]: pods running trtllm-serve --role context
  ├── RoleSet[decode]:  pods running trtllm-serve --role generation
  └── Orchestrator pod: TRT-LLM disagg coordinator
```

Pooled mode 允许独立扩缩 prefill 和 decode 池。StormService 需要 TRT-LLM 特定注解用于不同角色的并行策略（TP、PP、EP 可在 prefill/decode 间不同）。

#### 步骤 6: 网络基础设施

**KV Cache 跨 Pod 传输**:
- **同节点 pods**: NVLink 或共享内存
- **跨节点 pods**: RDMA（InfiniBand 或 RoCE v2）

**传输性能估算**（70B FP16 KV cache @ 4K tokens ≈ 5GB）:
- 100 Gbps RDMA: ~400ms
- 400 Gbps NDR InfiniBand: ~100ms

**Kubernetes 网络要求**:
- NVIDIA Network Operator 部署 RDMA device plugins + SR-IOV 配置
- Pods 需要 `rdma/rdma_shared_device_a` 资源请求
- `hostNetwork: true` 或 Multus CNI 配置二级 RDMA 网络接口
- GPUDirect RDMA: 需 NVIDIA peer memory kernel module、MOFED 驱动、ConnectX-6+ NIC

### 6.5 替代方案：通过 Dynamo 间接集成

NVIDIA Dynamo 已提供完整的 TRT-LLM disaggregated serving 编排能力。AIBrix 已在 Helm values 中引用 Dynamo 作为支持引擎。

**建议**: 与其从零构建深度 TRT-LLM 集成，不如**利用 Dynamo 作为中间层**:
- AIBrix 管理 Dynamo 的生命周期和流量
- Dynamo 处理 TRT-LLM 的编排、路由、KV cache 传输
- AIBrix 提供 Kubernetes 原生的自动扩缩容、监控、多租户能力

```
AIBrix (K8s 控制面)
    │
    ▼
Dynamo (推理编排)
    │
    ├── Frontend (HTTP)
    ├── Smart Router
    ├── Prefill Workers (TRT-LLM)
    └── Decode Workers (TRT-LLM)
```

---

## 7. 关键结论与建议

### 7.1 核心发现

1. **ai-dynamo 是目前 TRT-LLM disaggregated serving 的最佳编排框架**。它提供完整的端到端解决方案：智能路由、KV cache 管理、NIXL 传输、动态资源调度。

2. **路由是成本最小化问题**。核心公式 `cost = weight × prefill_blocks + decode_blocks` 平衡了 cache 重用（降低 TTFT）和负载均衡（优化 ITL），通过单一可调参数控制。

3. **RadixTree 是核心数据结构**。O(n) 前缀匹配确定哪些 worker 对任何请求有最有用的缓存 blocks。

4. **KV Events 是通信骨干**。Workers 发布 Stored/Removed 事件，Router 消费它们维护所有 worker 的全局 cache 状态同步视图。

5. **Disaggregated serving 使用 pull-based queue 架构**。Decode workers push prefill 请求到全局 NATS queue，Prefill workers pull，NIXL 处理直接 GPU-to-GPU KV 传输。

6. **系统正从集中式基础设施（NATS、etcd）向去中心化 planes（ZMQ、K8s 发现）演进**。

### 7.2 在 AIBrix 中支持 TRT-LLM 的建议

**短期（推荐）**: 利用 Dynamo 作为中间层
- 最小工作量：添加 metrics mapping + Helm 模板
- 利用 Dynamo 已有的成熟 TRT-LLM 编排能力
- 通过 Grove 在 K8s 上编排 Dynamo 组件

**中期**: 原生集成
- 构建 TRT-LLM 引擎编译流水线（Builder Job）
- 扩展 StormService 支持 TRT-LLM 特定注解
- 集成 NIXL 进行 KV cache 跨 Pod 传输

**长期**: 深度优化
- 实现 KV-aware prefix routing（参考 Dynamo 的 RadixTree 方案）
- 支持动态 GPU 资源重分配（Planner 功能）
- 层级 KV cache offloading 到多层存储

### 7.3 关键风险

| 风险 | 影响 | 缓解策略 |
|------|------|----------|
| TRT-LLM Prometheus metrics 不稳定 | 路由决策不准确 | 使用 Triton 作为前端 |
| 引擎编译冷启动（10-30 分钟） | 扩缩容延迟 | 预构建引擎镜像、Builder Job |
| KV cache 格式不兼容 | 无法跨引擎复用 cache | 短期避免跨引擎场景 |
| RDMA 基础设施依赖 | 无 RDMA 则 TCP 传输抵消延迟优势 | 确保 InfiniBand/RoCE 部署 |
| Dynamo 功能重叠 | 重复建设 | 优先借助 Dynamo 而非重新实现 |

---

## 参考资料

### 官方文档
- [NVIDIA Dynamo Technical Blog](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [ai-dynamo/dynamo GitHub](https://github.com/ai-dynamo/dynamo)
- [Dynamo TRT-LLM Backend Docs](https://docs.nvidia.com/dynamo/components/backends/tensor-rt-llm)
- [TRT-LLM Disaggregated Serving Blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [TRT-LLM Disaggregated Serving Docs](https://nvidia.github.io/TensorRT-LLM/1.2.0rc6/features/disagg-serving.html)
- [Grove Deployment Guide](https://docs.nvidia.com/dynamo/latest/kubernetes/grove.html)
- [KV Cache Routing Architecture](https://github.com/ai-dynamo/dynamo/blob/main/docs/architecture/kv_cache_routing.md)
- [KV Cache Aware Routing](https://docs.nvidia.com/dynamo/latest/user-guides/kv-cache-aware-routing)

### AIBrix
- [AIBrix GitHub](https://github.com/vllm-project/aibrix)
- [AIBrix Architecture](https://aibrix.readthedocs.io/latest/designs/architecture.html)
- [AIBrix StormService](https://aibrix.readthedocs.io/latest/designs/aibrix-stormservice.html)
- [AIBrix Paper (arXiv)](https://arxiv.org/html/2504.03648v1)

### 云部署
- [AWS EKS + Dynamo](https://aws.amazon.com/blogs/machine-learning/accelerate-generative-ai-inference-with-nvidia-dynamo-and-amazon-eks/)
- [Azure AKS + Dynamo](https://blog.aks.azure.com/2025/10/24/dynamo-on-aks)
- [Google Cloud + Dynamo](https://cloud.google.com/blog/products/compute/ai-inference-recipe-using-nvidia-dynamo-with-ai-hypercomputer)

### 相关 Issues
- [TRT-LLM Layer-wise KV Transfer (#9212)](https://github.com/NVIDIA/TensorRT-LLM/issues/9212)
- [TRT-LLM Prometheus Metrics Bug (#9678)](https://github.com/NVIDIA/TensorRT-LLM/issues/9678)
