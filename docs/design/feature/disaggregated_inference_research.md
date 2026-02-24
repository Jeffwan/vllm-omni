# Disaggregated Inference Research: Architecture, Multi-Node, and AIBrix KVCache Extension Plan

## Table of Contents
1. [Disaggregated Inference Architecture & GPU Setup](#1-disaggregated-inference-architecture--gpu-setup)
2. [Multi-Host (Cross-Node) Disaggregated Inference](#2-multi-host-cross-node-disaggregated-inference)
3. [Deep Dive: Existing Multi-Node Connectors](#3-deep-dive-existing-multi-node-connectors)
4. [Plan: Extending AIBrix KVCache (PrisKV) to Multi-Node in vLLM-Omni](#4-plan-extending-aibrix-kvcache-priskv-to-multi-node-in-vllm-omni)

---

## 1. Disaggregated Inference Architecture & GPU Setup

### 1.1 What "Disaggregated" Means in vLLM-Omni

Disaggregated inference in vLLM-Omni means **splitting a multi-stage omni-modal pipeline into separate processes**, each potentially running on a different GPU. The system uses E/P/D/G (Encoding/Processing/Decoding/Generation) disaggregation across stages.

For example, a Qwen2.5-Omni model is decomposed into:

```
Stage 0 (Thinker) → Stage 1 (Talker) → Stage 2 (Code2Wav)
   GPU 0               GPU 1               GPU 2
```

Each stage:
- Runs in its **own process** (`runtime.process: true`)
- Has its **own GPU assignment** (`runtime.devices: "0"`, `"1"`, `"2"`)
- Has its **own vLLM engine** with independent scheduler, worker, and memory config
- Communicates with adjacent stages via **OmniConnectors**

### 1.2 Yes, Stages Run on Different GPUs

The YAML stage config explicitly assigns GPUs per stage:

```yaml
# From qwen2_5_omni_multiconnector.yaml
stage_args:
  - stage_id: 0
    runtime:
      devices: "0"          # Thinker on GPU 0
  - stage_id: 1
    runtime:
      devices: "1"          # Talker on GPU 1
  - stage_id: 2
    runtime:
      devices: "2"          # Code2Wav on GPU 2
```

A single stage can also span **multiple GPUs** via tensor parallelism:

```yaml
# From qwen3_omni_moe_multiconnector.yaml (2x H100)
stage_args:
  - stage_id: 0
    runtime:
      devices: "0,1"        # Thinker with TP=2 across 2 GPUs
    engine_args:
      tensor_parallel_size: 2
```

### 1.3 Data Flow Between Stages

The data path is currently **D2H2D** (Device → Host → Device):

```
GPU 0 (Thinker)                           GPU 1 (Talker)
    │                                          ▲
    ├─ .cpu()  ──→  Serialize (msgpack)        │
    │               ──→  Connector.put()       │
    │                        │                 │
    │              [SharedMem / Mooncake]       │
    │                        │                 │
    │               Connector.get()  ──→       │
    │               Deserialize  ──→  .cuda()  │
    │                                          │
    ▼───────── Control Queue (notify) ────────►│
```

Key details:
- **Serialization**: Uses msgpack via `OmniSerializer` — handles `torch.Tensor`, `numpy.ndarray`, `PIL.Image`, `RequestOutput`, etc.
- **Control plane**: Lightweight queue notifications carry metadata (request_id, from/to stage, connector metadata like SHM block names)
- **Data plane**: Heavy payloads go through the connector (SHM blocks or Mooncake store)

### 1.4 Connector Interface

The `OmniConnectorBase` abstraction (`vllm_omni/distributed/omni_connectors/connectors/base.py`):

```python
class OmniConnectorBase(ABC):
    def put(self, from_stage, to_stage, request_id, data)
        -> (success: bool, serialized_size: int, metadata: dict | None)

    def get(self, from_stage, to_stage, request_id, metadata=None)
        -> (data: Any, size: int) | None

    def cleanup(self, request_id) -> None
    def health() -> dict
```

Critical design: `put()` returns **metadata** (e.g., SHM block name) that must be forwarded via the control plane to the consumer, so `get()` knows where to find the data.

---

## 2. Multi-Host (Cross-Node) Disaggregated Inference

### 2.1 Yes, Stages Can Run on Different Hosts

The MooncakeConnector is specifically designed for **cross-node** communication:

```
Node A (Host 10.0.0.1)              Node B (Host 10.0.0.2)
┌─────────────────────┐            ┌─────────────────────┐
│  Stage 0 (Thinker)  │            │  Stage 1 (Talker)   │
│  GPU 0              │            │  GPU 0              │
│                     │            │                     │
│  MooncakeConnector  │            │  MooncakeConnector  │
│  host: 10.0.0.1     │            │  host: 10.0.0.2     │
└────────┬────────────┘            └──────▲──────────────┘
         │                                │
         │         ┌──────────────┐       │
         └────────►│  Mooncake    │───────┘
                   │  Master      │
                   │  10.0.0.1:   │
                   │  50051/8080  │
                   └──────────────┘
```

The connector uses:
- **Mooncake Master** (gRPC on port 50051) for global state management
- **Metadata Server** (HTTP on port 8080) for service discovery / RDMA QP exchange
- **TCP or RDMA** data plane for actual payload transfer
- **Deterministic keying**: `{request_id}/{from_stage}_to_{to_stage}` — no metadata passing needed (unlike SHM)

### 2.2 Current Multi-Node Setup Requirements

1. **Install Mooncake** on all nodes:
   ```bash
   pip install mooncake-transfer-engine
   ```

2. **Start Mooncake Master** on a designated node:
   ```bash
   mooncake_master \
     --rpc_port=50051 \
     --enable_http_metadata_server=true \
     --http_metadata_server_port=8080 \
     --root_fs_dir=./mc_storage/ \
     --cluster_id=mc-local-1
   ```

3. **Configure each node's YAML** with Mooncake connector pointing to the master:
   ```yaml
   runtime:
     connectors:
       mooncake_connector:
         name: MooncakeConnector
         extra:
           host: "<THIS_NODE_IP>"
           metadata_server: "http://<MASTER_IP>:8080/metadata"
           master: "<MASTER_IP>:50051"
           proto: "tcp"    # or "rdma"
   ```

### 2.3 Limitations of Current Multi-Node Support

| Limitation | Detail |
|---|---|
| **D2H2D only** | No direct GPU-to-GPU (D2D) transfer across nodes yet |
| **Only Mooncake** | No alternative distributed backends (Redis, gRPC, etc.) |
| **Single orchestrator** | The Omni orchestrator process runs on one node; stages are child processes or remote via Ray |
| **No KV cache sharing** | The OmniConnector transfers full payloads (hidden states, embeddings), not incremental KV cache blocks |
| **Serialization overhead** | msgpack serialization to CPU, then network transfer, then deserialization — significant for large tensors |

---

## 3. Deep Dive: Existing Multi-Node Connectors

### 3.1 vLLM-Omni OmniConnector System

Two connectors are registered in the factory:

| Connector | Transport | Scope | Use Case |
|---|---|---|---|
| `SharedMemoryConnector` | POSIX SHM / inline | Single node | Default, zero-copy on host |
| `MooncakeConnector` | TCP / RDMA via Mooncake Store | Multi-node | Distributed inference |

**SharedMemoryConnector** (`shm_connector.py`):
- Threshold-based: < 64KB → inline in metadata dict; >= 64KB → POSIX shared memory block
- Metadata passthrough required (SHM block name)
- Auto-configured for any edge without explicit connector
- For Ray backend: threshold set to `sys.maxsize` (everything inline, Ray handles transport)

**MooncakeConnector** (`mooncake_connector.py`):
- Wraps `MooncakeDistributedStore` from the `mooncake` library
- Keying: deterministic `f"{rid}/{from_stage}_to_{to_stage}"`
- Retry on get: 20 attempts × 50ms sleep = 1s timeout
- No explicit delete (GC-based cleanup)
- Supports `ReplicateConfig` with soft pinning for data locality

### 3.2 vLLM Native KV Transfer System (Upstream)

vLLM upstream has a separate, specialized KV cache transfer system for **prefill/decode disaggregation** (different from vLLM-Omni's general-purpose OmniConnector):

**Location**: `vllm.distributed.kv_transfer.kv_connector.v1`

**Available backends** (from vLLM docs):
| Connector | Transport | Notes |
|---|---|---|
| PyNcclConnector | NCCL | GPU-to-GPU, requires shared NCCL group |
| P2PNcclConnector | NCCL P2P | Direct peer-to-peer GPU transfer |
| NixlConnector | NIXL library | Multi-node, RDMA-capable |
| LMCacheConnector | LMCache | External cache store |
| MooncakeConnector | Mooncake | Same Mooncake engine |
| OffloadingConnector | CPU/disk | For KV cache offloading |

These connectors are **KV-cache-specific**: they transfer actual attention key-value tensors between prefill and decode instances, operating at the block/page granularity that vLLM's PagedAttention uses. This is fundamentally different from OmniConnector's general-purpose data transport.

### 3.3 AIBrix KVCache System

AIBrix provides a **tiered KV cache offloading framework** that integrates with vLLM:

**Architecture:**
```
┌──────────────────────────────────────────────────────┐
│                   vLLM Engine                         │
│  ┌──────────┐    ┌──────────┐    ┌────────────────┐  │
│  │ GPU HBM  │ ←→ │ L1 Cache │ ←→ │   L2 Cache     │  │
│  │ KV Cache │    │ (DRAM)   │    │ (Distributed)  │  │
│  └──────────┘    └──────────┘    └───────┬────────┘  │
│                                          │           │
└──────────────────────────────────────────┼───────────┘
                                           │
                    ┌──────────────────────▼─────────────┐
                    │     L2 Backend (PrisKV / EIC /     │
                    │     InfiniStore)                    │
                    │     ┌────────────────────────────┐ │
                    │     │  Hot Tier: In-memory hash   │ │
                    │     │  tables + slab allocators   │ │
                    │     ├────────────────────────────┤ │
                    │     │  Cold Tier: Local FS /      │ │
                    │     │  Redis-compatible           │ │
                    │     └────────────────────────────┘ │
                    │     Transport: TCP / RDMA / GDR    │
                    └────────────────────────────────────┘
```

**Connector Interface Features** (`ConnectorFeature`):
- `mput/mget`: Batch put/get for multiple KV blocks
- `prefetch`: Pre-load keys into faster storage tiers
- `rdma`: Remote Direct Memory Access support
- `gdr_put/gdr_get`: GPU Direct RDMA — transfer without CPU intermediation

**Existing L2 Connectors**:
- **PrisKV**: High-performance colocated tiered KV cache (default port 6379, TCP-based, optional mput/mget)
- **InfiniStore**: RDMA-native, GDR-capable connector
- **EIC** (ByteDance): Low-latency, multi-tier caching

**Key Performance Numbers**:
- Framework overhead < 3% (70B model, TP=8)
- 89% reduction in TTFT with EIC under high concurrency
- 50% throughput increase with distributed KV cache

---

## 4. Plan: Extending AIBrix KVCache (PrisKV) to Multi-Node in vLLM-Omni

### 4.1 Goal

Create an `AIBrixKVCacheConnector` (PrisKV-backed) for vLLM-Omni's OmniConnector system, enabling:
1. **Multi-node stage disaggregation** using PrisKV as the data plane
2. **KV cache reuse** across engines/stages via AIBrix's L2 distributed cache
3. **RDMA support** for high-performance cross-node transfer
4. **Tiered caching** (hot/cold) for cost-efficient large-scale deployments

### 4.2 Architecture

```
vLLM-Omni Orchestrator
        │
        ├─ Stage 0 (Node A, GPU 0)
        │   └─ AIBrixKVCacheConnector.put() ──→ PrisKV Cluster
        │
        ├─ Stage 1 (Node B, GPU 0)
        │   └─ AIBrixKVCacheConnector.get() ←── PrisKV Cluster
        │
        └─ Stage 2 (Node C, GPU 0)
            └─ AIBrixKVCacheConnector.get() ←── PrisKV Cluster
```

### 4.3 Implementation Plan

#### Phase 1: Core Connector Implementation

**Step 1: Create `aibrix_kvcache_connector.py`**

Location: `vllm_omni/distributed/omni_connectors/connectors/aibrix_kvcache_connector.py`

```python
class AIBrixKVCacheConnector(OmniConnectorBase):
    """
    OmniConnector backed by AIBrix KVCache (PrisKV).
    Supports TCP and optional RDMA transport for multi-node
    stage disaggregation.
    """

    def __init__(self, config: dict[str, Any]):
        self.host = config.get("host", "127.0.0.1")
        self.port = config.get("port", 6379)
        self.use_rdma = config.get("use_rdma", False)
        self.use_mput_mget = config.get("use_mput_mget", True)
        self.password = config.get("password", None)
        self.pool_size = config.get("pool_size", 8)
        self.timeout_ms = config.get("timeout_ms", 1000)
        self.key_prefix = config.get("key_prefix", "omni")
        self.ttl_seconds = config.get("ttl_seconds", 300)
        # ... initialize PrisKV client

    def _make_key(self, rid, from_stage, to_stage):
        return f"{self.key_prefix}/{rid}/{from_stage}_to_{to_stage}"

    def put(self, from_stage, to_stage, request_id, data):
        payload = self.serialize_obj(data)
        key = self._make_key(request_id, from_stage, to_stage)
        # Use PrisKV client to store
        # Optional: use mput for batch efficiency
        # Optional: set TTL for auto-cleanup
        return True, len(payload), None  # No metadata needed (key-based)

    def get(self, from_stage, to_stage, request_id, metadata=None):
        key = self._make_key(request_id, from_stage, to_stage)
        # Poll with retry (similar to MooncakeConnector pattern)
        # Use PrisKV client to retrieve
        raw = self.client.get(key)
        return self.deserialize_obj(raw), len(raw)

    def cleanup(self, request_id):
        # Delete all keys matching {prefix}/{request_id}/*
        pass

    def health(self):
        # Ping PrisKV, return metrics
        pass
```

**Step 2: Register in Factory**

In `vllm_omni/distributed/omni_connectors/factory.py`:

```python
def _create_aibrix_kvcache_connector(config):
    from .connectors.aibrix_kvcache_connector import AIBrixKVCacheConnector
    return AIBrixKVCacheConnector(config)

OmniConnectorFactory.register_connector(
    "AIBrixKVCacheConnector", _create_aibrix_kvcache_connector
)
```

**Step 3: YAML Configuration**

```yaml
runtime:
  connectors:
    aibrix_connector:
      name: AIBrixKVCacheConnector
      extra:
        host: "10.0.0.5"           # PrisKV server address
        port: 6379
        use_rdma: false             # Enable when RDMA available
        use_mput_mget: true         # Batch operations
        pool_size: 8
        timeout_ms: 1000
        key_prefix: "omni"
        ttl_seconds: 300            # Auto-expire keys
```

#### Phase 2: Optimized Tensor Transfer

**Step 4: GPU Direct RDMA (GDR) Support**

If AIBrix connector supports `gdr_put`/`gdr_get`, bypass D2H2D entirely:

```python
def put_tensor_direct(self, from_stage, to_stage, request_id, tensor):
    """Direct GPU → PrisKV transfer via GDR (skip CPU)."""
    if self.features.gdr_put and tensor.is_cuda:
        key = self._make_key(request_id, from_stage, to_stage)
        # Use GDR put: GPU memory → RDMA → PrisKV
        self.client.gdr_put(key, tensor.data_ptr(), tensor.nbytes)
        return True, tensor.nbytes, {"gdr": True, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
    else:
        # Fall back to D2H2D
        return self.put(from_stage, to_stage, request_id, tensor)
```

This requires extending `OmniConnectorBase` with an optional `put_tensor`/`get_tensor` method for zero-copy GPU transfers.

**Step 5: Batch Transfer with mput/mget**

When a stage produces multiple outputs (e.g., KV cache blocks), use batch operations:

```python
def put_batch(self, from_stage, to_stage, request_id, data_dict):
    """Batch put multiple key-value pairs."""
    keys = []
    values = []
    for sub_key, data in data_dict.items():
        keys.append(f"{self._make_key(request_id, from_stage, to_stage)}/{sub_key}")
        values.append(self.serialize_obj(data))
    self.client.mput(keys, values)
```

#### Phase 3: Tiered Caching Integration

**Step 6: L1 + L2 Cache Awareness**

Leverage AIBrix's tiered caching in the connector:

```python
class AIBrixKVCacheConnector(OmniConnectorBase):
    def __init__(self, config):
        # L1 config (local DRAM cache)
        self.l1_enabled = config.get("l1_enabled", True)
        self.l1_capacity_gb = config.get("l1_capacity_gb", 10)
        self.l1_eviction_policy = config.get("l1_eviction_policy", "s3fifo")

        # L2 config (distributed PrisKV)
        self.l2_host = config.get("host", "127.0.0.1")
        self.l2_port = config.get("port", 6379)

    def get(self, from_stage, to_stage, request_id, metadata=None):
        key = self._make_key(request_id, from_stage, to_stage)
        # Try L1 first (local DRAM)
        result = self.l1_cache.get(key)
        if result:
            return result
        # Fall back to L2 (distributed PrisKV)
        result = self.l2_client.get(key)
        if result:
            self.l1_cache.put(key, result)  # Promote to L1
        return result
```

#### Phase 4: Multi-Node Orchestration

**Step 7: Extend Orchestrator for Cross-Node Stage Management**

Currently the orchestrator runs on a single node. For true multi-node:

Option A: **Ray-based orchestration** (already partially supported):
```yaml
# Each stage can target a different Ray node
stage_args:
  - stage_id: 0
    runtime:
      devices: "0"
      ray_node: "node-a"    # Future: node affinity
  - stage_id: 1
    runtime:
      devices: "0"
      ray_node: "node-b"
```

Option B: **Independent stage processes** with shared PrisKV:
- Each node runs its own stage process
- PrisKV provides the shared data plane
- A lightweight control plane (e.g., Redis pub/sub, gRPC) handles notifications

**Step 8: KV Event Synchronization**

Integrate with AIBrix's KV event sync to broadcast cache state across nodes:
- When a stage completes and stores output in PrisKV, emit a KV event
- Downstream stage nodes subscribe to these events
- Enables cache-aware scheduling and prefetching

### 4.4 Implementation Checklist

| # | Task | Priority | Complexity |
|---|------|----------|------------|
| 1 | Implement `AIBrixKVCacheConnector` with basic TCP `put`/`get` | P0 | Low |
| 2 | Register in `OmniConnectorFactory` | P0 | Low |
| 3 | Add YAML config support and tests | P0 | Low |
| 4 | Add retry logic with configurable timeout | P0 | Low |
| 5 | Add `cleanup()` with TTL-based and explicit key deletion | P1 | Low |
| 6 | Add `health()` with PrisKV ping and metrics | P1 | Low |
| 7 | Add `mput`/`mget` batch operations | P1 | Medium |
| 8 | Add RDMA transport option | P1 | Medium |
| 9 | Add GDR put/get for zero-copy GPU transfer | P2 | High |
| 10 | Add L1 (DRAM) + L2 (PrisKV) tiered caching | P2 | Medium |
| 11 | Add KV event synchronization for cross-node cache visibility | P2 | High |
| 12 | Extend orchestrator for multi-node stage placement (Ray) | P3 | High |
| 13 | Benchmark against MooncakeConnector (latency, throughput) | P1 | Medium |
| 14 | Add integration tests with multi-stage pipeline | P1 | Medium |

### 4.5 File Changes Summary

```
New files:
  vllm_omni/distributed/omni_connectors/connectors/aibrix_kvcache_connector.py

Modified files:
  vllm_omni/distributed/omni_connectors/factory.py  (register new connector)

New test files:
  tests/distributed/omni_connectors/test_aibrix_kvcache_connector.py

New example configs:
  vllm_omni/model_executor/stage_configs/qwen2_5_omni_aibrix.yaml
```

### 4.6 Key Design Decisions

1. **Deterministic keying** (like MooncakeConnector): Use `{prefix}/{request_id}/{from}_to_{to}` keys so `get()` doesn't require metadata passthrough. This simplifies cross-node operation since the control plane only needs to carry the request_id.

2. **TTL-based cleanup**: Set TTL on keys rather than relying on explicit cleanup, providing automatic resource reclamation even if stages crash.

3. **Connection pooling**: Use connection pools (configurable `pool_size`) to handle concurrent requests across multiple stages.

4. **Gradual RDMA adoption**: Start with TCP, add RDMA as optional, then GDR as advanced. Each step provides value independently.

5. **Compatibility with AIBrix V1 connector interface**: Align with AIBrix's `Connector` interface (`from_envs`, `open/close`, `get/put/delete`, `exists`, `mget/mput`, `prefetch`) so the implementation can eventually be upstreamed into AIBrix.

### 4.7 Dependencies

```
# Core (required)
aibrix-kvcache  # or specific PrisKV client library

# Optional (for RDMA/GDR)
# RDMA drivers and libibverbs
# CUDA toolkit (for GDR)
```

---

## References

- [vLLM-Omni Disaggregated Inference Design](../feature/disaggregated_inference.md)
- [AIBrix KVCache Offloading Framework](https://aibrix.readthedocs.io/latest/designs/aibrix-kvcache-offloading-framework.html)
- [AIBrix v0.4.0 Release Notes](https://aibrix.github.io/posts/2025-08-04-v0.4.0-release/)
- [AIBrix GitHub Repository](https://github.com/vllm-project/aibrix)
- [AIBrix Paper (arXiv:2504.03648)](https://arxiv.org/abs/2504.03648)
- [Mooncake Repository](https://github.com/kvcache-ai/Mooncake)
