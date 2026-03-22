# BAGEL Multi-Node Deployment: Architecture Deep Dive

This document explains how BAGEL's multi-stage pipeline (Thinker + DiT) is distributed across nodes in vllm-omni, covering the role of Mooncake connectors, stage allocation, Ray, and the roadmap toward full disaggregation.

## Table of Contents

- [1. High-Level Architecture](#1-high-level-architecture)
- [2. Mooncake: Cross-Node KV Cache Transfer](#2-mooncake-cross-node-kv-cache-transfer)
- [3. Stage Allocation Across Nodes](#3-stage-allocation-across-nodes)
- [4. Ray: Current Status and Roadmap](#4-ray-current-status-and-roadmap)
- [5. Full Disaggregation Roadmap (EPDG)](#5-full-disaggregation-roadmap-epdg)
- [6. Gap Analysis: What's Missing for True Multi-Node](#6-gap-analysis-whats-missing-for-true-multi-node)
- [7. Approach A: Minimal Changes to Make It Work](#7-approach-a-minimal-changes-to-make-it-work)
- [8. Approach B: Kubernetes-Native Orchestration](#8-approach-b-kubernetes-native-orchestration)
- [9. In-Progress Work](#9-in-progress-work-as-of-2026-03-20)
- [10. Related Documents](#10-related-documents)

---

## 1. High-Level Architecture

BAGEL is an "AR-main + DiT" model — an autoregressive Thinker (Stage 0) generates text tokens and KV cache, then a Diffusion Transformer (Stage 1) consumes the KV cache to produce images.

```
                          ┌─────────────────────────┐
                          │   Client Request         │
                          │   "A cute cat" text2img  │
                          └───────────┬─────────────┘
                                      │
                                      ▼
                     ┌────────────────────────────────┐
                     │        API Server (:8000)       │
                     │        AsyncOmniEngine          │
                     └────────────┬───────────────────┘
                                  │
                                  ▼
                     ┌────────────────────────────────┐
                     │     Orchestrator (bg thread)    │
                     │     Routes requests between     │
                     │     stages via janus queues     │
                     └──┬──────────────────────────┬──┘
                        │                          │
            ┌───────────▼──────────┐   ┌──────────▼───────────┐
            │  Stage 0: Thinker    │   │  Stage 1: DiT        │
            │  (LLM - Qwen2.5)    │   │  (Diffusion)         │
            │                      │   │                      │
            │  1. Prefill prompt   │   │  4. Receive KV cache │
            │  2. Generate tokens  │   │     from Mooncake    │
            │     (thinking +      │   │  5. Run diffusion    │
            │      image tokens)   │   │     denoising steps  │
            │  3. Extract KV cache │   │  6. Output latents   │
            │     → Mooncake put() │   │     → decode to PNG  │
            │                      │   │                      │
            │  GPU: ~15GB + KV     │   │  GPU: ~26.5GB        │
            └──────────┬───────────┘   └──────────┬───────────┘
                       │                          ▲
                       │    ┌──────────────┐      │
                       └───▶│   Mooncake   │──────┘
                            │   Master     │
                            │              │
                            │  KV Store    │
                            │  :50051 RPC  │
                            │  :8080 HTTP  │
                            └──────────────┘
```

### Execution Flow

| Step | Component | Action |
|:-----|:----------|:-------|
| 1 | API Server | Receives chat completion request, routes to Orchestrator |
| 2 | Orchestrator | Submits request to Stage 0 (Thinker) |
| 3 | Stage 0 | Prefills prompt, generates tokens autoregressively |
| 4 | Stage 0 | On prefill completion, extracts KV cache → `connector.put()` to Mooncake |
| 5 | Orchestrator | Detects Stage 0 output, forwards request metadata to Stage 1 |
| 6 | Stage 1 | Calls `connector.get()` to retrieve KV cache from Mooncake |
| 7 | Stage 1 | Runs diffusion denoising with KV cache as conditioning |
| 8 | Orchestrator | Collects Stage 1 output (image), returns to API Server |

---

## 2. Mooncake: Cross-Node KV Cache Transfer

Mooncake is a distributed key-value store that acts as the **data plane** between stages. It solves one specific problem: moving the KV cache that Stage 0 produces to Stage 1, which may be on a different physical node.

### 2.1 Write Path (Stage 0 → Mooncake)

Code path: `kv_transfer_manager.py` → `mooncake_store_connector.py`

```
GPU KV blocks (per-layer, per-head tensors)
     │
     ▼
_extract_kv_cache()                 ← kv_transfer_manager.py:224-301
  Reads GPU blocks, reshapes to [seq_len, n_heads, head_dim], moves to CPU
     │
     ▼
_transfer_kv_cache()                ← kv_transfer_manager.py:303-323
  Wraps as KVCacheTransferData with request_id + metadata
     │
     ▼
connector.put(                      ← mooncake_store_connector.py:73-98
  from_stage="0",
  to_stage="1",
  key="omni_0_to_1_kv_cache_{req_id}",
  data=serialized_bytes
)
     │
     ▼
MooncakeDistributedStore.put()      ← Mooncake library
  Stores bytes under key "{req_key}@0_1" with retry (20x, 50ms backoff)
```

The trigger is configured in the stage YAML:

```yaml
# bagel_multiconnector.yaml, Stage 0
omni_kv_config:
  need_send_cache: true
  kv_transfer_criteria:
    type: prefill_finished    # extract KV after prefill completes
```

### 2.2 Read Path (Mooncake → Stage 1)

Code path: `kv_transfer_manager.py` → `mooncake_store_connector.py`

```
connector.get(                      ← mooncake_store_connector.py:100-148
  from_stage="0",
  to_stage="1",
  key="omni_0_to_1_kv_cache_{req_id}"
)
     │  Polls Mooncake store (20 retries × 50ms = ~1s timeout)
     ▼
deserialize → move tensors to GPU   ← kv_transfer_manager.py:363-436
     │
     ▼
apply_kv_cache_to_request()         ← kv_transfer_manager.py:438-460
  Attaches KV as request.sampling_params.past_key_values
     │
     ▼
DiT forward pass uses cached KV as conditioning
```

### 2.3 Connector Key Format

```
{request_key}@{from_stage}_{to_stage}
```

Example: `omni_0_to_1_kv_cache_req-abc123@0_1`

Both sides compute the same key independently — no metadata handoff needed. The store itself is the rendezvous point.

### 2.4 Connector Choices

| Connector | Transport | Best For | Status |
|:----------|:----------|:---------|:-------|
| `SharedMemoryConnector` | `/dev/shm` | Same-node stages | Default, auto-configured |
| `MooncakeStoreConnector` | TCP via Mooncake Store | Multi-node, simple setup | Active in `bagel_multiconnector.yaml` |
| `MooncakeTransferEngineConnector` | RDMA / TCP direct | Multi-node, high throughput | Config templated, next PR |

The RDMA connector (`MooncakeTransferEngineConnector`) supports zero-copy fast paths for tensors and managed memory pools. It is the P0 target per the [Q1 2026 roadmap](https://github.com/vllm-project/vllm-omni/issues/1192).

### 2.5 Mooncake Master: Two Services, One Process

The `mooncake_master` process exposes **two separate services** on different ports. Both are passed to `MooncakeDistributedStore.setup()` (`mooncake_store_connector.py:58-59`):

```
┌────────────────────── mooncake_master process ──────────────────────┐
│                                                                      │
│   ┌───────────────────────────┐   ┌───────────────────────────────┐ │
│   │  gRPC Master (:50051)     │   │  HTTP Metadata Server (:8080) │ │
│   │                           │   │                               │ │
│   │  • Cluster management     │   │  • Key → node location lookup │ │
│   │  • Store lifecycle        │   │  • Service discovery          │ │
│   │  • Replication config     │   │  • REST API for readers       │ │
│   │  • Write coordination     │   │  • "Where does this key live?"│ │
│   └───────────────────────────┘   └───────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

| Config Key | Example Value | Protocol | Purpose |
|:-----------|:-------------|:---------|:--------|
| `master` | `10.90.67.86:50051` | gRPC | **Control plane** — cluster membership, store lifecycle, write coordination |
| `metadata_server` | `http://10.90.67.86:8080/metadata` | HTTP | **Data plane routing** — key-to-node lookup so `get()` knows which node holds a key |

**How they work together during a KV transfer:**

```
Stage 0 calls put("kv_cache_req123", data):
  1. store.put() contacts Master (:50051 gRPC) to register the key
  2. Master records: "kv_cache_req123 lives on node 10.x.x.1"
  3. Data stored in local memory segment

Stage 1 calls get("kv_cache_req123"):
  1. store.get() queries Metadata Server (:8080 HTTP):
     GET http://10.90.67.86:8080/metadata?key=kv_cache_req123
     → Response: "node 10.x.x.1, segment offset 0xABCD"
  2. Fetches data from that node over TCP/RDMA
  3. Returns deserialized object
```

The split allows the read path (metadata lookups) to scale independently from the write/control path (cluster management). Both Stage 0 and Stage 1 connect to the same master. For single-node testing, `127.0.0.1` works. For multi-node, use the orchestrator node's actual IP.

---

## 3. Stage Allocation Across Nodes

### 3.1 Current Architecture: Single-Process All-Stage

Today, `AsyncOmniEngine` loads and runs **all stages** from the YAML config in a single process:

```
┌───────────── Single Process (AsyncOmniEngine) ──────────────┐
│                                                              │
│  Orchestrator (background thread)                            │
│    ├── StageEngineCoreClient[0] ←→ vLLM EngineCore[0]       │
│    │     (ZMQ IPC sockets)          (subprocess, GPU 0)      │
│    └── StageEngineCoreClient[1] ←→ EngineCore[1]            │
│          (ZMQ IPC sockets)          (subprocess, GPU 0/1)    │
│                                                              │
│  Data transfer via connectors:                               │
│    SharedMemoryConnector (same node)                         │
│    MooncakeStoreConnector (cross-node KV cache)              │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 CLI Arguments for Multi-Node

The multi-node CLI arguments are defined in `serve.py:114-178`:

```bash
# Node A: Stage 0 (Thinker / Orchestrator)
vllm serve ByteDance-Seed/BAGEL-7B-MoT --omni \
    --port 8000 \
    --stage-configs-path bagel_multiconnector.yaml \
    --stage-id 0 \
    -oma <ORCHESTRATOR_IP> -omp 8091

# Node B: Stage 1 (DiT, headless)
vllm serve ByteDance-Seed/BAGEL-7B-MoT --omni \
    --stage-configs-path bagel_multiconnector.yaml \
    --stage-id 1 --headless \
    -oma <ORCHESTRATOR_IP> -omp 8091
```

**What each argument means:**

| Argument | Full Name | Purpose |
|:---------|:----------|:--------|
| `--stage-id <N>` | Stage ID | Tells this process to run **only** stage N from the YAML config (0 = Thinker, 1 = DiT) |
| `-oma <IP>` | `--omni-master-address` | IP of the **orchestrator node** (Stage 0). Stage 1 uses this to connect back to Stage 0 for task coordination |
| `-omp <port>` | `--omni-master-port` | Port on the orchestrator that Stage 1 connects to for receiving work assignments |
| `--headless` | Headless mode | Run **without** the API server — worker-only mode. Only Stage 0 exposes the HTTP API; Stage 1 just processes tasks and returns results |

**The intended design:**

```
  Client → HTTP :8000
               │
  ┌────────────▼─────────────┐        ┌──────────────────────────┐
  │  Node A (Stage 0)        │        │  Node B (Stage 1)        │
  │                          │        │                          │
  │  API Server (:8000)      │        │  NO API Server           │
  │  Orchestrator            │◄──────▶│  (--headless)            │
  │  Thinker Engine          │ -oma   │  DiT Engine              │
  │                          │ -omp   │                          │
  │  Listens on -omp (:8091) │ :8091  │  Connects to -oma:-omp  │
  └──────────────────────────┘        └──────────────────────────┘
          │                                     ▲
          │          ┌──────────────┐            │
          └─────────▶│   Mooncake   │────────────┘
                     │  (KV data)   │
                     └──────────────┘
```

Stage 0 is the "master" — it owns the API server and the orchestrator. Stage 1 is a "worker" — it connects to Stage 0's `-omp` port to receive work, and uses Mooncake to fetch KV cache. The client only talks to Stage 0.

### 3.3 Implementation Status

| Feature | CLI Arg | Status |
|:--------|:--------|:-------|
| Stage selection | `--stage-id <N>` | Parsed and validated, but **not wired** to filter stages in `AsyncOmniEngine` |
| Headless mode | `--headless` | **Deprecated** — raises `RuntimeError` (`serve.py:375-391`) |
| Master address | `-oma <IP>` | Parsed, placeholder for future cross-node coordination |
| Master port | `-omp <port>` | Parsed, placeholder for future cross-node coordination |

The `--stage-id` argument requires `-oma` and `-omp` (validated in `serve.py:66-67`), but `_resolve_stage_configs()` still loads **all** stages regardless. The arguments flow through `AsyncOmni` → `OmniBase` → `AsyncOmniEngine` as kwargs, but are never consumed.

### 3.4 Stage Discovery Infrastructure (Built, Not Activated)

An `OmniCoordinator` (ZMQ-based) exists in `vllm_omni/distributed/omni_coordinator/`:

```
┌──────────────────────────────────────────────────────┐
│  OmniCoordinator (ZMQ ROUTER + PUB)                  │
│                                                      │
│  ┌─────────────────────┐  ┌─────────────────────┐   │
│  │ OmniCoordClient     │  │ OmniCoordClient     │   │
│  │   (Stage 0)         │  │   (Stage 1)         │   │
│  │   Sends heartbeats  │  │   Sends heartbeats  │   │
│  │   every 5s          │  │   every 5s          │   │
│  └─────────────────────┘  └─────────────────────┘   │
│                                                      │
│  Message: {stage_id, status, input_addr,             │
│            output_addr, queue_length}                 │
└──────────────────────────────────────────────────────┘
```

This infrastructure is ready but **not called from `AsyncOmniEngine`** yet.

### 3.5 What Works Today for Cross-Node

Even without per-node stage isolation, you can achieve cross-node KV cache transfer:

1. Run `AsyncOmniEngine` on a single node with both stages
2. Use `MooncakeStoreConnector` with the Mooncake Master IP pointing to a remote node
3. The KV cache is transferred over TCP/RDMA to the remote store, making it accessible to any node that queries the same key

True per-node disaggregation (Stage 0 on Node A, Stage 1 on Node B as independent processes) is the next step — see [Section 5](#5-full-disaggregation-roadmap-epdg).

---

## 4. Ray: Current Status and Roadmap

### 4.1 Current: Multiprocessing Backend (No Ray)

```
┌──────────────────────────────────────────────────┐
│  Active Execution Backend                        │
│                                                  │
│  AsyncOmniEngine                                 │
│    → Orchestrator thread                         │
│    → StageEngineCoreClient (inherits AsyncMPClient)│
│    → launch_core_engines() (subprocess)          │
│    → ZMQ IPC sockets                             │
│    → File-based GPU device locks                 │
│                                                  │
│  distributed_executor_backend: "mp"              │
└──────────────────────────────────────────────────┘
```

Ray is **not used** in the current BAGEL deployment. All stage management uses:

- **vLLM's `AsyncMPClient`** for subprocess management
- **ZMQ sockets** for control-plane communication
- **File locks** for GPU device allocation
- **Mooncake connectors** for data-plane (KV cache) transfer

### 4.2 Ray Utilities (Prepared, Not Called)

`vllm_omni/distributed/ray_utils/utils.py` provides:

| Function | Purpose | Called? |
|:---------|:--------|:-------|
| `initialize_ray_cluster(address)` | Connect to Ray cluster | No |
| `create_placement_group(n_stages)` | Allocate GPU/CPU bundles per stage | No |
| `start_ray_actor(fn, pg, idx)` | Spawn `@ray.remote(num_gpus=1)` actor | No |
| `is_ray_task_alive(ref)` / `get_ray_task_error(ref)` | Monitor actor health | No |

These can be activated with `--worker-backend ray`, but this code path is not the default.

### 4.3 Roadmap: Native Worker Actor Pattern

Per [RFC #1192](https://github.com/vllm-project/vllm-omni/issues/1192), the direction is:

> **Remove the `OmniStage` Ray actor wrapper.** Currently, `OmniStage` and the worker must stay on the same node, which limits scaling (e.g., TP>8). Refactor to let the Worker act as the Ray Actor — consistent with vLLM main repo — enabling a single stage to span multiple GPUs/nodes.

```
  Current (OmniStage wrapper)              Target (Native Worker Actor)
  ┌──────────────────────┐                ┌──────────────────────────┐
  │ OmniStage (Ray Actor)│                │ Worker IS the Ray Actor  │
  │   ┌────────────────┐ │                │                          │
  │   │ Worker         │ │                │  Can span multiple GPUs  │
  │   │ (same node)    │ │                │  Can span multiple nodes │
  │   └────────────────┘ │                │  TP > 8 supported        │
  └──────────────────────┘                └──────────────────────────┘
  Limited: stage + worker                  Flexible: worker directly
  must share one node                      scheduled by Ray
```

---

## 5. Full Disaggregation Roadmap (EPDG)

The Q1 2026 roadmap ([RFC #1192](https://github.com/vllm-project/vllm-omni/issues/1192)) targets **EPDG disaggregation**: Encoder, Prefill, Decode, Generate as independently scalable stages.

### 5.1 Priority Matrix

| Priority | Task | Description | Impact on BAGEL |
|:---------|:-----|:------------|:----------------|
| **P0** | Mooncake RDMA | Full RDMA support via `MooncakeTransferEngineConnector` | Cross-node KV transfer with minimal latency |
| **P0** | Async Transfer | Non-blocking `put()`/`get()` overlapping compute | Hide transfer latency during generation |
| **P0** | PD Separation | Prefill-Decode disaggregation (Qwen Omni) | Architecture pattern applicable to BAGEL Thinker |
| **P0** | Bagel Optimization | Cross-node RDMA KV cache for BAGEL specifically | Direct improvement |
| **P1** | Native Worker Actor | Remove `OmniStage` wrapper, enable TP>8 | Unlocks true multi-node stages |
| **P1** | Communication Refactor | Move from stage level → model runner level | Cleaner separation of concerns |
| **P1** | E→P Separation | Decouple encoders from prefill | Offload vision/audio to separate workers |

### 5.2 Target Architecture

```
┌─────────── Node A ──────────────┐  ┌─────────── Node B ──────────────┐
│                                  │  │                                  │
│  ┌────────────┐  ┌────────────┐ │  │  ┌────────────┐                 │
│  │ Encoder    │  │ Prefill    │ │  │  │ Decode     │                 │
│  │ (Vision)   │──│ (Thinker)  │─│──│──│ (Thinker)  │                 │
│  │            │  │            │ │  │  │            │                 │
│  └────────────┘  └────────────┘ │  │  └─────┬──────┘                 │
│                                  │  │        │                        │
│                                  │  │  ┌─────▼──────┐                │
│                                  │  │  │ Generate   │                │
│                                  │  │  │ (DiT)      │                │
│                                  │  │  │ TP=2, GPU  │                │
│                                  │  │  │ 0,1        │                │
│                                  │  │  └────────────┘                │
│                                  │  │                                 │
└──────────────────────────────────┘  └─────────────────────────────────┘
         │                                         ▲
         │         ┌──────────────┐                │
         └────────▶│   Mooncake   │────────────────┘
                   │   (RDMA)     │
                   └──────────────┘
```

### 5.3 Key Architectural Changes

1. **Communication moves to model runner level** (from stage/scheduler level), aligning with vLLM's scheduler/model-runner split: model runner exchanges data, scheduler manages request scheduling.

2. **Heterogeneous rank settings** between connected stages — e.g., Stage 0 at TP=1, Stage 1 at TP=2 with different GPU types.

3. **PD separation leverages vLLM's native Mooncake-based KV connector** (`vllm.distributed.kv_transfer`) rather than reimplementing it, ensuring compatibility with upstream optimizations.

---

## 6. Gap Analysis: What's Missing for True Multi-Node

This section identifies the exact code gaps between the current single-process architecture and true multi-node deployment (Stage 0 on Node A, Stage 1 on Node B as separate processes).

### 6.1 Gap Overview

```
  What EXISTS today                     What's MISSING
  ─────────────────                     ──────────────
  ✅ CLI args parsed                    ❌ --stage-id doesn't filter stages
  ✅ Mooncake KV transfer works         ❌ Control plane is local-only
  ✅ OmniCoordinator built              ❌ Not wired into Orchestrator
  ✅ Connector abstraction              ❌ No RemoteStageClient
  ✅ Stage config YAML                  ❌ --headless deprecated/broken
```

### 6.2 Gap 1: Stage Config Filtering (P0)

**File:** `async_omni_engine.py:857-904`

`_resolve_stage_configs()` loads **all** stages from YAML regardless of `--stage-id`. The `stage_id` kwarg flows all the way from CLI → `AsyncOmniEngine` but is never consumed.

**What to change:** After `load_and_resolve_stage_configs()` returns, filter by `kwargs.get("stage_id")`:

```python
# Pseudocode for the fix
config_path, stage_configs = load_and_resolve_stage_configs(...)
stage_id = kwargs.get("stage_id")
if stage_id is not None:
    stage_configs = [c for c in stage_configs if c.stage_id == stage_id]
```

**Effort:** Small. But everything else depends on this.

### 6.3 Gap 2: Orchestrator Assumes All Stages Are Local (P0)

**File:** `orchestrator.py:108-531`

The Orchestrator holds all stages in `self.stage_clients` (a local list) and calls methods directly:

```python
# orchestrator.py:443-531 - _forward_to_next_stage()
next_client = self.stage_clients[next_stage_id]  # Direct array access
await next_client.add_request_async(request)       # Local method call
```

If Stage 1 is on another node, `self.stage_clients[1]` doesn't exist on Node A.

**What to change:** Introduce a dispatch layer:
- Local stage → current direct call
- Remote stage → network call via `RemoteStageEngineCoreClient`

The Orchestrator on Node A needs a **proxy client** for Stage 1 that forwards requests over the network.

### 6.4 Gap 3: No Remote Stage Client (P0)

**File:** `stage_engine_core_client.py:25-169`

`StageEngineCoreClient` inherits from vLLM's `AsyncMPClient` — hardcoded to local subprocess + ZMQ IPC. There's no network-capable variant.

**What to create:** A `RemoteStageEngineCoreClient` with the same interface:

```python
class RemoteStageEngineCoreClient:
    """Proxy to a stage running on a remote node."""

    def __init__(self, remote_addr: str, remote_port: int, stage_id: int):
        self.zmq_socket = ...  # Connect to remote stage's ZMQ endpoint

    async def add_request_async(self, request):
        # Serialize request → send over network
        ...

    async def get_output_async(self):
        # Poll remote stage for outputs
        ...
```

**Protocol choice:**
- ZMQ (aligned with existing codebase, low overhead)
- gRPC (type-safe, streaming, but adds dependency)

### 6.5 Gap 4: Request Forwarding Is In-Process Only (P1)

**File:** `orchestrator.py:443-531`

`_forward_to_next_stage()` calls `next_client.process_engine_inputs()` and `next_client.add_request_async()` — both are synchronous local calls. For remote stages, the request (token IDs, embeddings, sampling params) must be serialized and sent over the network.

The data itself is small (token IDs + metadata), so latency impact is ~1-5ms per hop — acceptable since DiT compute is much longer.

### 6.6 Gap 5: Output Collection Is Local Polling (P1)

**File:** `orchestrator.py:217-299`

```python
# _orchestration_loop() polls every stage locally
for stage_id in range(self.num_stages):
    raw_outputs = await asyncio.wait_for(
        self._poll_stage_raw(stage_id), timeout=0.001
    )
```

For remote stages, 1ms polling over network is wasteful. Options:
- **Push model:** Remote stage pushes outputs to Orchestrator via ZMQ PUB/SUB
- **Long polling:** Remote stage holds connection until output is ready

### 6.7 Gap 6: Headless Mode Deprecated (P1)

**File:** `serve.py:375-391`

`--headless` raises `RuntimeError`. Worker nodes need a way to start without the API server.

**What to change:** Resurrect headless as "worker mode":
```python
if args.stage_id is not None and args.omni_master_address:
    # Worker mode: start only the assigned stage, connect to master
    run_worker(args)
else:
    # Master mode: start API server + orchestrator
    omni_run_server(args)
```

### 6.8 Gap 7: OmniCoordinator Not Wired In (P1)

**File:** `omni_coordinator/omni_coordinator.py:19-185`

The coordinator is fully implemented (ZMQ ROUTER/PUB, heartbeats, instance tracking) but **never instantiated** from `AsyncOmniEngine`. It's the right infrastructure for cross-node discovery.

**What to wire:**
- Master node: start `OmniCoordinator`, bind to `-omp` port
- Worker node: start `OmniCoordClientForStage`, connect to `-oma`:`-omp`
- Worker sends registration: "I am Stage N at tcp://my-ip:port"
- Master's Orchestrator queries coordinator for stage locations

### 6.9 Summary: Implementation Order

```
┌─────────────────────────────────────────────────────────────────┐
│  P0 (Must have — unblocks everything)                           │
│                                                                  │
│  1. Stage filtering by --stage-id     → async_omni_engine.py    │
│  2. RemoteStageEngineCoreClient       → new file                │
│  3. Orchestrator remote dispatch      → orchestrator.py         │
├─────────────────────────────────────────────────────────────────┤
│  P1 (Enable clean multi-node UX)                                │
│                                                                  │
│  4. Network request forwarding        → orchestrator.py         │
│  5. Network output collection         → orchestrator.py         │
│  6. Resurrect headless/worker mode    → serve.py, api_server.py │
│  7. Wire OmniCoordinator              → orchestrator.py         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Approach A: Minimal Changes to Make It Work

For a quick path to multi-node without a full rewrite, the key insight is: **the data plane (Mooncake) already works cross-node — only the control plane is missing.**

### 7.1 Minimal Architecture

```
  Node A (Master)                        Node B (Worker)
  ┌──────────────────────────┐          ┌──────────────────────────┐
  │  vllm serve ... --stage-id 0       │  vllm serve ... --stage-id 1
  │                          │          │                          │
  │  API Server (:8000)      │          │  NO API Server           │
  │  Orchestrator            │          │  Stage 1 Engine only     │
  │  Stage 0 Engine          │          │                          │
  │       │                  │          │       ▲                  │
  │       │ ZMQ control (:8091)────────────────┘                  │
  │       │                  │          │                          │
  │       ▼                  │          │       ▼                  │
  │  connector.put(KV)       │          │  connector.get(KV)       │
  └───────────┬──────────────┘          └───────────┬──────────────┘
              │                                     │
              │         ┌──────────────┐            │
              └────────▶│   Mooncake   │◀───────────┘
                        └──────────────┘
```

### 7.2 Minimal Change List

| # | Change | File | Lines of Code |
|:--|:-------|:-----|:-------------|
| 1 | Filter stages by `--stage-id` in `_resolve_stage_configs()` | `async_omni_engine.py` | ~10 |
| 2 | Add `RemoteStageEngineCoreClient` (ZMQ DEALER/ROUTER) | new file | ~150 |
| 3 | Orchestrator: if stage is remote, use remote client | `orchestrator.py` | ~30 |
| 4 | Worker mode: skip API server when `--stage-id` + `-oma` are set | `serve.py`, `api_server.py` | ~20 |
| 5 | Worker startup: connect to master, register stage | `serve.py` | ~40 |
| **Total** | | | **~250 lines** |

### 7.3 What You Get

- Stage 0 on Node A, Stage 1 on Node B — each loading only its own model
- KV cache transferred via Mooncake (already works)
- Control plane via ZMQ between Orchestrator and remote stage
- API endpoint only on Node A

### 7.4 What You Don't Get (Future Work)

- Auto-discovery (stages must be manually started in order)
- Failover / health monitoring
- Dynamic scaling (fixed 1:1 stage-to-node mapping)

### 7.5 Dev Testing: Single-Machine Two-Process Setup

You don't need two physical machines to develop and test the distributed control plane. Run both stages as **separate processes on the same machine** with Mooncake connectors — this exercises the exact same code paths as real multi-node.

**Why this works:**
- Mooncake connector uses TCP even on localhost — same protocol as cross-node
- ZMQ control plane uses TCP sockets, not IPC — same protocol as cross-node
- Each process loads only its own stage (via `--stage-id`) — same isolation as cross-node
- The only difference vs. real multi-node is the IP address in the YAML (`127.0.0.1` → real IPs)

**Architecture:**

```
┌──────────────── Single Machine ─────────────────────────┐
│                                                          │
│  Process 0: Mooncake Master (start first)                │
│  ┌────────────────────────────────────────┐              │
│  │ mooncake_master --rpc_port=50051       │              │
│  │   --http_metadata_server_port=8080     │              │
│  └────────────────────────────────────────┘              │
│                                                          │
│  Process 1: Stage 0 (Master)    Process 2: Stage 1       │
│  ┌─────────────────────────┐   ┌────────────────────┐   │
│  │ vllm serve              │   │ vllm serve         │   │
│  │   BAGEL-7B-MoT --omni  │   │   BAGEL-7B-MoT    │   │
│  │   --stage-id 0          │   │   --stage-id 1     │   │
│  │   -oma 127.0.0.1       │   │   -oma 127.0.0.1   │   │
│  │   -omp 8091             │   │   -omp 8091        │   │
│  │   --port 8000           │   │   (worker mode)    │   │
│  │                         │   │                    │   │
│  │ CUDA_VISIBLE_DEVICES=0  │   │ CUDA_VISIBLE_      │   │
│  │                         │   │   DEVICES=1        │   │
│  │ API Server ✅           │   │ API Server ❌      │   │
│  │ Orchestrator ✅         │   │ DiT Engine only    │   │
│  │ Thinker Engine          │   │                    │   │
│  └────────┬────────────────┘   └──────┬─────────────┘   │
│           │      ZMQ :8091      ▲     │                  │
│           │      ◄──────────────┘     │                  │
│           │                           │                  │
│           │   ┌──────────────────┐    │                  │
│           └──▶│ Mooncake Master  │◀───┘                  │
│               │ 127.0.0.1:50051  │                       │
│               └──────────────────┘                       │
└──────────────────────────────────────────────────────────┘
```

**Startup sequence (3 terminals):**

```bash
# Terminal 1: Mooncake Master
mooncake_master \
  --rpc_port=50051 \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 \
  --http_metadata_server_port=8080

# Terminal 2: Stage 0 (Thinker / Master) — wait for Mooncake to be ready
CUDA_VISIBLE_DEVICES=0 python -m vllm_omni.entrypoints.cli.serve \
  ByteDance-Seed/BAGEL-7B-MoT --omni \
  --port 8000 \
  --stage-configs-path bagel_multiconnector.yaml \
  --stage-id 0 \
  -oma 127.0.0.1 -omp 8091

# Terminal 3: Stage 1 (DiT / Worker) — wait for Stage 0 to be ready
CUDA_VISIBLE_DEVICES=1 python -m vllm_omni.entrypoints.cli.serve \
  ByteDance-Seed/BAGEL-7B-MoT --omni \
  --stage-configs-path bagel_multiconnector.yaml \
  --stage-id 1 \
  -oma 127.0.0.1 -omp 8091
```

**Single GPU testing:** If you only have one GPU, both processes can share it by setting `gpu_memory_utilization: 0.45` for each stage (already the default in `bagel_multiconnector.yaml`). Use the same `CUDA_VISIBLE_DEVICES=0` for both.

**Stage config for localhost testing** — use `bagel_multiconnector.yaml` as-is, just ensure the Mooncake addresses point to `127.0.0.1`:

```yaml
connectors:
  mooncake_connector:
    name: MooncakeStoreConnector
    extra:
      host: "127.0.0.1"
      metadata_server: "http://127.0.0.1:8080/metadata"
      master: "127.0.0.1:50051"
      segment: 512000000
      proto: "tcp"
```

**Test request:**

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ByteDance-Seed/BAGEL-7B-MoT",
    "messages": [{"role": "user", "content": "Generate an image of a cute cat"}]
  }'
```

**What this validates:**
- `--stage-id` correctly filters to one stage per process
- Worker mode (no API server on Stage 1) works
- ZMQ control plane connects Stage 0 ↔ Stage 1
- Mooncake KV cache transfer via TCP on localhost
- Orchestrator dispatches to remote stage client
- Output flows back from Stage 1 → Orchestrator → API response

**Transitioning to real multi-node:** Change `127.0.0.1` to actual IPs in the YAML and CLI args. No code changes needed.

---

## 8. Approach B: Kubernetes-Native Orchestration

For production multi-node deployment, Kubernetes provides the infrastructure that the codebase currently lacks: service discovery, lifecycle management, and scaling.

### 8.1 Architecture

```
┌─────────────────────── Kubernetes Cluster ───────────────────────┐
│                                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Mooncake     │  │ Stage 0 Pod  │  │ Stage 1 Pod          │   │
│  │ Master Pod   │  │              │  │                      │   │
│  │              │  │ Container:   │  │ Container:           │   │
│  │ :50051 gRPC  │  │  vllm serve  │  │  vllm serve          │   │
│  │ :8080  HTTP  │  │  --stage-id 0│  │  --stage-id 1        │   │
│  │              │  │  -oma self   │  │  -oma stage-0-svc    │   │
│  │              │  │  -omp 8091   │  │  -omp 8091           │   │
│  │              │  │  --port 8000 │  │  (no API server)     │   │
│  │              │  │              │  │                      │   │
│  │  Service:    │  │  Service:    │  │  (headless svc for   │   │
│  │  mooncake-   │  │  stage-0-svc │  │   ZMQ discovery)     │   │
│  │  master-svc  │  │  :8000 API   │  │                      │   │
│  │              │  │  :8091 ctrl  │  │  Resources:          │   │
│  │              │  │              │  │  nvidia.com/gpu: 1    │   │
│  │              │  │  Resources:  │  │                      │   │
│  │              │  │  nvidia/gpu:1│  │                      │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  Ingress / LoadBalancer                                   │    │
│  │  External traffic → stage-0-svc:8000                      │    │
│  └──────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────┘
```

### 8.2 Kubernetes Handles What's Missing

| vllm-omni Gap | K8s Solution |
|:-------------|:-------------|
| Service discovery (where is Stage 1?) | K8s Service DNS: `stage-1-svc.namespace.svc.cluster.local` |
| Health monitoring | Liveness/readiness probes on ZMQ or HTTP health endpoint |
| Startup ordering (Stage 0 before Stage 1) | `initContainers` or K8s Job dependencies |
| Failover | Pod restart policies + `ReplicaSet` |
| GPU scheduling | `nvidia.com/gpu` resource requests |
| Mooncake Master lifecycle | Separate Deployment + Service |

### 8.3 What Still Needs Code Changes

Even with K8s, the **same P0 code gaps** from Section 6 apply:

| Gap | K8s Helps? | Still Need Code? |
|:----|:-----------|:-----------------|
| Stage filtering by `--stage-id` | No | **Yes** — K8s sets env/args, but engine must filter |
| RemoteStageClient | No | **Yes** — K8s provides DNS, but code must use it |
| Orchestrator remote dispatch | No | **Yes** — core routing logic unchanged |
| Skip API server on workers | No | **Yes** — container args trigger it, but code must support |
| Service discovery | **Yes** — K8s DNS replaces OmniCoordinator | Can skip OmniCoordinator |
| Health checks | **Partially** — K8s probes, but need health endpoint | Add `/health` endpoint |
| Startup ordering | **Yes** — `initContainers` wait for Stage 0 | Minimal |

### 8.4 K8s-Specific Additions

Beyond the P0 code changes, K8s-native support needs:

| Addition | Purpose |
|:---------|:--------|
| Helm chart or Kustomize manifests | Deploy Mooncake + Stage 0 + Stage 1 as a unit |
| `MOONCAKE_MASTER_ADDR` env var | Point all stages to `mooncake-master-svc:50051` |
| `OMNI_MASTER_ADDR` env var | Point workers to `stage-0-svc:8091` |
| Readiness probe on `-omp` port | Stage 0 ready when Orchestrator is listening |
| GPU topology-aware scheduling | Place stages on nodes with right GPU types |
| Shared YAML ConfigMap | Mount `bagel_multiconnector.yaml` with K8s DNS addresses |

### 8.5 Example K8s Manifests (Sketch)

**Mooncake Master:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mooncake-master
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: mooncake
        image: mooncake:latest
        command: ["mooncake_master"]
        args:
          - "--rpc_port=50051"
          - "--enable_http_metadata_server=true"
          - "--http_metadata_server_host=0.0.0.0"
          - "--http_metadata_server_port=8080"
        ports:
          - containerPort: 50051
          - containerPort: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: mooncake-master-svc
spec:
  selector:
    app: mooncake-master
  ports:
    - name: grpc
      port: 50051
    - name: http
      port: 8080
```

**Stage 0 (Thinker / Master):**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bagel-stage-0
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: vllm
        image: vllm-omni:latest
        command: ["python", "-m", "vllm_omni.entrypoints.cli.serve"]
        args:
          - "ByteDance-Seed/BAGEL-7B-MoT"
          - "--omni"
          - "--port=8000"
          - "--stage-id=0"
          - "--stage-configs-path=/config/bagel_k8s.yaml"
          - "-oma=0.0.0.0"
          - "-omp=8091"
        ports:
          - containerPort: 8000  # API
          - containerPort: 8091  # Control plane
        resources:
          limits:
            nvidia.com/gpu: 1
        volumeMounts:
          - name: config
            mountPath: /config
          - name: model-cache
            mountPath: /root/.cache/huggingface
      volumes:
        - name: config
          configMap:
            name: bagel-stage-config
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: stage-0-svc
spec:
  selector:
    app: bagel-stage-0
  ports:
    - name: api
      port: 8000
    - name: control
      port: 8091
```

**Stage 1 (DiT / Worker):**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bagel-stage-1
spec:
  replicas: 1
  template:
    spec:
      initContainers:
      - name: wait-for-master
        image: busybox
        command: ['sh', '-c', 'until nc -z stage-0-svc 8091; do sleep 2; done']
      containers:
      - name: vllm
        image: vllm-omni:latest
        command: ["python", "-m", "vllm_omni.entrypoints.cli.serve"]
        args:
          - "ByteDance-Seed/BAGEL-7B-MoT"
          - "--omni"
          - "--stage-id=1"
          - "--stage-configs-path=/config/bagel_k8s.yaml"
          - "-oma=stage-0-svc"
          - "-omp=8091"
        resources:
          limits:
            nvidia.com/gpu: 1
        volumeMounts:
          - name: config
            mountPath: /config
          - name: model-cache
            mountPath: /root/.cache/huggingface
      volumes:
        - name: config
          configMap:
            name: bagel-stage-config
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache-pvc
```

**ConfigMap for stage config (with K8s DNS addresses):**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: bagel-stage-config
data:
  bagel_k8s.yaml: |
    stage_args:
      - stage_id: 0
        stage_type: llm
        # ... (same as bagel_multiconnector.yaml)
        output_connectors:
          to_stage_1: mooncake_connector
      - stage_id: 1
        stage_type: diffusion
        # ...
        input_connectors:
          from_stage_0: mooncake_connector
    runtime:
      connectors:
        mooncake_connector:
          name: MooncakeStoreConnector
          extra:
            host: "0.0.0.0"
            metadata_server: "http://mooncake-master-svc:8080/metadata"
            master: "mooncake-master-svc:50051"
            segment: 512000000
            proto: "tcp"
      edges:
        - from: 0
          to: 1
```

### 8.6 Relationship: A is Foundation, B is Infra Layer

Approach A and B are **not alternatives** — they are **layers**. A is the engine-level capability that makes distributed stages possible. B is the infrastructure layer that makes it easy to deploy and operate.

```
  ┌──────────────────────────────────────────────────┐
  │  Approach B: Kubernetes Infra Layer              │
  │  (Helm chart, DNS discovery, probes, GPU sched)  │
  │                                                    │
  │  ─── uses the same CLI args from Approach A ───   │
  └──────────────────────────────────────────────────┘
                        ▲
                        │ wraps (no conflict)
                        │
  ┌──────────────────────────────────────────────────┐
  │  Approach A: Engine Code Changes                 │
  │  (--stage-id filtering, RemoteStageClient,       │
  │   Orchestrator remote dispatch, worker mode)     │
  └──────────────────────────────────────────────────┘
```

| Aspect | A: Engine Changes | B: K8s Layer |
|:-------|:-----------------|:-------------|
| **What it does** | Makes the engine capable of distributed stages | Makes distributed stages easy to deploy |
| **Code changes** | ~250 lines in vllm-omni | Helm chart + manifests (no engine changes) |
| **Required?** | **Yes** — prerequisite for any multi-node | Optional — one of many deployment targets |
| **Service discovery** | Manual (start Stage 0 first) | K8s DNS automatic |
| **Failover** | None | Pod restart + readiness probes |
| **GPU scheduling** | Manual device assignment | K8s GPU scheduler |
| **Startup ordering** | Manual | `initContainers` |
| **Can run without the other?** | Yes (bare metal, VMs) | No — needs A first |

**Implementation order:** Approach A first (engine capability), then Approach B (K8s manifests). The engine code is infrastructure-agnostic — it works on bare metal, VMs, or K8s. K8s just provides the operational layer.

---

## 9. In-Progress Work (as of 2026-03-20)

Before starting implementation, check these active PRs and RFCs — they directly overlap with the gaps identified in this document.

### Active PRs

| PR | Title | Overlaps With | Status |
|:---|:------|:-------------|:-------|
| [#2020](https://github.com/vllm-project/vllm-omni/pull/2020) | **Stage CLI Refactor** | Gap 1 (stage filtering), Gap 6 (headless/worker mode) — refactoring `--stage-id`, `-oma`, `-omp` handling | Open, WIP |
| [#2006](https://github.com/vllm-project/vllm-omni/pull/2006) | **Refactor StageDiffusionClient and StageEngineCoreClient** | Gap 3 (RemoteStageClient) — moves stage inference into ZMQ-connected subprocesses (`StageDiffusionProc`, `StageEngineCoreProc`). Once stages talk via ZMQ, switching from local→remote ZMQ is straightforward. | Open, WIP |
| [#2000](https://github.com/vllm-project/vllm-omni/pull/2000) | **Port Bagel RDMA flow to latest main** | Section 2.4 — rebasing `MooncakeTransferEngineConnector` for BAGEL onto current main. P0 RDMA work from RFC #1192. | Open, WIP |
| [#1988](https://github.com/vllm-project/vllm-omni/pull/1988) | **Ray executor backend for diffusion** | Section 4 — `RayDiffusionExecutor` for multi-node GPU within a single DiT stage (TP/SP). Not cross-stage, but enables Stage 1 to span multiple GPUs/nodes. | Open |
| [#1899](https://github.com/vllm-project/vllm-omni/pull/1899) | **Fix OmniCoordinator reconnect/heartbeat bugs** | Gap 7 — hardening the coordinator before wiring it in. Fixes reconnect, close/update race, heartbeat stall. | Open |

### Active RFCs

| Issue | Title | Relevance |
|:------|:------|:----------|
| [#1192](https://github.com/vllm-project/vllm-omni/issues/1192) | **Full Disaggregation Roadmap Q1 2026** | Master roadmap — EPDG, native worker actor, communication refactor |
| [#1823](https://github.com/vllm-project/vllm-omni/issues/1823) | **Mooncake Transfer Engine for Bagel AR/DiT** | Detailed RDMA RFC — covers CFG multi-KV correctness, img2img metadata, port namespace separation. PR #2000 implements this. |
| [#1940](https://github.com/vllm-project/vllm-omni/issues/1940) | **NCCL-Based Connector** | GPU-direct inter-stage connector — zero CPU copy via NCCL put/wait. Alternative to Mooncake for NVLink/IB connected nodes. |
| [#1792](https://github.com/vllm-project/vllm-omni/issues/1792) | **Ray Connector** | Ray object store as connector — eliminates Mooncake dependency if Ray cluster exists. Uses Ray Direct Transport (RDT) for RDMA-level perf. |
| [#1867](https://github.com/vllm-project/vllm-omni/issues/1867) | **Multi-Stage KV Cache Management Roadmap** | CPU offloading, LMCache, cross-stage KV reuse. Builds on top of connector work. |
| [#1902](https://github.com/vllm-project/vllm-omni/issues/1902) | **OmniCoordinator liveness desync bug** | Coordinator failure modes — relevant when wiring Gap 7. PR #1899 fixes known issues. |

### Coordination Notes

- **PR #2020 + #2006 are the critical ones to track** — they directly address Gaps 1, 3, and 6. Coordinate with those authors before starting Approach A to avoid duplicate work.
- **PR #2000** is orthogonal (data plane RDMA) and can proceed independently from control plane work.
- **RFC #1792 (Ray Connector)** could eventually replace Mooncake for teams already on Ray, making the K8s deployment simpler (no Mooncake Master pod needed).

---

## 10. Related Documents

- [Disaggregated Inference](disaggregated_inference.md) — Connector API and configuration model
- [Ray-based Execution](ray_based_execution.md) — Ray cluster setup and `--worker-backend ray`
- [MooncakeStoreConnector](omni_connectors/mooncake_store_connector.md) — TCP/store-based connector design
- [MooncakeTransferEngineConnector](omni_connectors/mooncake_transfer_engine_connector.md) — RDMA connector design
- [SharedMemoryConnector](omni_connectors/shared_memory_connector.md) — Single-node connector
- [Architecture Overview](../architecture_overview.md) — vllm-omni high-level design
