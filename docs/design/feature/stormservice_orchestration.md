# StormService-Based Orchestration for vLLM-Omni Multi-Modality

## Problem Statement

The current vLLM-Omni orchestrator is a **single-process, tightly-coupled pipeline** that manages all stages as child processes via `mp.Queue` or Ray actors. This design:

1. Cannot natively deploy stages on **different Kubernetes nodes**
2. Has a **single point of failure** (the orchestrator process)
3. Cannot **independently scale** individual stages (e.g., scale Thinker to 2 replicas while Talker stays at 1)
4. Has no built-in **health checking, rolling updates, or self-healing**
5. Uses placement group strategy `"PACK"` — placing all stages on the **same node**

We propose using AIBrix **StormService** as a Kubernetes-native orchestrator that treats each pipeline stage as an independently deployable role.

## Current vs Proposed Architecture

### Current: Single-Process Orchestrator

```
┌─────────────────────────────────────────────────────┐
│              Omni Orchestrator (single process)      │
│                                                      │
│  in_q[0] ──→ Stage-0 Worker ──→ out_q[0]           │
│              (mp.Process)                            │
│                    │ try_send_via_connector()        │
│                    ▼                                 │
│  in_q[1] ──→ Stage-1 Worker ──→ out_q[1]           │
│              (mp.Process)                            │
│                    │                                 │
│                    ▼                                 │
│  in_q[2] ──→ Stage-2 Worker ──→ out_q[2]           │
│              (mp.Process)                            │
└─────────────────────────────────────────────────────┘
          All on same node, coupled by mp.Queue
```

**Coupling points:**
- `mp.Queue` / `RayQueue` for task dispatch and result collection
- Orchestrator creates/kills all stage processes
- SharedMemory assumes same machine
- Device locks on local filesystem
- Single YAML config read by orchestrator

### Proposed: StormService + Microservice Stages

```
                     ┌────────────────────────┐
                     │   Kubernetes Cluster    │
                     └────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ StormService CRD: qwen25-omni-3stage                             │
│                                                                   │
│  ┌─ RoleSet ──────────────────────────────────────────────────┐  │
│  │                                                             │  │
│  │  Role: thinker (Node A)     Role: talker (Node B)          │  │
│  │  ┌───────────────────┐     ┌───────────────────┐          │  │
│  │  │  Pod: stage-0     │     │  Pod: stage-1     │          │  │
│  │  │  ┌─────────────┐  │     │  ┌─────────────┐  │          │  │
│  │  │  │ stage_service│  │     │  │ stage_service│  │          │  │
│  │  │  │  /generate   │  │     │  │  /generate   │  │          │  │
│  │  │  │  /health     │──┼─────┼──│  /health     │  │          │  │
│  │  │  └──────┬───────┘  │     │  └──────┬───────┘  │          │  │
│  │  │         │          │     │         │          │          │  │
│  │  │         │ put()    │     │  get()  │          │          │  │
│  │  │         ▼          │     │         ▼          │          │  │
│  │  └─────────┼──────────┘     └─────────┼──────────┘          │  │
│  │            │                          │                     │  │
│  │            │   ┌──────────────────┐   │                     │  │
│  │            └──►│   PrisKV         │◄──┘                     │  │
│  │                │  (Data Plane)    │                          │  │
│  │                └──────────────────┘                          │  │
│  │                                                             │  │
│  │  Role: code2wav (Node C)                                   │  │
│  │  ┌───────────────────┐                                     │  │
│  │  │  Pod: stage-2     │                                     │  │
│  │  │  ┌─────────────┐  │                                     │  │
│  │  │  │ stage_service│  │                                     │  │
│  │  │  └──────┬───────┘  │                                     │  │
│  │  │         │ get()    │                                     │  │
│  │  │         ▼          │                                     │  │
│  │  └─────────┼──────────┘                                     │  │
│  │            │                                                │  │
│  │            └──► PrisKV                                      │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌─ OmniRouter Service ──────────────────────────────────────┐   │
│  │  Client → Thinker → (PrisKV) → Talker → (PrisKV) →       │   │
│  │           Code2Wav → Client                                │   │
│  └────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

## Why StormService for Multi-Modality?

StormService was designed for P/D (Prefill/Decode) disaggregation with 2 roles. But its `spec.template.spec.roles` array is **not limited to 2 entries** — it's a generic list. This maps naturally to vLLM-Omni's N-stage pipeline:

| P/D Disaggregation | Omni-Modality |
|---|---|
| Role: prefill | Role: thinker (Stage 0) |
| Role: decode | Role: talker (Stage 1) |
| — | Role: code2wav (Stage 2) |
| — | Role: image_gen (Stage N) |

StormService provides:
- **Independent scaling**: Scale Thinker to 2 replicas while Talker stays at 1
- **Rolling updates**: Update the Talker model without touching Thinker
- **Self-healing**: Kubernetes restarts crashed stage pods automatically
- **Placement control**: Pod anti-affinity spreads stages across nodes
- **Stateful identity**: Stable pod names for service discovery

## Implementation: OmniQueue (ZMQ-Based Transport)

The core problem is that `mp.Queue` only works within a single host. Instead of
adding a full HTTP/gRPC service layer, we replace the queue transport with
**ZMQ PUSH/PULL sockets** via a new `OmniQueue` abstraction.

ZMQ is a natural replacement because:
- `PUSH`/`PULL` is semantically identical to `mp.Queue` (FIFO, blocking get, non-blocking get_nowait)
- Same API for `ipc://` (single-host) and `tcp://` (cross-host) — **zero code changes** in the worker loop
- Sub-millisecond latency (vs HTTP overhead)
- No web server, no JSON serialization, no new framework dependencies
- Built-in reconnection, buffering, and back-pressure (HWM)

### What Changes from the Current Architecture

| Component | Current (Omni class) | New (StormService + ZMQ) |
|---|---|---|
| Process lifecycle | Orchestrator spawns/kills | Kubernetes manages pods |
| Task dispatch | `mp.Queue.put(task)` | `zmq.PUSH.send_pyobj(task)` |
| Result collection | `mp.Queue.get()` | `zmq.PULL.recv_pyobj()` |
| Data plane | SharedMemory / Mooncake | Mooncake / PrisKV (unchanged) |
| Control plane | In-process queue | ZMQ `ipc://` or `tcp://` |
| Health checking | Orchestrator polls queues | K8s TCP probe on ZMQ port |
| Configuration | Single YAML, split by orchestrator | Per-stage env vars / ConfigMaps |
| Scaling | Not supported | `kubectl scale` or HPA per role |

### The OmniQueue Interface

```python
# vllm_omni/distributed/omni_queue.py

class OmniQueue(ABC):
    """Drop-in replacement for mp.Queue that supports multiple transports."""

    def put(self, obj): ...
    def get(self, timeout=None): ...
    def get_nowait(self): ...      # raises queue.Empty
    def empty(self) -> bool: ...
    def close(self): ...

    @staticmethod
    def create(transport, *, endpoint=None, bind=False) -> "OmniQueue":
        """
        transport="mp"   → MPQueue (wraps mp.Queue, backward compatible)
        transport="ipc"  → ZMQQueue("ipc:///tmp/omni-stage-0-in")
        transport="tcp"  → ZMQQueue("tcp://stage-1-pod:5560")
        """
```

### Code Change in Stage Worker: Minimal

The existing `_stage_worker()` loop changes by exactly **zero lines**. It
already calls `in_q.get()`, `in_q.get_nowait()`, `in_q.empty()`, and
`out_q.put()` — all of which `OmniQueue` implements.

The only change is in the orchestrator's `_start_stages()` method, which
swaps `mp.Queue()` for `OmniQueue.create(...)`:

```python
# Before:
in_q = self._ctx.Queue(maxsize=0)      # mp.Queue (local only)

# After (single-node, backward compatible):
in_q = OmniQueue.create("ipc", endpoint=f"ipc:///tmp/omni-stage-{stage_id}-in", bind=True)

# After (multi-node, cross-host):
in_q = OmniQueue.create("tcp", endpoint=f"tcp://0.0.0.0:{5560 + stage_id}", bind=True)
```

### Transport Comparison

```
              mp.Queue         ZMQ ipc://         ZMQ tcp://
              ────────         ──────────         ──────────
scope         same process     same host          cross-host
latency       ~10-50µs         ~5-20µs            ~50-200µs
serialization pickle           pickle             pickle
max msg size  limited by RAM   limited by RAM     limited by RAM
reconnect     N/A              N/A                automatic
back-pressure blocks           HWM (configurable) HWM (configurable)
new deps      none             pyzmq              pyzmq
```

### Request Flow in StormService Deployment (ZMQ)

```
Client
  │
  ├─── POST /v1/chat/completions ───► OmniRouter (K8s Service)
  │                                        │
  │                                        ▼
  │                     zmq.PUSH (tcp://) ───► Thinker Pod (Stage 0)
  │                                               │  task = in_q.get()
  │                                               ├─ engine.generate()
  │                                               ├─ connector.put() → Mooncake
  │                                               └─ out_q.put(result)
  │                                                      │
  │                     zmq.PUSH (tcp://) ◄──────────────┘
  │                            │
  │                            ▼
  │                     zmq.PUSH (tcp://) ───► Talker Pod (Stage 1)
  │                                               │  task = in_q.get()
  │                                               ├─ connector.get() ← Mooncake
  │                                               ├─ engine.generate()
  │                                               ├─ connector.put() → Mooncake
  │                                               └─ out_q.put(result)
  │                                                      │
  │                     zmq.PUSH (tcp://) ◄──────────────┘
  │                            │
  │                            ▼
  │                     zmq.PUSH (tcp://) ───► Code2Wav Pod (Stage 2)
  │                                               │  task = in_q.get()
  │                                               ├─ connector.get() ← Mooncake
  │                                               ├─ engine.generate()
  │                                               └─ out_q.put(audio_output)
  │                                                      │
  │◄───── zmq.PUSH (tcp://) ◄───────────────────────────┘
```

## Can We Implement a Different Orchestrator?

**Yes.** The current orchestrator is not fundamental to vLLM-Omni — it's a convenience layer. Here are the viable alternatives:

### Option 1: StormService + ZMQ (Recommended for Production)

**Pros:**
- Battle-tested Kubernetes operator from AIBrix
- Built-in rolling updates, scaling, health management
- Supports N roles (not just 2)
- Works with any OmniConnector backend
- ZMQ replaces mp.Queue with zero changes to stage worker code
- Same ZMQ API for single-host (ipc://) and multi-node (tcp://)

**Cons:**
- Requires Kubernetes
- Requires pyzmq dependency

**Implementation effort:** Low — OmniQueue is a drop-in; StormService CRD already exists

### Option 2: Ray Serve

**Pros:**
- Already partially supported (`worker_backend="ray"`)
- Native multi-node without Kubernetes
- Built-in autoscaling

**Cons:**
- Requires Ray cluster
- Less standardized than Kubernetes
- Different operational model from Kubernetes

**Implementation effort:** Low — mainly fix placement strategy from PACK→SPREAD

### Option 3: Custom gRPC Orchestrator

**Pros:**
- Full control over protocol and semantics
- No external dependencies (no K8s, no Ray)
- Can optimize for omni-modal specific patterns

**Cons:**
- Significant implementation effort
- Must build health checking, scaling, recovery from scratch
- Reinvents what Kubernetes already provides

**Implementation effort:** High

### Recommended Approach: Hybrid

```
                    ┌───────────────────────────────────┐
                    │      Deployment Environment       │
                    ├───────────────────────────────────┤
                    │                                   │
                    │  Development / Single Node:       │
                    │    → Current Omni orchestrator    │
                    │    → OmniQueue("mp") or ("ipc")  │
                    │    → SharedMemoryConnector        │
                    │                                   │
                    │  Multi-Node / Ray Cluster:        │
                    │    → Ray backend (fix SPREAD)     │
                    │    → MooncakeConnector             │
                    │                                   │
                    │  Production / Kubernetes:         │
                    │    → StormService orchestrator    │
                    │    → OmniQueue("tcp")             │
                    │    → MooncakeConnector / PrisKV    │
                    │                                   │
                    └───────────────────────────────────┘
```

The key insight is that the **OmniConnector abstraction already decouples data transfer from orchestration**. By replacing `mp.Queue` with `OmniQueue` (ZMQ-backed), any orchestrator can drive the pipeline — the worker loop is literally unchanged; only the transport URL changes from `ipc://` to `tcp://`.

## Implementation Roadmap (Two-Step Approach)

### Step 1: OmniQueue + StormService + MooncakeConnector

Replace `mp.Queue` with `OmniQueue` (ZMQ) and use **MooncakeConnector** as the data plane.

- [x] `OmniQueue` abstraction — MPQueue, ZMQQueue (ipc:// + tcp://)
- [x] Unit tests for OmniQueue (mp, ipc, tcp transports)
- [x] StormService YAML with 3 roles (thinker/talker/code2wav) using MooncakeConnector
- [ ] Wire `OmniQueue` into `Omni._start_stages()` and `OmniStage.attach_queues()`
- [ ] OmniRouter service for request ingress (lightweight ZMQ → ZMQ forwarder)
- [ ] Dockerfile for stage worker (existing `_stage_worker` + OmniQueue tcp://)
- [ ] Extend AIBrix Gateway Plugin for multi-stage routing (beyond P/D)
- [ ] E2E test: 3-node deployment with StormService + Mooncake + ZMQ

### Step 2: Replace Mooncake with PrisKV / AIBrix Connector

Once orchestration works, swap the data plane to PrisKV for tighter AIBrix integration.

- [x] `AIBrixKVCacheConnector` implementation
- [x] Factory registration
- [x] YAML stage config with AIBrix connector
- [ ] Unit tests + integration tests with Redis/PrisKV
- [ ] Update StormService YAML to use PrisKV instead of Mooncake Master
- [ ] Benchmark: PrisKV vs Mooncake latency/throughput for stage payloads
- [ ] Production hardening: streaming, cancellation, tracing, autoscaling

## References

- [AIBrix StormService Design](https://aibrix.readthedocs.io/latest/designs/aibrix-stormservice.html)
- [AIBrix KVCache Offloading Framework](https://aibrix.readthedocs.io/latest/designs/aibrix-kvcache-offloading-framework.html)
- [vLLM-Omni Disaggregated Inference](disaggregated_inference.md)
- [vLLM-Omni Ray-Based Execution](ray_based_execution.md)
