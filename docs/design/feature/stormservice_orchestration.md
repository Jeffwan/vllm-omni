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

## Implementation: The Stage Service

The key new component is `stage_service.py` — a standalone HTTP/gRPC service that wraps a single vLLM-Omni stage, making it independently deployable as a container.

### What Changes from the Current Architecture

| Component | Current (Omni class) | New (StormService) |
|---|---|---|
| Process lifecycle | Orchestrator spawns/kills | Kubernetes manages pods |
| Task dispatch | `mp.Queue.put(task)` | HTTP POST `/generate` |
| Result collection | `mp.Queue.get()` | HTTP response / SSE stream |
| Data plane | SharedMemory / Mooncake | PrisKV via AIBrixKVCacheConnector |
| Control plane | In-process queue notify | HTTP callbacks / PrisKV pub/sub |
| Health checking | Orchestrator polls queues | K8s liveness/readiness probes |
| Configuration | Single YAML, split by orchestrator | Per-stage env vars / ConfigMaps |
| Metrics | In-band with queue results | Prometheus /metrics endpoint |
| Scaling | Not supported | `kubectl scale` or HPA per role |

### Stage Service Skeleton

```python
# vllm_omni/entrypoints/stage_service.py
"""
Standalone stage service for Kubernetes deployment.

Each stage runs as an independent HTTP service. Stages communicate
via OmniConnectors (PrisKV) for heavy data and HTTP callbacks for
lightweight notifications.

Usage:
    python -m vllm_omni.entrypoints.stage_service \
        --model Qwen/Qwen2.5-Omni-7B \
        --stage-id 0 \
        --stage-type llm \
        --model-stage thinker \
        --connector-type AIBrixKVCacheConnector \
        --connector-host priskv-service \
        --connector-port 6379 \
        --port 8000
"""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class GenerateRequest(BaseModel):
    request_id: str
    engine_inputs: dict      # For stage-0: prompt + multimodal data
    sampling_params: dict    # Stage-specific sampling params
    from_connector: bool = False  # True if inputs are in PrisKV

class GenerateResponse(BaseModel):
    request_id: str
    status: str              # "completed" | "forwarded" | "error"
    outputs: dict | None     # Final outputs (if final_output stage)
    metrics: dict | None

@app.post("/generate")
async def generate(req: GenerateRequest) -> GenerateResponse:
    """
    Process a generation request for this stage.

    Flow:
    1. If from_connector: fetch inputs from PrisKV
    2. Run engine.generate()
    3. If not final_output: store results in PrisKV, notify next stage
    4. If final_output: return results directly
    """
    # 1. Resolve inputs
    if req.from_connector:
        inputs, _ = connector.get(
            from_stage=req.from_stage,
            to_stage=str(STAGE_ID),
            request_id=req.request_id
        )
    else:
        inputs = req.engine_inputs

    # 2. Generate
    outputs = await engine.generate(inputs, req.sampling_params)

    # 3. Forward or return
    if not FINAL_OUTPUT:
        connector.put(
            from_stage=str(STAGE_ID),
            to_stage=str(NEXT_STAGE_ID),
            request_id=req.request_id,
            data={"engine_inputs": processed_outputs}
        )
        # Notify next stage via HTTP
        await notify_next_stage(req.request_id)
        return GenerateResponse(
            request_id=req.request_id,
            status="forwarded"
        )
    else:
        return GenerateResponse(
            request_id=req.request_id,
            status="completed",
            outputs=serialize_outputs(outputs)
        )

@app.get("/health")
async def health():
    """Kubernetes health probe endpoint."""
    return {
        "status": "healthy",
        "stage_id": STAGE_ID,
        "connector": connector.health()
    }
```

### Request Flow in StormService Deployment

```
Client
  │
  ├─── POST /v1/chat/completions ───► OmniRouter (K8s Service)
  │                                        │
  │                                        ▼
  │                              POST /generate ───► Thinker Pod (Stage 0)
  │                                                      │
  │                                                      ├─ engine.generate()
  │                                                      ├─ connector.put() → PrisKV
  │                                                      └─ POST /generate → Talker Pod
  │                                                                             │
  │                                                      ┌──────────────────────┘
  │                                                      │
  │                                                      ├─ connector.get() ← PrisKV
  │                                                      ├─ engine.generate()
  │                                                      ├─ connector.put() → PrisKV
  │                                                      └─ POST /generate → Code2Wav Pod
  │                                                                             │
  │                                                      ┌──────────────────────┘
  │                                                      │
  │                                                      ├─ connector.get() ← PrisKV
  │                                                      ├─ engine.generate()
  │                                                      └─ Return audio output
  │                                                              │
  │◄──── SSE stream / JSON response ────────────────────────────┘
```

## Can We Implement a Different Orchestrator?

**Yes.** The current orchestrator is not fundamental to vLLM-Omni — it's a convenience layer. Here are the viable alternatives:

### Option 1: StormService (Recommended for Production)

**Pros:**
- Battle-tested Kubernetes operator from AIBrix
- Built-in rolling updates, scaling, health management
- Supports N roles (not just 2)
- Works with any OmniConnector backend

**Cons:**
- Requires Kubernetes
- Requires new `stage_service.py` entrypoint
- Control plane moves from in-process queues to HTTP/gRPC

**Implementation effort:** Medium — StormService CRD already exists; need `stage_service.py` + OmniRouter

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
                    │    → mp.Queue + SharedMemory      │
                    │                                   │
                    │  Multi-Node / Ray Cluster:        │
                    │    → Ray backend (fix SPREAD)     │
                    │    → AIBrixKVCacheConnector        │
                    │                                   │
                    │  Production / Kubernetes:         │
                    │    → StormService orchestrator    │
                    │    → stage_service.py per pod     │
                    │    → AIBrixKVCacheConnector        │
                    │                                   │
                    └───────────────────────────────────┘
```

The key insight is that the **OmniConnector abstraction already decouples data transfer from orchestration**. By adding `stage_service.py` (HTTP wrapper around the existing `_stage_worker` loop), any orchestrator can drive the pipeline — the stages don't care whether tasks come from `mp.Queue`, Ray, or HTTP.

## Implementation Roadmap (Two-Step Approach)

### Step 1: Multi-Modality Orchestration with StormService + Mooncake

Use the **existing MooncakeConnector** as the data plane first. This avoids introducing a new connector and focuses on solving the orchestration problem.

- [ ] `stage_service.py` — FastAPI wrapper for stage worker
- [ ] Health endpoint (`/health`), Generate endpoint (`/generate`), Metrics endpoint (`/metrics`)
- [ ] StormService YAML with 3 roles (thinker/talker/code2wav) using MooncakeConnector
- [ ] OmniRouter service for request ingress
- [ ] Dockerfile for stage service
- [ ] Extend AIBrix Gateway Plugin for multi-stage routing (beyond P/D)
- [ ] E2E test: 3-node deployment with StormService + Mooncake

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
