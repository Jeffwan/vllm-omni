# Cross-Node Distributed Stage Deployment for BAGEL-7B-MoT

## Context

vllm-omni's BAGEL model runs a 2-stage pipeline (Stage 0: Thinker LLM → Stage 1: DiT Diffusion). Today both stages run in the same process on one machine. The CLI args `--stage-id`, `-oma`, `-omp` exist but are never consumed — all stages are always loaded. `--headless` raises RuntimeError.

**Goal:** Enable Stage 0 on Node A (master with API server) and Stage 1 on Node B (worker, no API server), connected via ZMQ for control and Mooncake for KV cache. Dev-testable on a single machine with two processes.

**Upstream PRs reviewed (both Draft, unmerged):**
- **PR #2020**: Implements "single-stage mode" filtering, `run_headless()`, `register_stage_with_omni_master()`. We adopt the stage filtering concept but write our own since it's unmerged.
- **PR #2006**: Decouples diffusion into ZMQ subprocess (`StageDiffusionProc`) with PUSH/PULL + msgpack. We adopt the ZMQ+msgpack pattern for our remote client.

---

## Architecture

```
Master Process (Node A)                    Worker Process (Node B)
┌─────────────────────────────┐           ┌─────────────────────────────┐
│ API Server (:8000)          │           │ No API Server               │
│ AsyncOmniEngine             │           │                             │
│   Orchestrator              │           │ WorkerStageServer           │
│     stage_clients[0]: local │           │   ZMQ ROUTER (:8092)       │
│       StageEngineCoreClient │           │   StageDiffusionClient      │
│     stage_clients[1]: proxy │  ZMQ TCP  │     (local diffusion engine)│
│       RemoteDiffusionClient ├──────────►│                             │
│                             │           │                             │
│   KV → Mooncake put()      │           │   KV ← Mooncake get()      │
└──────────────┬──────────────┘           └──────────────┬──────────────┘
               │         ┌──────────┐                    │
               └────────►│ Mooncake ├◄───────────────────┘
                         └──────────┘
```

**Alignment with the design doc** (`docs/design/feature/bagel_multinode_deployment.md`):
- Step 1 = Doc Gap 1 (stage filtering)
- Step 2 = Doc Gap 3 (RemoteStageClient)
- Step 3 = Doc Gap 4+5 (network request forwarding + output collection)
- Step 4 = Doc Gap 6 (headless/worker mode)
- Step 5 = Doc Gap 2 (Orchestrator remote dispatch)
- Step 6 = Minimal Orchestrator wiring
- Doc Gap 7 (OmniCoordinator) deferred to future work

**Port semantics:** `-oma`/`-omp` specify where the Worker's ZMQ server listens. Worker binds `tcp://0.0.0.0:{omp}`, Master connects `tcp://{oma}:{omp}`. Uses port 8091 to match the doc convention. A second port (`omp+1`) is used for the response channel (PUSH/PULL pair).

---

## Implementation Plan

### Step 1: Stage Filtering in `_resolve_stage_configs`

**File:** `vllm_omni/engine/async_omni_engine.py`

**What:** After `load_and_resolve_stage_configs()` returns all stages, filter by `kwargs["stage_id"]` if set. Keep the full stage list as `self._all_stage_configs` for pipeline metadata.

**Changes to `__init__` (~215-235):**
- Extract and store `self._stage_id = kwargs.pop("stage_id", None)`
- Extract and store `self._worker_address = kwargs.get("omni_master_address")`, `self._worker_port = kwargs.get("omni_master_port")`
- After `_resolve_stage_configs`, set `self._all_stage_configs = self.stage_configs` (full list)
- If `self._stage_id is not None`, filter: `self.stage_configs = [c for c in self.stage_configs if c.stage_id == self._stage_id]`
- `self.num_stages` stays as `len(self._all_stage_configs)` (full pipeline length) so the Orchestrator knows the topology

**Changes to `_resolve_stage_configs` (lines 857-904):** None needed — filtering happens in `__init__` after the call.

### Step 2: Remote Diffusion Client (new file)

**New File:** `vllm_omni/engine/remote_stage_client.py`

**What:** A ZMQ-based proxy that the Master's Orchestrator uses as a drop-in for `StageDiffusionClient` for remote stages. Adopts the PUSH/PULL + msgpack pattern from PR #2006.

**Interface (must match StageDiffusionClient):**
- Attributes: `stage_type="diffusion"`, `stage_id`, `final_output`, `final_output_type`, `default_sampling_params`, `custom_process_input_func`, `engine_input_source`
- `add_request_async(request_id, prompt, sampling_params)` → serialize + send via ZMQ PUSH
- `get_diffusion_output_async() -> OmniRequestOutput | None` → non-blocking poll from output queue
- `abort_requests_async(request_ids)` → send abort msg
- `shutdown()` → send shutdown msg + close sockets

**ZMQ topology:**
- PUSH socket connects to Worker's PULL address (for requests)
- PULL socket connects to Worker's PUSH address (for responses)
- Uses `OmniMsgpackEncoder`/`OmniMsgpackDecoder` from `vllm_omni/distributed/omni_connectors/utils/serialization.py` (same as PR #2006)
- For complex objects (sampling_params with custom types), use pickle wrapped in msgpack bytes

**Background recv:** `_drain_responses()` async task polls PULL socket, deserializes outputs, puts in `asyncio.Queue`

### Step 3: Worker Stage Server (new file)

**New File:** `vllm_omni/engine/worker_stage_server.py`

**What:** Runs on the Worker node. Binds ZMQ sockets, receives requests from Master, dispatches to a local `StageDiffusionClient`, sends results back.

**ZMQ topology (mirror of RemoteDiffusionClient):**
- PULL socket binds on `tcp://0.0.0.0:{port}` (receives requests from Master's PUSH)
- PUSH socket binds on `tcp://0.0.0.0:{port+1}` (sends responses to Master's PULL)

**Loops:**
- `_recv_loop()`: recv from PULL → unpack → dispatch to local `StageDiffusionClient.add_request_async()`
- `_output_loop()`: poll `StageDiffusionClient.get_diffusion_output_async()` → pack → send via PUSH

**Message protocol:**
```
Request:  {"type": "add_request", "request_id": str, "prompt": bytes(pickle), "sampling_params": bytes(pickle)}
Abort:    {"type": "abort", "request_ids": list[str]}
Shutdown: {"type": "shutdown"}
Response: {"type": "output", "output": bytes(pickle)}
Error:    {"type": "error", "request_id": str, "error": str}
```

### Step 4: Worker Mode in serve.py

**File:** `vllm_omni/entrypoints/cli/serve.py`

**What:** Replace the broken `run_headless()` with a working worker mode.

**Changes:**
1. Ensure `--headless` arg exists in `subparser_init` (already added at line ~168-178, just verify)
2. Rewrite `run_headless()` (lines 375-391):

```python
def run_headless(args):
    """Run a single stage in worker mode (no API server)."""
    uvloop.run(_run_worker_stage(args))

async def _run_worker_stage(args):
    stage_id = args.stage_id
    if stage_id is None:
        raise ValueError("--headless requires --stage-id")

    model = args.model or args.model_tag
    kwargs = vars(args).copy()

    # Load stage configs, pick the selected one
    config_path, all_stage_configs = load_and_resolve_stage_configs(
        model, kwargs.get("stage_configs_path"), kwargs
    )
    stage_cfg = next(c for c in all_stage_configs if c.stage_id == stage_id)
    metadata = extract_stage_metadata(stage_cfg)

    # Initialize KV transfer config for this stage
    omni_transfer_config = load_omni_transfer_config_for_model(model, config_path)
    omni_kv_connector = resolve_omni_kv_config_for_stage(omni_transfer_config, stage_id)
    # Inject KV config into stage_cfg if needed
    omni_conn_cfg, omni_from, omni_to = omni_kv_connector
    if omni_conn_cfg:
        inject_omni_kv_config(stage_cfg, omni_conn_cfg, omni_from, omni_to)

    # Setup GPU devices
    setup_stage_devices(stage_id, metadata.runtime_cfg)

    # Initialize the local diffusion stage client
    stage_client = initialize_diffusion_stage(model, stage_cfg, metadata)

    # Start ZMQ server
    bind_host = args.omni_master_address or "0.0.0.0"
    bind_port = args.omni_master_port or 8092
    server = WorkerStageServer(bind_host, bind_port, stage_client)
    logger.info("Worker stage %d ready on tcp://%s:%d", stage_id, bind_host, bind_port)
    await server.run()
```

### Step 5: Master-Side Stage Initialization Changes

**File:** `vllm_omni/engine/async_omni_engine.py` — `_initialize_stages()` (lines 427-538)

**What:** When running as Master (`self._stage_id == 0`), iterate ALL stage configs. For the local stage, initialize normally. For remote stages, create `RemoteDiffusionClient` instead.

**Key change in the loop (around line 454):**
```python
for global_stage_id, stage_cfg in enumerate(self._all_stage_configs):
    metadata = extract_stage_metadata(stage_cfg)

    # Remote stage — create proxy client
    if self._stage_id is not None and stage_cfg.stage_id != self._stage_id:
        if metadata.stage_type == "diffusion":
            worker_addr = self._worker_address or "127.0.0.1"
            worker_port = self._worker_port or 8092
            remote_client = RemoteDiffusionClient(worker_addr, worker_port, metadata)
            stage_clients[global_stage_id] = remote_client
        continue

    # Local stage — existing initialization code...
```

**Also:** After Orchestrator creation in `_bootstrap_orchestrator` (line 569), start recv loops for remote clients:
```python
for client in self.stage_clients:
    if isinstance(client, RemoteDiffusionClient):
        client.start(asyncio.get_running_loop())
```

### Step 6: Orchestrator — Minimal Changes

**File:** `vllm_omni/engine/orchestrator.py`

**What:** Almost no changes needed. The Orchestrator already dispatches based on `stage_client.stage_type == "diffusion"` and calls the same interface methods. The only addition:

- In `_orchestration_loop` diffusion branch (around line 240): the existing `get_diffusion_output_async()` call works identically for `RemoteDiffusionClient`.
- In `_forward_to_next_stage` diffusion branch (line 459): the existing `add_request_async(req_id, prompt, params)` call works identically.
- `output_processors[stage_id]` for remote diffusion stages should be `None` (diffusion outputs bypass LLM output processor).

**One guard to add:** In `_orchestration_loop`, the `continue` after diffusion output check already exists (line ~250), so no structural change.

---

## Files Summary

| File | Action | ~LOC |
|------|--------|------|
| `vllm_omni/engine/remote_stage_client.py` | **NEW** | ~150 |
| `vllm_omni/engine/worker_stage_server.py` | **NEW** | ~120 |
| `vllm_omni/entrypoints/cli/serve.py` | MODIFY | ~50 |
| `vllm_omni/engine/async_omni_engine.py` | MODIFY | ~40 |
| `vllm_omni/engine/orchestrator.py` | MODIFY | ~5 |
| **Total** | | **~365** |

---

## Key Reusable Code

- `OmniMsgpackEncoder` / `OmniMsgpackDecoder` from `vllm_omni/distributed/omni_connectors/utils/serialization.py` — for ZMQ message serialization
- `StageMetadata` and `extract_stage_metadata()` from `vllm_omni/engine/stage_init_utils.py` — for stage config parsing
- `initialize_diffusion_stage()` from `vllm_omni/engine/stage_init_utils.py` — for Worker's local stage init
- `setup_stage_devices()` from `vllm_omni/engine/stage_init_utils.py` — for GPU setup on Worker
- `load_omni_transfer_config_for_model()` and `resolve_omni_kv_config_for_stage()` from connector utils — for KV config
- `StageDiffusionClient` interface from `vllm_omni/diffusion/stage_diffusion_client.py` — contract for RemoteDiffusionClient

---

## Verification / Testing

### Single-Machine Two-Process Test

Matches the dev testing setup from `docs/design/feature/bagel_multinode_deployment.md` Section 7.5.

**Prerequisites:** Mooncake master running, or use mock for unit tests.

**Terminal 1 — Mooncake Master:**
```bash
mooncake_master --rpc_port=50051 \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 \
  --http_metadata_server_port=8080
```

**Terminal 2 — Master (Stage 0):**
```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm_omni.entrypoints.cli.main serve \
  ByteDance-Seed/BAGEL-7B-MoT --omni \
  --port 8000 \
  --stage-configs-path vllm_omni/model_executor/stage_configs/bagel_multiconnector.yaml \
  --stage-id 0 \
  -oma 127.0.0.1 -omp 8091
```

**Terminal 3 — Worker (Stage 1):**
```bash
CUDA_VISIBLE_DEVICES=1 python -m vllm_omni.entrypoints.cli.main serve \
  ByteDance-Seed/BAGEL-7B-MoT --omni \
  --stage-configs-path vllm_omni/model_executor/stage_configs/bagel_multiconnector.yaml \
  --stage-id 1 --headless \
  -oma 127.0.0.1 -omp 8091
```

**Terminal 4 — Test request:**
```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ByteDance-Seed/BAGEL-7B-MoT","messages":[{"role":"user","content":"Generate an image of a cute cat"}]}'
```

**Note:** `-oma`/`-omp` specify the Worker's ZMQ server address. The Worker binds on `0.0.0.0:-omp`, the Master connects to `-oma:-omp`. ZMQ handles connect-before-bind gracefully.

### Unit Test (no GPU needed)

Create `tests/engine/test_remote_stage_client.py`:
- Test ZMQ message round-trip: `RemoteDiffusionClient` → `WorkerStageServer` → back
- Mock `StageDiffusionClient` on the Worker side
- Verify: request serialization, output deserialization, abort handling, shutdown

### What to Validate E2E

1. `--stage-id 0` starts only the Thinker LLM, not the diffusion engine
2. `--stage-id 1 --headless` starts only the diffusion engine, no API server
3. Master connects to Worker via ZMQ and can send/receive messages
4. KV cache flows from Stage 0 → Mooncake → Stage 1 (existing path, just verify it still works)
5. Diffusion output flows from Worker → Master → API response
6. CFG companion KV caches (cfg_text, cfg_img) also transfer correctly
7. Graceful shutdown: Master shutdown sends shutdown to Worker

---

## Implementation Order

1. **Step 1** (stage filtering) + **Step 5** (master init) — get single-stage loading working
2. **Step 2** (RemoteDiffusionClient) + **Step 3** (WorkerStageServer) — the ZMQ bridge
3. **Step 4** (serve.py worker mode) — CLI entry point
4. **Step 6** (orchestrator) — minimal wiring
5. **Unit test** — ZMQ round-trip without GPU
6. **E2E test** — full pipeline with model (requires GPU + Mooncake)
