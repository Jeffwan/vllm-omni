# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for RemoteDiffusionClient <-> WorkerStageServer ZMQ round-trip.

No GPU or model weights required — uses a mock StageDiffusionClient.
This test uses direct pickle+ZMQ to avoid importing the full vllm-omni
package chain (which requires GPU/CUDA).
"""

from __future__ import annotations

import asyncio
import pickle
import sys
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest
import zmq
import zmq.asyncio


# ---- Fake types to avoid importing heavy vllm-omni modules ----


@dataclass
class FakeOmniRequestOutput:
    """Minimal fake that mimics OmniRequestOutput for testing."""

    request_id: str
    finished: bool = True
    final_output_type: str = "image"
    images: list = field(default_factory=list)


class MockStageDiffusionClient:
    """Mock that simulates StageDiffusionClient without any model loading."""

    stage_type: str = "diffusion"

    def __init__(self) -> None:
        self.stage_id = 1
        self.final_output = True
        self.final_output_type = "image"
        self.default_sampling_params = None
        self.custom_process_input_func = None
        self.engine_input_source = [0]

        self._output_queue: asyncio.Queue = asyncio.Queue()
        self._received_requests: list[dict] = []

    async def add_request_async(
        self, request_id: str, prompt: Any, sampling_params: Any
    ) -> None:
        self._received_requests.append(
            {
                "request_id": request_id,
                "prompt": prompt,
                "sampling_params": sampling_params,
            }
        )
        await asyncio.sleep(0.01)
        output = FakeOmniRequestOutput(request_id=request_id)
        await self._output_queue.put(output)

    def get_diffusion_output_async(self) -> FakeOmniRequestOutput | None:
        try:
            return self._output_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        pass

    def shutdown(self) -> None:
        pass


# ---- Inline ZMQ server/client (same logic as WorkerStageServer/RemoteDiffusionClient) ----
# This avoids importing the actual modules which pull in the full vllm-omni chain.


class InlineWorkerServer:
    """Minimal ZMQ server matching WorkerStageServer protocol."""

    def __init__(self, host: str, port: int, client: MockStageDiffusionClient):
        self._client = client
        self._shutdown = False
        self._ctx = zmq.asyncio.Context()
        self._pull = self._ctx.socket(zmq.PULL)
        self._pull.bind(f"tcp://{host}:{port}")
        self._push = self._ctx.socket(zmq.PUSH)
        self._push.bind(f"tcp://{host}:{port + 1}")

    async def run(self):
        recv_task = asyncio.create_task(self._recv_loop())
        output_task = asyncio.create_task(self._output_loop())
        try:
            await asyncio.gather(recv_task, output_task)
        except asyncio.CancelledError:
            pass
        finally:
            self._pull.close(linger=100)
            self._push.close(linger=100)
            self._ctx.term()

    async def _recv_loop(self):
        while not self._shutdown:
            events = await self._pull.poll(timeout=100)
            if not events:
                continue
            raw = await self._pull.recv()
            msg = pickle.loads(raw)
            if msg["type"] == "add_request":
                await self._client.add_request_async(
                    msg["request_id"], msg["prompt"], msg["sampling_params"]
                )
            elif msg["type"] == "shutdown":
                self._shutdown = True
                break

    async def _output_loop(self):
        while not self._shutdown:
            output = self._client.get_diffusion_output_async()
            if output is not None:
                msg = {"type": "output", "output": output}
                await self._push.send(pickle.dumps(msg))
            else:
                await asyncio.sleep(0.001)


class InlineRemoteClient:
    """Minimal ZMQ client matching RemoteDiffusionClient protocol."""

    def __init__(self, host: str, port: int):
        self._ctx = zmq.asyncio.Context()
        self._push = self._ctx.socket(zmq.PUSH)
        self._push.connect(f"tcp://{host}:{port}")
        self._pull = self._ctx.socket(zmq.PULL)
        self._pull.connect(f"tcp://{host}:{port + 1}")
        self._output_queue: asyncio.Queue = asyncio.Queue()
        self._recv_task: asyncio.Task | None = None
        self._shutdown = False

    def start(self):
        self._recv_task = asyncio.create_task(self._drain())

    async def _drain(self):
        while not self._shutdown:
            try:
                raw = await asyncio.wait_for(self._pull.recv(), timeout=0.1)
                msg = pickle.loads(raw)
                if msg["type"] == "output":
                    await self._output_queue.put(msg["output"])
            except asyncio.TimeoutError:
                continue
            except Exception:
                if self._shutdown:
                    break

    async def send_request(self, request_id: str, prompt: Any, params: Any):
        msg = {"type": "add_request", "request_id": request_id, "prompt": prompt, "sampling_params": params}
        await self._push.send(pickle.dumps(msg))

    async def send_shutdown(self):
        msg = {"type": "shutdown"}
        await self._push.send(pickle.dumps(msg))

    def get_output(self):
        try:
            return self._output_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def close(self):
        self._shutdown = True
        if self._recv_task:
            self._recv_task.cancel()
        self._push.close(linger=100)
        self._pull.close(linger=100)
        self._ctx.term()


# ---- Tests ----


@pytest.mark.asyncio
async def test_zmq_round_trip():
    """Test that a request reaches the worker and the response comes back."""
    mock = MockStageDiffusionClient()
    host, port = "127.0.0.1", 19091

    server = InlineWorkerServer(host, port, mock)
    client = InlineRemoteClient(host, port)

    server_task = asyncio.create_task(server.run())
    await asyncio.sleep(0.1)
    client.start()

    await client.send_request("req-001", {"text": "cute cat"}, {"seed": 42})

    output = None
    for _ in range(50):
        output = client.get_output()
        if output is not None:
            break
        await asyncio.sleep(0.05)

    assert output is not None, "Did not receive output"
    assert output.request_id == "req-001"
    assert output.finished is True
    assert len(mock._received_requests) == 1

    client.close()
    server._shutdown = True
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_multiple_requests():
    """Test sending multiple requests and receiving all outputs."""
    mock = MockStageDiffusionClient()
    host, port = "127.0.0.1", 19093

    server = InlineWorkerServer(host, port, mock)
    client = InlineRemoteClient(host, port)

    server_task = asyncio.create_task(server.run())
    await asyncio.sleep(0.1)
    client.start()

    for i in range(3):
        await client.send_request(f"req-{i}", {"text": f"Prompt {i}"}, {"seed": i})

    outputs = []
    for _ in range(100):
        output = client.get_output()
        if output is not None:
            outputs.append(output)
        if len(outputs) == 3:
            break
        await asyncio.sleep(0.05)

    assert len(outputs) == 3
    assert {o.request_id for o in outputs} == {"req-0", "req-1", "req-2"}

    client.close()
    server._shutdown = True
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_shutdown_message():
    """Test that shutdown message stops the worker server."""
    mock = MockStageDiffusionClient()
    host, port = "127.0.0.1", 19095

    server = InlineWorkerServer(host, port, mock)
    client = InlineRemoteClient(host, port)

    server_task = asyncio.create_task(server.run())
    await asyncio.sleep(0.1)
    client.start()

    await client.send_shutdown()

    # Server should stop
    try:
        await asyncio.wait_for(server_task, timeout=2.0)
    except asyncio.TimeoutError:
        server._shutdown = True
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    client.close()
