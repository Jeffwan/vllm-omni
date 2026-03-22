# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Remote Diffusion Stage Client for cross-node distributed deployment.

ZMQ-based proxy that the Master's Orchestrator uses as a drop-in replacement
for StageDiffusionClient when the diffusion stage runs on a remote worker node.

Uses PUSH/PULL socket pairs:
  - PUSH (connect) → Worker's PULL (bind): sends requests
  - PULL (connect) → Worker's PUSH (bind): receives responses
"""

from __future__ import annotations

import asyncio
import pickle
from typing import TYPE_CHECKING, Any

import zmq
import zmq.asyncio

from vllm.logger import init_logger

from vllm_omni.engine.stage_init_utils import StageMetadata
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType

logger = init_logger(__name__)


class RemoteDiffusionClient:
    """ZMQ client that proxies diffusion requests to a remote WorkerStageServer.

    Exposes the same interface as StageDiffusionClient so the Orchestrator
    can use it transparently for remote stages.
    """

    stage_type: str = "diffusion"

    def __init__(
        self,
        worker_address: str,
        worker_port: int,
        metadata: StageMetadata,
    ) -> None:
        # Match StageDiffusionClient attributes
        self.stage_id = metadata.stage_id
        self.final_output = metadata.final_output
        self.final_output_type = metadata.final_output_type
        self.default_sampling_params = metadata.default_sampling_params
        self.custom_process_input_func = metadata.custom_process_input_func
        self.engine_input_source = metadata.engine_input_source

        self._worker_address = worker_address
        self._worker_port = worker_port

        # ZMQ sockets (created lazily in start())
        self._ctx: zmq.asyncio.Context | None = None
        self._push_socket: zmq.asyncio.Socket | None = None
        self._pull_socket: zmq.asyncio.Socket | None = None

        # Output queue fed by background recv task
        self._output_queue: asyncio.Queue[OmniRequestOutput] = asyncio.Queue()
        self._recv_task: asyncio.Task | None = None
        self._shutdown = False

        logger.info(
            "[RemoteDiffusionClient] Stage-%s targeting worker at %s:%d",
            self.stage_id,
            worker_address,
            worker_port,
        )

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Create ZMQ sockets and start the background recv loop.

        Must be called from within the Orchestrator's asyncio event loop.
        """
        self._ctx = zmq.asyncio.Context()

        # PUSH to send requests → Worker's PULL on port
        self._push_socket = self._ctx.socket(zmq.PUSH)
        request_addr = f"tcp://{self._worker_address}:{self._worker_port}"
        self._push_socket.connect(request_addr)

        # PULL to receive responses ← Worker's PUSH on port+1
        self._pull_socket = self._ctx.socket(zmq.PULL)
        response_addr = f"tcp://{self._worker_address}:{self._worker_port + 1}"
        self._pull_socket.connect(response_addr)

        self._recv_task = asyncio.ensure_future(self._drain_responses())

        logger.info(
            "[RemoteDiffusionClient] Stage-%s connected: requests→%s, responses←%s",
            self.stage_id,
            request_addr,
            response_addr,
        )

    async def _drain_responses(self) -> None:
        """Background task: receive responses from Worker and enqueue them."""
        assert self._pull_socket is not None
        while not self._shutdown:
            try:
                raw = await self._pull_socket.recv()
                msg = pickle.loads(raw)

                if msg["type"] == "output":
                    output = msg["output"]
                    await self._output_queue.put(output)
                elif msg["type"] == "error":
                    logger.error(
                        "[RemoteDiffusionClient] Stage-%s worker error for req=%s: %s",
                        self.stage_id,
                        msg.get("request_id", "?"),
                        msg.get("error", "unknown"),
                    )
            except zmq.ZMQError as e:
                if self._shutdown:
                    break
                logger.warning(
                    "[RemoteDiffusionClient] Stage-%s recv error: %s",
                    self.stage_id,
                    e,
                )
            except Exception:
                if self._shutdown:
                    break
                logger.exception(
                    "[RemoteDiffusionClient] Stage-%s unexpected recv error",
                    self.stage_id,
                )

    async def add_request_async(
        self,
        request_id: str,
        prompt: OmniPromptType,
        sampling_params: OmniDiffusionSamplingParams,
    ) -> None:
        """Serialize and send a diffusion request to the remote Worker."""
        assert self._push_socket is not None
        msg = {
            "type": "add_request",
            "request_id": request_id,
            "prompt": prompt,
            "sampling_params": sampling_params,
        }
        await self._push_socket.send(pickle.dumps(msg))
        logger.debug(
            "[RemoteDiffusionClient] Stage-%s sent request %s",
            self.stage_id,
            request_id,
        )

    def get_diffusion_output_async(self) -> OmniRequestOutput | None:
        """Non-blocking poll for completed outputs (same as StageDiffusionClient)."""
        try:
            return self._output_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        """Send abort message to the remote Worker."""
        if self._push_socket is None:
            return
        msg = {
            "type": "abort",
            "request_ids": request_ids,
        }
        await self._push_socket.send(pickle.dumps(msg))

    async def collective_rpc_async(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Stub for control RPCs — not supported for remote stages yet."""
        return {
            "supported": False,
            "reason": f"Remote diffusion stage does not support collective_rpc: {method}",
        }

    def shutdown(self) -> None:
        """Send shutdown message and close ZMQ sockets."""
        self._shutdown = True

        if self._push_socket is not None:
            try:
                # Send shutdown synchronously (non-async) since this may be
                # called from a non-async context during cleanup.
                msg = {"type": "shutdown"}
                self._push_socket.send(pickle.dumps(msg), zmq.NOBLOCK)
            except zmq.ZMQError:
                pass

        if self._recv_task is not None:
            self._recv_task.cancel()

        if self._push_socket is not None:
            self._push_socket.close(linger=100)
        if self._pull_socket is not None:
            self._pull_socket.close(linger=100)
        if self._ctx is not None:
            self._ctx.term()

        logger.info("[RemoteDiffusionClient] Stage-%s shut down", self.stage_id)
