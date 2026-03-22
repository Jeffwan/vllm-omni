# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Worker Stage Server for cross-node distributed deployment.

Runs on the Worker node. Binds ZMQ sockets, receives requests from the Master's
RemoteDiffusionClient, dispatches them to a local StageDiffusionClient, and
sends results back.

ZMQ topology (mirror of RemoteDiffusionClient):
  - PULL (bind on port)   ← Master's PUSH: receives requests
  - PUSH (bind on port+1) → Master's PULL: sends responses
"""

from __future__ import annotations

import asyncio
import pickle
import signal
from typing import TYPE_CHECKING

import zmq
import zmq.asyncio

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient

logger = init_logger(__name__)


class WorkerStageServer:
    """ZMQ server wrapping a local StageDiffusionClient for remote access."""

    def __init__(
        self,
        bind_host: str,
        bind_port: int,
        stage_client: StageDiffusionClient,
    ) -> None:
        self._bind_host = bind_host
        self._bind_port = bind_port
        self._stage_client = stage_client
        self._shutdown = False

        self._ctx = zmq.asyncio.Context()

        # PULL socket: receive requests from Master
        self._pull_socket = self._ctx.socket(zmq.PULL)
        self._pull_socket.bind(f"tcp://{bind_host}:{bind_port}")

        # PUSH socket: send responses to Master
        self._push_socket = self._ctx.socket(zmq.PUSH)
        self._push_socket.bind(f"tcp://{bind_host}:{bind_port + 1}")

        logger.info(
            "[WorkerStageServer] Stage-%s bound: requests←tcp://%s:%d, responses→tcp://%s:%d",
            stage_client.stage_id,
            bind_host,
            bind_port,
            bind_host,
            bind_port + 1,
        )

    async def run(self) -> None:
        """Main event loop. Runs until shutdown signal received."""
        loop = asyncio.get_running_loop()

        # Handle SIGINT/SIGTERM gracefully
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._signal_shutdown)

        recv_task = asyncio.create_task(self._recv_loop(), name="worker-recv")
        output_task = asyncio.create_task(self._output_loop(), name="worker-output")

        try:
            await asyncio.gather(recv_task, output_task)
        except asyncio.CancelledError:
            pass
        finally:
            self._cleanup()

    def _signal_shutdown(self) -> None:
        """Handle OS signals for graceful shutdown."""
        logger.info("[WorkerStageServer] Shutdown signal received")
        self._shutdown = True

    async def _recv_loop(self) -> None:
        """Receive requests from Master and dispatch to local stage client."""
        while not self._shutdown:
            try:
                # Use poll with timeout so we can check shutdown flag
                events = await self._pull_socket.poll(timeout=100)  # 100ms
                if not events:
                    continue

                raw = await self._pull_socket.recv()
                msg = pickle.loads(raw)
                msg_type = msg.get("type")

                if msg_type == "add_request":
                    request_id = msg["request_id"]
                    prompt = msg["prompt"]
                    sampling_params = msg["sampling_params"]
                    logger.debug(
                        "[WorkerStageServer] Received request %s",
                        request_id,
                    )
                    await self._stage_client.add_request_async(
                        request_id, prompt, sampling_params
                    )

                elif msg_type == "abort":
                    request_ids = msg["request_ids"]
                    logger.info(
                        "[WorkerStageServer] Aborting requests: %s",
                        request_ids,
                    )
                    await self._stage_client.abort_requests_async(request_ids)

                elif msg_type == "shutdown":
                    logger.info("[WorkerStageServer] Received shutdown from Master")
                    self._shutdown = True
                    break

            except zmq.ZMQError as e:
                if self._shutdown:
                    break
                logger.warning("[WorkerStageServer] Recv error: %s", e)
            except Exception:
                if self._shutdown:
                    break
                logger.exception("[WorkerStageServer] Unexpected recv error")

    async def _output_loop(self) -> None:
        """Poll local stage client for outputs and send back to Master."""
        while not self._shutdown:
            output = self._stage_client.get_diffusion_output_async()
            if output is not None:
                try:
                    msg = {
                        "type": "output",
                        "output": output,
                    }
                    await self._push_socket.send(pickle.dumps(msg))
                    logger.debug(
                        "[WorkerStageServer] Sent output for req=%s",
                        output.request_id,
                    )
                except Exception:
                    logger.exception(
                        "[WorkerStageServer] Failed to send output for req=%s",
                        output.request_id,
                    )
            else:
                await asyncio.sleep(0.001)  # Avoid busy-wait

    def _cleanup(self) -> None:
        """Close sockets and terminate ZMQ context."""
        self._stage_client.shutdown()
        self._pull_socket.close(linger=100)
        self._push_socket.close(linger=100)
        self._ctx.term()
        logger.info("[WorkerStageServer] Cleaned up")
