# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
OmniQueue — drop-in replacement for mp.Queue that supports multiple transports.

Transports:
    mp      — multiprocessing.Queue (current default, single-host only)
    ipc     — ZMQ IPC (single-host, faster than mp.Queue, no pickle size limits)
    tcp     — ZMQ TCP (cross-host, same API as ipc)

Usage:
    # Single-host (behaves exactly like mp.Queue):
    q = OmniQueue.create("ipc", endpoint="ipc:///tmp/omni-stage-0-in")

    # Multi-node (cross-host, no code change):
    q = OmniQueue.create("tcp", endpoint="tcp://stage-1-pod:5560")

    # Default (wraps mp.Queue for backward compatibility):
    q = OmniQueue.create("mp")

    # Then use like mp.Queue:
    q.put({"request_id": "req_42", ...})
    msg = q.get()              # blocking
    msg = q.get_nowait()       # non-blocking, raises Empty on miss
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class OmniQueue(ABC):
    """Abstract queue interface matching mp.Queue's put/get/get_nowait API."""

    @abstractmethod
    def put(self, obj: Any) -> None:
        """Enqueue an object (blocking)."""
        ...

    @abstractmethod
    def put_nowait(self, obj: Any) -> None:
        """Enqueue an object (non-blocking)."""
        ...

    @abstractmethod
    def get(self, timeout: float | None = None) -> Any:
        """Dequeue an object (blocking with optional timeout).

        Raises:
            queue.Empty: If timeout expires and no object is available.
        """
        ...

    @abstractmethod
    def get_nowait(self) -> Any:
        """Dequeue an object without blocking.

        Raises:
            queue.Empty: If no object is available.
        """
        ...

    @abstractmethod
    def empty(self) -> bool:
        """Return True if the queue appears empty (best-effort)."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release resources."""
        ...

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def create(
        transport: str = "mp",
        *,
        endpoint: str | None = None,
        bind: bool = False,
        ctx: mp.context.BaseContext | None = None,
        hwm: int = 1000,
    ) -> OmniQueue:
        """Create a queue with the specified transport.

        Args:
            transport: "mp", "ipc", or "tcp"
            endpoint: ZMQ endpoint URL (required for ipc/tcp).
                      For ipc: "ipc:///tmp/omni-stage-0-in"
                      For tcp: "tcp://0.0.0.0:5560" (bind) or
                               "tcp://host:5560" (connect)
            bind: If True, this end binds (server side).
                  If False, this end connects (client side).
                  Typically the receiver (PULL) binds and the sender
                  (PUSH) connects.
            ctx: multiprocessing context (only for transport="mp")
            hwm: ZMQ high-water mark (max queued messages before
                 back-pressure). Default 1000.

        Returns:
            OmniQueue instance.
        """
        if transport == "mp":
            return MPQueue(ctx=ctx)
        elif transport in ("ipc", "tcp"):
            return ZMQQueue(endpoint=endpoint, bind=bind, hwm=hwm)
        else:
            raise ValueError(f"Unknown transport: {transport!r}. Use 'mp', 'ipc', or 'tcp'.")


# ======================================================================
# mp.Queue wrapper (backward compatible default)
# ======================================================================


class MPQueue(OmniQueue):
    """Wraps mp.Queue behind the OmniQueue interface."""

    def __init__(self, ctx: mp.context.BaseContext | None = None) -> None:
        if ctx is None:
            ctx = mp.get_context("spawn")
        self._q: mp.Queue = ctx.Queue(maxsize=0)

    def put(self, obj: Any) -> None:
        self._q.put(obj)

    def put_nowait(self, obj: Any) -> None:
        self._q.put_nowait(obj)

    def get(self, timeout: float | None = None) -> Any:
        return self._q.get(timeout=timeout)

    def get_nowait(self) -> Any:
        return self._q.get_nowait()

    def empty(self) -> bool:
        return self._q.empty()

    def close(self) -> None:
        try:
            self._q.close()
            self._q.join_thread()
        except Exception:
            pass

    @property
    def raw(self) -> mp.Queue:
        """Access the underlying mp.Queue (for legacy code that type-checks)."""
        return self._q


# ======================================================================
# ZMQ-based queue (ipc:// and tcp://)
# ======================================================================


class ZMQQueue(OmniQueue):
    """
    ZMQ PUSH/PULL queue.

    PULL side (bind=True)  = receiver = replaces ``in_q.get()``
    PUSH side (bind=False) = sender   = replaces ``out_q.put()``

    For a pair of communicating stages, the typical wiring is:

        # Stage-1 receiver (runs inside the stage worker):
        in_q = OmniQueue.create("tcp", endpoint="tcp://0.0.0.0:5560", bind=True)
        task = in_q.get()

        # Orchestrator or Stage-0 sender:
        out_q = OmniQueue.create("tcp", endpoint="tcp://stage-1:5560", bind=False)
        out_q.put({"request_id": "req_42", ...})

    Serialization uses pickle (same as mp.Queue) via zmq's send_pyobj/recv_pyobj.
    """

    def __init__(self, endpoint: str | None, bind: bool, hwm: int = 1000) -> None:
        if endpoint is None:
            raise ValueError("endpoint is required for ZMQ transport")

        try:
            import zmq
        except ImportError as e:
            raise ImportError("ZMQ transport requires pyzmq: pip install pyzmq") from e

        self._zmq = zmq
        self._endpoint = endpoint
        self._bind = bind

        self._ctx = zmq.Context.instance()

        if bind:
            # Receiver (PULL) side
            self._socket = self._ctx.socket(zmq.PULL)
            self._socket.setsockopt(zmq.RCVHWM, hwm)
            self._socket.setsockopt(zmq.LINGER, 1000)  # 1s linger on close
            self._socket.bind(endpoint)
            logger.info("OmniQueue ZMQ PULL bound to %s", endpoint)
        else:
            # Sender (PUSH) side
            self._socket = self._ctx.socket(zmq.PUSH)
            self._socket.setsockopt(zmq.SNDHWM, hwm)
            self._socket.setsockopt(zmq.LINGER, 1000)
            self._socket.connect(endpoint)
            logger.info("OmniQueue ZMQ PUSH connected to %s", endpoint)

        self._poller = zmq.Poller()
        self._poller.register(self._socket, zmq.POLLIN)

    def put(self, obj: Any) -> None:
        self._socket.send_pyobj(obj)

    def put_nowait(self, obj: Any) -> None:
        try:
            self._socket.send_pyobj(obj, self._zmq.NOBLOCK)
        except self._zmq.Again:
            raise queue.Full("ZMQ send buffer full")

    def get(self, timeout: float | None = None) -> Any:
        if timeout is None:
            # Blocking
            return self._socket.recv_pyobj()
        else:
            # Poll with timeout (milliseconds)
            timeout_ms = int(timeout * 1000)
            events = dict(self._poller.poll(timeout=timeout_ms))
            if self._socket in events:
                return self._socket.recv_pyobj()
            raise queue.Empty("ZMQ recv timed out")

    def get_nowait(self) -> Any:
        try:
            return self._socket.recv_pyobj(self._zmq.NOBLOCK)
        except self._zmq.Again:
            raise queue.Empty("No message available")

    def empty(self) -> bool:
        # Non-blocking poll: check if there's a message waiting
        events = dict(self._poller.poll(timeout=0))
        return self._socket not in events

    def close(self) -> None:
        try:
            self._socket.close()
            logger.info("OmniQueue ZMQ socket closed (%s)", self._endpoint)
        except Exception as e:
            logger.warning("Error closing ZMQ socket: %s", e)

    @property
    def endpoint(self) -> str:
        return self._endpoint
