# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for OmniQueue — mp.Queue and ZMQ transports."""

import multiprocessing as mp
import queue
import threading
import time

import pytest

from vllm_omni.distributed.omni_queue import MPQueue, OmniQueue

# ---------------------------------------------------------------------------
# Skip ZMQ tests if pyzmq is not installed
# ---------------------------------------------------------------------------
zmq = pytest.importorskip("zmq", reason="pyzmq not installed")
from vllm_omni.distributed.omni_queue import ZMQQueue  # noqa: E402


# ===========================================================================
# MPQueue tests (backward compatibility)
# ===========================================================================


class TestMPQueue:
    def test_create_mp(self):
        q = OmniQueue.create("mp")
        assert isinstance(q, MPQueue)
        q.close()

    def test_put_get(self):
        q = OmniQueue.create("mp")
        q.put({"request_id": "req_1", "type": "GENERATE"})
        msg = q.get(timeout=2)
        assert msg["request_id"] == "req_1"
        q.close()

    def test_get_nowait_empty(self):
        q = OmniQueue.create("mp")
        with pytest.raises(queue.Empty):
            q.get_nowait()
        q.close()

    def test_put_get_nowait(self):
        q = OmniQueue.create("mp")
        q.put({"data": 42})
        time.sleep(0.05)  # let the queue propagate
        msg = q.get_nowait()
        assert msg["data"] == 42
        q.close()

    def test_empty(self):
        q = OmniQueue.create("mp")
        assert q.empty()
        q.put("x")
        time.sleep(0.05)
        assert not q.empty()
        q.get()
        q.close()


# ===========================================================================
# ZMQ IPC tests
# ===========================================================================


class TestZMQIPC:
    @pytest.fixture
    def ipc_pair(self, tmp_path):
        """Create a PULL (bind) / PUSH (connect) pair over ipc."""
        endpoint = f"ipc://{tmp_path}/test_queue"
        receiver = OmniQueue.create("ipc", endpoint=endpoint, bind=True)
        sender = OmniQueue.create("ipc", endpoint=endpoint, bind=False)
        time.sleep(0.05)  # let ZMQ handshake
        yield sender, receiver
        sender.close()
        receiver.close()

    def test_send_recv(self, ipc_pair):
        sender, receiver = ipc_pair
        payload = {"request_id": "req_42", "type": "GENERATE", "from_connector": True}
        sender.put(payload)
        msg = receiver.get(timeout=2)
        assert msg == payload

    def test_multiple_messages(self, ipc_pair):
        sender, receiver = ipc_pair
        for i in range(10):
            sender.put({"seq": i})
        for i in range(10):
            msg = receiver.get(timeout=2)
            assert msg["seq"] == i

    def test_get_nowait_empty(self, ipc_pair):
        _, receiver = ipc_pair
        with pytest.raises(queue.Empty):
            receiver.get_nowait()

    def test_get_timeout(self, ipc_pair):
        _, receiver = ipc_pair
        with pytest.raises(queue.Empty):
            receiver.get(timeout=0.1)

    def test_empty_check(self, ipc_pair):
        sender, receiver = ipc_pair
        assert receiver.empty()
        sender.put({"data": 1})
        time.sleep(0.05)
        assert not receiver.empty()

    def test_large_payload(self, ipc_pair):
        """Test with a payload similar to what stage notifications carry."""
        sender, receiver = ipc_pair
        payload = {
            "type": "GENERATE",
            "request_id": "req_large",
            "sampling_params": {"temperature": 0.9, "top_k": 40, "max_tokens": 2048},
            "from_connector": True,
            "from_stage": "0",
            "to_stage": "1",
            "sent_ts": time.time(),
            "connector_metadata": None,
        }
        sender.put(payload)
        msg = receiver.get(timeout=2)
        assert msg["request_id"] == "req_large"
        assert msg["sampling_params"]["temperature"] == 0.9


# ===========================================================================
# ZMQ TCP tests
# ===========================================================================


class TestZMQTCP:
    @pytest.fixture
    def tcp_pair(self):
        """Create a PULL (bind) / PUSH (connect) pair over tcp."""
        endpoint = "tcp://127.0.0.1:15560"
        receiver = OmniQueue.create("tcp", endpoint=endpoint, bind=True)
        sender = OmniQueue.create("tcp", endpoint=endpoint, bind=False)
        time.sleep(0.1)  # TCP handshake takes slightly longer
        yield sender, receiver
        sender.close()
        receiver.close()

    def test_send_recv(self, tcp_pair):
        sender, receiver = tcp_pair
        sender.put({"request_id": "tcp_req_1"})
        msg = receiver.get(timeout=2)
        assert msg["request_id"] == "tcp_req_1"

    def test_multiple_messages(self, tcp_pair):
        sender, receiver = tcp_pair
        for i in range(5):
            sender.put({"seq": i})
        for i in range(5):
            msg = receiver.get(timeout=2)
            assert msg["seq"] == i


# ===========================================================================
# Concurrent producer-consumer test (simulates orchestrator → stage)
# ===========================================================================


class TestConcurrentFlow:
    def test_producer_consumer_ipc(self, tmp_path):
        """Simulate orchestrator sending tasks to a stage worker."""
        endpoint = f"ipc://{tmp_path}/concurrent_test"
        received = []
        n_messages = 50

        def consumer():
            recv_q = OmniQueue.create("ipc", endpoint=endpoint, bind=True)
            for _ in range(n_messages):
                msg = recv_q.get(timeout=5)
                received.append(msg)
            recv_q.close()

        def producer():
            time.sleep(0.05)  # let consumer bind first
            send_q = OmniQueue.create("ipc", endpoint=endpoint, bind=False)
            time.sleep(0.05)
            for i in range(n_messages):
                send_q.put({"request_id": f"req_{i}", "stage_id": 0})
            send_q.close()

        t_consumer = threading.Thread(target=consumer)
        t_producer = threading.Thread(target=producer)
        t_consumer.start()
        t_producer.start()
        t_producer.join(timeout=10)
        t_consumer.join(timeout=10)

        assert len(received) == n_messages
        assert received[0]["request_id"] == "req_0"
        assert received[-1]["request_id"] == f"req_{n_messages - 1}"


# ===========================================================================
# Factory tests
# ===========================================================================


class TestFactory:
    def test_unknown_transport(self):
        with pytest.raises(ValueError, match="Unknown transport"):
            OmniQueue.create("grpc")

    def test_zmq_requires_endpoint(self):
        with pytest.raises(ValueError, match="endpoint is required"):
            OmniQueue.create("tcp", endpoint=None, bind=True)
