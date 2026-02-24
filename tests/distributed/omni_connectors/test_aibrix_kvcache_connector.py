# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for AIBrixKVCacheConnector using a mock Redis client."""

from unittest.mock import MagicMock, patch

import pytest

from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_redis():
    """Create a mock Redis client that stores data in a dict."""
    store: dict[str, bytes] = {}
    client = MagicMock()
    client.ping.return_value = True

    def _setex(key, ttl, value):
        store[key] = value

    def _get(key):
        return store.get(key)

    def _scan(cursor=0, match="*", count=100):
        # Simple mock: return all matching keys in one go
        import fnmatch

        matched = [k for k in store if fnmatch.fnmatch(k, match)]
        return (0, [k.encode() for k in matched])

    def _delete(*keys):
        for k in keys:
            k_str = k.decode() if isinstance(k, bytes) else k
            store.pop(k_str, None)

    client.setex.side_effect = _setex
    client.get.side_effect = _get
    client.scan.side_effect = _scan
    client.delete.side_effect = _delete
    client.close.return_value = None

    return client, store


@pytest.fixture
def aibrix_connector(mock_redis):
    """Create an AIBrixKVCacheConnector with a mocked Redis backend."""
    client, _ = mock_redis
    config = {
        "host": "127.0.0.1",
        "port": 6379,
        "key_prefix": "test",
        "ttl_seconds": 60,
        "retry_attempts": 3,
        "retry_delay_s": 0.001,  # Fast retries for tests
    }

    with patch("vllm_omni.distributed.omni_connectors.connectors.aibrix_kvcache_connector._REDIS_AVAILABLE", True), patch(
        "vllm_omni.distributed.omni_connectors.connectors.aibrix_kvcache_connector._AIBRIX_AVAILABLE", False
    ), patch("vllm_omni.distributed.omni_connectors.connectors.aibrix_kvcache_connector.redis") as mock_redis_mod:
        mock_pool = MagicMock()
        mock_redis_mod.ConnectionPool.return_value = mock_pool
        mock_redis_mod.Redis.return_value = client

        from vllm_omni.distributed.omni_connectors.connectors.aibrix_kvcache_connector import AIBrixKVCacheConnector

        connector = AIBrixKVCacheConnector(config)
        return connector


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_factory_registration():
    """AIBrixKVCacheConnector should be registered in the factory."""
    registered = OmniConnectorFactory.list_registered_connectors()
    assert "AIBrixKVCacheConnector" in registered


def test_put_get_roundtrip(aibrix_connector):
    """Test basic put/get roundtrip."""
    data = {"engine_inputs": {"prompt": "Hello"}, "sampling_params": {}}
    success, size, metadata = aibrix_connector.put("0", "1", "req_1", data)

    assert success is True
    assert size > 0
    # Deterministic keying: no metadata needed
    assert metadata is None

    result = aibrix_connector.get("0", "1", "req_1")
    assert result is not None

    retrieved_data, ret_size = result
    assert retrieved_data == data
    assert ret_size == size


def test_put_get_with_nested_data(aibrix_connector):
    """Test with complex nested data structures."""
    data = {
        "engine_inputs": {
            "prompt_token_ids": [1, 2, 3, 4, 5],
            "hidden_states": list(range(100)),
        },
        "sampling_params": {"temperature": 0.9, "top_k": 40},
        "metadata": {"stage_transition": "0->1"},
    }

    success, size, _ = aibrix_connector.put("0", "1", "req_2", data)
    assert success is True

    result = aibrix_connector.get("0", "1", "req_2")
    assert result is not None
    retrieved, _ = result
    assert retrieved["engine_inputs"]["prompt_token_ids"] == [1, 2, 3, 4, 5]


def test_get_nonexistent_key(aibrix_connector):
    """Test get() returns None for missing key after retries."""
    result = aibrix_connector.get("0", "1", "nonexistent_req")
    assert result is None
    assert aibrix_connector._metrics["timeouts"] == 1


def test_key_generation(aibrix_connector):
    """Test deterministic key format."""
    key = aibrix_connector._make_key("req_123", "0", "1")
    assert key == "test/req_123/0_to_1"

    key2 = aibrix_connector._make_key("req_456", "1", "2")
    assert key2 == "test/req_456/1_to_2"


def test_cleanup(aibrix_connector):
    """Test cleanup removes request keys."""
    # Put some data
    aibrix_connector.put("0", "1", "req_cleanup", {"data": "value"})

    # Cleanup should not raise
    aibrix_connector.cleanup("req_cleanup")


def test_health_healthy(aibrix_connector):
    """Test health returns healthy status."""
    status = aibrix_connector.health()
    assert status["status"] == "healthy"
    assert status["host"] == "127.0.0.1"
    assert status["port"] == 6379
    assert status["backend"] == "redis"


def test_health_unhealthy():
    """Test health returns unhealthy when client is None."""
    with patch("vllm_omni.distributed.omni_connectors.connectors.aibrix_kvcache_connector._REDIS_AVAILABLE", True), patch(
        "vllm_omni.distributed.omni_connectors.connectors.aibrix_kvcache_connector._AIBRIX_AVAILABLE", False
    ), patch("vllm_omni.distributed.omni_connectors.connectors.aibrix_kvcache_connector.redis") as mock_redis_mod:
        mock_redis_mod.ConnectionPool.return_value = MagicMock()
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis_mod.Redis.return_value = mock_client

        from vllm_omni.distributed.omni_connectors.connectors.aibrix_kvcache_connector import AIBrixKVCacheConnector

        connector = AIBrixKVCacheConnector({"host": "127.0.0.1"})
        connector._client = None  # Simulate disconnected state

        status = connector.health()
        assert status["status"] == "unhealthy"


def test_metrics_tracking(aibrix_connector):
    """Test that metrics are updated on put/get."""
    data = {"key": "value"}

    aibrix_connector.put("0", "1", "req_m1", data)
    assert aibrix_connector._metrics["puts"] == 1

    aibrix_connector.get("0", "1", "req_m1")
    assert aibrix_connector._metrics["gets"] == 1
    assert aibrix_connector._metrics["bytes_transferred"] > 0


def test_close(aibrix_connector):
    """Test clean shutdown."""
    aibrix_connector.close()
    assert aibrix_connector._client is None


def test_multiple_stages(aibrix_connector):
    """Test data flow across 3 stages (Thinker→Talker→Code2Wav)."""
    # Stage 0 → Stage 1
    thinker_output = {"engine_inputs": {"hidden_states": [0.1, 0.2, 0.3]}}
    success, _, _ = aibrix_connector.put("0", "1", "req_flow", thinker_output)
    assert success

    result = aibrix_connector.get("0", "1", "req_flow")
    assert result is not None
    talker_input, _ = result
    assert talker_input["engine_inputs"]["hidden_states"] == [0.1, 0.2, 0.3]

    # Stage 1 → Stage 2
    talker_output = {"engine_inputs": {"codec_codes": [10, 20, 30]}}
    success, _, _ = aibrix_connector.put("1", "2", "req_flow", talker_output)
    assert success

    result = aibrix_connector.get("1", "2", "req_flow")
    assert result is not None
    code2wav_input, _ = result
    assert code2wav_input["engine_inputs"]["codec_codes"] == [10, 20, 30]
