# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
AIBrix KVCache Connector for vLLM-Omni OmniConnector system.

Uses AIBrix KVCache (PrisKV) as a distributed key-value store for
cross-node stage data transfer. Supports TCP and optional RDMA transport.

Requirements:
    pip install aibrix-kvcache   # or: pip install redis
"""

import time
from typing import Any

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)

# Try importing the AIBrix KVCache client.
# Fall back to a generic Redis client since PrisKV is Redis-protocol compatible.
_AIBRIX_AVAILABLE = False
_REDIS_AVAILABLE = False

try:
    from aibrix.kvcache.connector import KVCacheConnector

    _AIBRIX_AVAILABLE = True
except ImportError:
    KVCacheConnector = None

try:
    import redis

    _REDIS_AVAILABLE = True
except ImportError:
    redis = None


class AIBrixKVCacheConnector(OmniConnectorBase):
    """
    OmniConnector backed by AIBrix KVCache (PrisKV).

    PrisKV exposes a Redis-compatible protocol, so this connector works with
    either the native ``aibrix-kvcache`` client or a plain ``redis-py`` client.

    Config keys (passed via ``extra`` in YAML):
        host            PrisKV server address (default: "127.0.0.1")
        port            PrisKV server port (default: 6379)
        password        Optional auth password
        pool_size       Connection pool size (default: 8)
        key_prefix      Key namespace (default: "omni")
        ttl_seconds     Auto-expire keys after N seconds (default: 300)
        retry_attempts  Number of get() retries (default: 40)
        retry_delay_s   Delay between retries in seconds (default: 0.05)
        use_mput_mget   Use batch operations if available (default: True)
        socket_timeout  Socket timeout in seconds (default: 5.0)
    """

    def __init__(self, config: dict[str, Any]):
        if not _AIBRIX_AVAILABLE and not _REDIS_AVAILABLE:
            raise ImportError(
                "AIBrix KVCache connector requires either 'aibrix-kvcache' "
                "or 'redis' package. Install with: pip install redis"
            )

        self.config = config
        self.host = config.get("host", "127.0.0.1")
        self.port = int(config.get("port", 6379))
        self.password = config.get("password", None)
        self.pool_size = int(config.get("pool_size", 8))
        self.key_prefix = config.get("key_prefix", "omni")
        self.ttl_seconds = int(config.get("ttl_seconds", 300))
        self.retry_attempts = int(config.get("retry_attempts", 40))
        self.retry_delay_s = float(config.get("retry_delay_s", 0.05))
        self.use_mput_mget = config.get("use_mput_mget", True)
        self.socket_timeout = float(config.get("socket_timeout", 5.0))

        self._client = None
        self._metrics = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
            "errors": 0,
            "timeouts": 0,
        }

        self._init_client()

    # ------------------------------------------------------------------
    # Client initialization
    # ------------------------------------------------------------------

    def _init_client(self) -> None:
        """Initialize the KVCache client (AIBrix native or Redis fallback)."""
        try:
            if _AIBRIX_AVAILABLE:
                self._client = KVCacheConnector.from_envs()
                logger.info(
                    "AIBrixKVCacheConnector: using native aibrix-kvcache client (%s:%s)",
                    self.host,
                    self.port,
                )
            elif _REDIS_AVAILABLE:
                pool = redis.ConnectionPool(
                    host=self.host,
                    port=self.port,
                    password=self.password,
                    max_connections=self.pool_size,
                    socket_timeout=self.socket_timeout,
                    socket_connect_timeout=self.socket_timeout,
                    decode_responses=False,
                )
                self._client = redis.Redis(connection_pool=pool)
                # Verify connectivity
                self._client.ping()
                logger.info(
                    "AIBrixKVCacheConnector: using redis client (%s:%s)",
                    self.host,
                    self.port,
                )
        except Exception as e:
            logger.error("Failed to initialize AIBrix KVCache client: %s", e)
            raise

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    def _make_key(self, request_id: str, from_stage: str, to_stage: str) -> str:
        """Deterministic key from request_id and stage edge."""
        return f"{self.key_prefix}/{request_id}/{from_stage}_to_{to_stage}"

    # ------------------------------------------------------------------
    # OmniConnectorBase interface
    # ------------------------------------------------------------------

    def put(
        self, from_stage: str, to_stage: str, request_id: str, data: Any
    ) -> tuple[bool, int, dict[str, Any] | None]:
        if not self._client:
            logger.error("AIBrixKVCacheConnector: client not initialized")
            return False, 0, None

        try:
            serialized = self.serialize_obj(data)
            key = self._make_key(request_id, from_stage, to_stage)

            if _AIBRIX_AVAILABLE and hasattr(self._client, "put"):
                self._client.put(key, serialized)
            else:
                # Redis fallback: SET with TTL
                self._client.setex(key, self.ttl_seconds, serialized)

            size = len(serialized)
            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += size

            logger.debug(
                "AIBrixKVCacheConnector: stored %s (%s -> %s) %d bytes",
                key,
                from_stage,
                to_stage,
                size,
            )
            # Deterministic keying — no metadata needed
            return True, size, None

        except Exception as e:
            self._metrics["errors"] += 1
            logger.error("AIBrixKVCacheConnector put failed: %s", e)
            return False, 0, None

    def get(
        self, from_stage: str, to_stage: str, request_id: str, metadata: dict[str, Any] | None = None
    ) -> tuple[Any, int] | None:
        if not self._client:
            logger.error("AIBrixKVCacheConnector: client not initialized")
            return None

        key = self._make_key(request_id, from_stage, to_stage)

        for attempt in range(self.retry_attempts):
            try:
                if _AIBRIX_AVAILABLE and hasattr(self._client, "get"):
                    raw = self._client.get(key)
                else:
                    raw = self._client.get(key)

                if raw:
                    data = self.deserialize_obj(raw)
                    size = len(raw)
                    self._metrics["gets"] += 1

                    logger.debug(
                        "AIBrixKVCacheConnector: retrieved %s (%s -> %s) %d bytes",
                        key,
                        from_stage,
                        to_stage,
                        size,
                    )
                    return data, size

            except Exception as e:
                logger.debug("AIBrixKVCacheConnector get attempt %d failed: %s", attempt, e)

            if attempt < self.retry_attempts - 1:
                time.sleep(self.retry_delay_s)

        self._metrics["timeouts"] += 1
        logger.warning("AIBrixKVCacheConnector: timeout waiting for %s after %d attempts", key, self.retry_attempts)
        return None

    def cleanup(self, request_id: str) -> None:
        """Delete all keys for a request.

        With TTL-based expiry this is best-effort; keys auto-expire even if
        cleanup is never called (e.g. on crash).
        """
        if not self._client:
            return

        try:
            # Scan for all keys matching this request
            pattern = f"{self.key_prefix}/{request_id}/*"
            if hasattr(self._client, "delete"):
                # Use SCAN to find matching keys (non-blocking)
                cursor = 0
                while True:
                    cursor, keys = self._client.scan(cursor=cursor, match=pattern, count=100)
                    if keys:
                        self._client.delete(*keys)
                    if cursor == 0:
                        break
            logger.debug("AIBrixKVCacheConnector: cleaned up keys for request %s", request_id)
        except Exception as e:
            logger.warning("AIBrixKVCacheConnector: cleanup failed for %s: %s", request_id, e)

    def health(self) -> dict[str, Any]:
        if not self._client:
            return {"status": "unhealthy", "error": "Client not initialized"}

        try:
            if hasattr(self._client, "ping"):
                self._client.ping()
            return {
                "status": "healthy",
                "host": self.host,
                "port": self.port,
                "backend": "aibrix-kvcache" if _AIBRIX_AVAILABLE else "redis",
                **self._metrics,
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e), **self._metrics}

    def close(self) -> None:
        """Clean shutdown."""
        if self._client:
            try:
                if hasattr(self._client, "close"):
                    self._client.close()
                self._client = None
                logger.info("AIBrixKVCacheConnector closed")
            except Exception as e:
                logger.error("Error closing AIBrix KVCache client: %s", e)
