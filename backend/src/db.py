"""Shared Neo4j driver management for the Knowledge Graph Framework backend."""
from neo4j import GraphDatabase

from src.config import Config


class DatabaseManager:
    """Singleton Neo4j driver. Reuse across retriever, API, etc."""
    _driver = None

    @classmethod
    def get_driver(cls):
        if cls._driver is None:
            cls._driver = GraphDatabase.driver(
                Config.NEO4J_URI,
                auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD),
                # Aura Free idles TCP connections at ~5 min; cap pool connection age
                # so new sessions open fresh sockets after long compute-only gaps.
                max_connection_lifetime=240,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60,
            )
        return cls._driver

    @classmethod
    def close(cls):
        if cls._driver:
            cls._driver.close()
            cls._driver = None

    @classmethod
    def refresh(cls):
        """Force a fresh driver. Call before writes that follow long idle periods
        (e.g., after multi-minute CompGCN training) where Aura Free may have
        closed the underlying TCP socket."""
        cls.close()
        return cls.get_driver()
