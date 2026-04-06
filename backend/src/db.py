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
            )
        return cls._driver

    @classmethod
    def close(cls):
        if cls._driver:
            cls._driver.close()
            cls._driver = None
