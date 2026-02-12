import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Logging Configuration ---
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("framework.log")
        ]
    )

# --- Configuration Settings ---
class Config:
    # Neo4j Settings
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
    
    # AI/ML Settings
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
    DISTILBERT_MODEL = os.getenv("DISTILBERT_MODEL", "distilbert-base-nli-stsb-mean-tokens")
    
    # API Settings
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    @classmethod
    def validate(cls):
        missing = []
        if not cls.NEO4J_URI: missing.append("NEO4J_URI")
        if not cls.NEO4J_PASSWORD: missing.append("NEO4J_PASSWORD")
        if not cls.GOOGLE_API_KEY: missing.append("GOOGLE_API_KEY")
        
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

# Global instances
Config.validate()
setup_logging()
logger = logging.getLogger("the-remembrance")

# For backward compatibility with existing code
NEO4J_URI = Config.NEO4J_URI
NEO4J_USERNAME = Config.NEO4J_USERNAME
NEO4J_PASSWORD = Config.NEO4J_PASSWORD
NEO4J_DATABASE = Config.NEO4J_DATABASE
GOOGLE_API_KEY = Config.GOOGLE_API_KEY
