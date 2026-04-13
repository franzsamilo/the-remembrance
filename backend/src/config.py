import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Logging Configuration ---
def setup_logging():
    log_path = os.path.join(Config.BASE_DIR, "framework.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path)
        ]
    )


def _parse_csv_env(name: str, default: str) -> tuple[str, ...]:
    raw_value = os.getenv(name, default)
    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    return tuple(values)

# --- Configuration Settings ---
class Config:
    # Path Settings
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DOCS_DIR = os.path.join(BASE_DIR, "documents")

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
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 768))
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "distilbert")
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 50))
    INGESTION_ON_ERROR = os.getenv("INGESTION_ON_ERROR", "RAISE").upper()
    INGESTION_LLM_MAX_RETRIES = int(os.getenv("INGESTION_LLM_MAX_RETRIES", 5))
    INGESTION_LLM_BASE_DELAY = int(os.getenv("INGESTION_LLM_BASE_DELAY", 5))
    COMPGCN_HIDDEN_CHANNELS = int(os.getenv("COMPGCN_HIDDEN_CHANNELS", 256))
    COMPGCN_EPOCHS = int(os.getenv("COMPGCN_EPOCHS", 100))
    COMPGCN_LEARNING_RATE = float(os.getenv("COMPGCN_LEARNING_RATE", 0.001))
    COMPGCN_WEIGHT_DECAY = float(os.getenv("COMPGCN_WEIGHT_DECAY", 0.0001))
    COMPGCN_VALIDATION_SPLIT = float(os.getenv("COMPGCN_VALIDATION_SPLIT", 0.2))
    COMPGCN_PATIENCE = int(os.getenv("COMPGCN_PATIENCE", 20))
    COMPGCN_DROPOUT = float(os.getenv("COMPGCN_DROPOUT", 0.2))
    COMPGCN_LABEL_SMOOTHING = float(os.getenv("COMPGCN_LABEL_SMOOTHING", 0.0))
    COMPGCN_GRAD_CLIP = float(os.getenv("COMPGCN_GRAD_CLIP", 1.0))
    COMPGCN_NEG_RATIO = int(os.getenv("COMPGCN_NEG_RATIO", 10))
    COMPGCN_SEED = int(os.getenv("COMPGCN_SEED", 42))  # Set for reproducible audit runs
    LEGAL_NODE_TYPES = _parse_csv_env(
        "LEGAL_NODE_TYPES",
        "__Entity__,Entity,Method,Researcher,Dataset,Concept,Result,Metric",
    )
    LEGAL_RELATIONSHIP_TYPES = _parse_csv_env(
        "LEGAL_RELATIONSHIP_TYPES",
        "USES,CONTRADICTS,EXTENDS,PROPOSES,EVALUATES,ACHIEVES,FROM_CHUNK",
    )
    SOURCE_DOCUMENT_LABEL = os.getenv("SOURCE_DOCUMENT_LABEL", "SourceDocument")
    INGESTION_RUN_LABEL = os.getenv("INGESTION_RUN_LABEL", "IngestionRun")
    AUDIT_RUN_LABEL = os.getenv("AUDIT_RUN_LABEL", "AuditRun")
    GRAPH_READY_MIN_PROVENANCE_LINKS = int(os.getenv("GRAPH_READY_MIN_PROVENANCE_LINKS", 1))
    EVALUATION_QUERIES_FILE = os.getenv("EVALUATION_QUERIES_FILE") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation_queries.json"
    )
    EVALUATION_RESULTS_PATH = os.getenv("EVALUATION_RESULTS_PATH") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation_results.json"
    )
    
    # Framework branding (consumer-specific via env)
    API_TITLE = os.getenv("API_TITLE", "Knowledge Graph Framework API")
    FRONTEND_APP_NAME = os.getenv("FRONTEND_APP_NAME", "The Remembrance Vault")
    SHOW_DASHBOARD_STATS = os.getenv("SHOW_DASHBOARD_STATS", "false").lower() == "true"
    SYNTHESIS_PERSONA = os.getenv("SYNTHESIS_PERSONA", "Analytical Consultant")
    SYNTHESIS_FRAMEWORK_NAME = os.getenv("SYNTHESIS_FRAMEWORK_NAME", "")
    LOGGER_NAME = os.getenv("LOGGER_NAME", "kg-framework")
    SESSION_PREFIX = os.getenv("SESSION_PREFIX", "session_")

    # API Settings
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
    UPLOAD_MAX_SIZE_MB = int(os.getenv("UPLOAD_MAX_SIZE_MB", 50))
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", 60))
    RATE_LIMIT_WINDOW_SEC = int(os.getenv("RATE_LIMIT_WINDOW_SEC", 60))

    GROUNDING_MIN_SCORE = float(os.getenv("GROUNDING_MIN_SCORE", 0.95))
    TARGET_AUC_ROC = float(os.getenv("TARGET_AUC_ROC", 0.95))
    TARGET_MRR = float(os.getenv("TARGET_MRR", 0.95))
    TARGET_GROUNDING = float(os.getenv("TARGET_GROUNDING", 0.95))
    RETRIEVAL_SEED_LIMIT = int(os.getenv("RETRIEVAL_SEED_LIMIT", 100))
    RETRIEVAL_EXPANSION_LIMIT = int(os.getenv("RETRIEVAL_EXPANSION_LIMIT", 20))
    LEIDEN_RESOLUTION = float(os.getenv("LEIDEN_RESOLUTION", 1.0))

    @classmethod
    def validate(cls):
        missing = []
        if not cls.NEO4J_URI: missing.append("NEO4J_URI")
        if not cls.NEO4J_PASSWORD: missing.append("NEO4J_PASSWORD")
        if not cls.GOOGLE_API_KEY: missing.append("GOOGLE_API_KEY")

        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        # Validate Neo4j labels are safe for Cypher interpolation
        from src.helpers import validate_neo4j_label
        validate_neo4j_label(cls.SOURCE_DOCUMENT_LABEL, "SOURCE_DOCUMENT_LABEL")
        validate_neo4j_label(cls.INGESTION_RUN_LABEL, "INGESTION_RUN_LABEL")
        validate_neo4j_label(cls.AUDIT_RUN_LABEL, "AUDIT_RUN_LABEL")

# Global instances
Config.validate()
setup_logging()
logger = logging.getLogger(Config.LOGGER_NAME)

# For backward compatibility with existing code
NEO4J_URI = Config.NEO4J_URI
NEO4J_USERNAME = Config.NEO4J_USERNAME
NEO4J_PASSWORD = Config.NEO4J_PASSWORD
NEO4J_DATABASE = Config.NEO4J_DATABASE
GOOGLE_API_KEY = Config.GOOGLE_API_KEY
