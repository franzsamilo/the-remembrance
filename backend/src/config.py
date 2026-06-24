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
    # Training loss: "bce" (BCEWithLogits, proved 0.9646 AUC baseline) or
    # "bpr" (pairwise Bayesian Personalized Ranking -log sigmoid(pos-neg); targets MRR)
    COMPGCN_LOSS = os.getenv("COMPGCN_LOSS", "bce").lower()
    COMPGCN_BPR_MARGIN = float(os.getenv("COMPGCN_BPR_MARGIN", 0.0))
    # Negative sampling strategy: "uniform" (draw from all nodes) or
    # "type_aware" (draw only from nodes sharing the corrupted endpoint's
    # schema label). Default is uniform — Run 7 (2026-04-19) confirmed
    # type-aware does not lift MRR on this corpus (label skew: 54% Concept).
    # Opt in via env var if exploring self-adversarial or per-type schemes.
    COMPGCN_NEG_SAMPLING = os.getenv("COMPGCN_NEG_SAMPLING", "uniform").lower()
    # AUC-ROC guardrail: if the trained model scores below this on validation,
    # run_audit skips the Neo4j score write-back so a regressed model never
    # clobbers production plausibility values. Paper target is 0.95.
    COMPGCN_AUC_GUARDRAIL = float(os.getenv("COMPGCN_AUC_GUARDRAIL", 0.95))
    # Self-adversarial negative weighting (RotatE Sun+ 2019, eq. 5). For each
    # positive, weight its K=neg_ratio negatives by softmax(α * neg_score).
    # α=0 disables (uniform-mean BPR — Run 6/7 reproduction). α=1.0 is the
    # RotatE canonical value; Run 8 uses 1.0.
    COMPGCN_ADV_TEMP = float(os.getenv("COMPGCN_ADV_TEMP", 0.0))
    # Decoder choice for CompGCN. "distmult" (Runs 1-8 default) computes
    # s(h,r,t) = sum(h * r * t). "rotate" computes -||h o r - t||_2 with
    # relations parameterized as phase angles in [-pi, pi] over 128 complex
    # dimensions (split halves of the 256-dim real encoder output).
    # Reference: Sun et al. 2019, "RotatE: Knowledge Graph Embedding by
    # Relational Rotation in Complex Space".
    COMPGCN_DECODER = os.getenv("COMPGCN_DECODER", "distmult").lower()
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
    # Default to the legal query set — the corpus is Philippine jurisprudence,
    # not research papers. Override with EVALUATION_QUERIES_FILE / EVALUATION_QUERIES.
    EVALUATION_QUERIES_FILE = os.getenv("EVALUATION_QUERIES_FILE") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation_queries_legal.json"
    )
    EVALUATION_RESULTS_PATH = os.getenv("EVALUATION_RESULTS_PATH") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation_results.json"
    )
    # Cap concurrent eval pipelines (synthesis + grounding judge + faithfulness judge
    # per query) so we don't trip Gemini's per-minute rate limits during /evaluate.
    EVALUATION_MAX_CONCURRENCY = int(os.getenv("EVALUATION_MAX_CONCURRENCY", 3))
    
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
    # Seed pool for nearest-neighbour ranking. Aura-free has no vector index, so seeds
    # are ranked in python over the fetched candidates — this MUST exceed the retrievable
    # node count (~5,500) so ranking sees ALL candidates, not an arbitrary subset.
    # A small limit (was 100) silently starved synthesis of the right triplets and
    # capped grounding ~0.96; exhaustive ranking restores grounding >0.98 (validated
    # via run_logs/reroll_retrieval_sweep.py + reroll_grounding_confirm.py).
    RETRIEVAL_SEED_LIMIT = int(os.getenv("RETRIEVAL_SEED_LIMIT", 6000))
    # Tight expansion: with good seeds, 10 triplets ground best; more breadth makes the
    # synthesiser generalise beyond the triplets and DROPS grounding (exp=40 -> 0.87).
    RETRIEVAL_EXPANSION_LIMIT = int(os.getenv("RETRIEVAL_EXPANSION_LIMIT", 10))
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
