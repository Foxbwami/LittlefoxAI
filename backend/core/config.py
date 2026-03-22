import os

# backend/ core config -> base is backend/
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Load HF token from backend/HF.env if present
_HF_ENV_PATH = os.path.join(BASE_DIR, "HF.env")
if os.path.exists(_HF_ENV_PATH) and "HF_TOKEN" not in os.environ:
    try:
        with open(_HF_ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key and value and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        pass
DATA_DIR = os.path.join(BASE_DIR, "data")

BLOCK_SIZE = 192
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 3e-4
MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "model.pth")
BPE_VOCAB_SIZE = 1000
EMBED_SIZE = 128
HEADS = 4
LAYERS = 4
TEMPERATURE = 0.9
TOP_K = 40
PROMPT_PREFIX = (
    "You are a helpful assistant. Be accurate, concise, and grounded. "
    "Use provided context when available and do not invent facts. "
    "If you are unsure or the context is missing, say so and ask a clarifying question. "
    "For sourced answers, include brief inline citations like [1], [2] tied to sources."
)
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "data.txt")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "cleaned.txt")
TOKENIZER_PATH = os.path.join(DATA_DIR, "processed", "tokenizer.json")
PERSONALITY_PATH = os.path.join(DATA_DIR, "personality.txt")
SHUFFLE_BUFFER = 1024
MAX_HISTORY_TURNS = 10
RAW_MAX_CHARS = 8000000
MAX_STEPS_PER_EPOCH = 400
TOKENIZER_TRAIN_CHARS = 100000
MEMORY_MAX_TOKENS = 80
MEMORY_TOP_K = 5
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
EMBEDDING_DIM = 384
VECTOR_INDEX_PATH = os.path.join(DATA_DIR, "memory", "faiss.index")
VECTOR_META_PATH = os.path.join(DATA_DIR, "memory", "texts.pkl")
MEMORY_MAX_PER_USER = 2000
MEMORY_PRUNE_TARGET = 1500
MEMORY_MIN_IMPORTANCE = 0.1
MEMORY_PRUNE_EVERY = 50
BROWSE_ON_QUESTION = True
WEB_MAX_CHARS = 800
WEB_TIMEOUT = 8
WEB_TOTAL_TIMEOUT = 18
LEARNING_LOG_PATH = os.path.join(DATA_DIR, "learning_data.txt")
RETRAIN_EPOCHS = 3
RETRAIN_STEPS = 200
SEARCH_INDEX_DIR = os.path.join(DATA_DIR, "search_index")
CRAWL_WORKERS = 4
CRAWL_TIMEOUT = 5
CRAWL_MAX_PAGES = 30
HYBRID_TOP_K = 5
WEB_SEARCH_TOP_K = 5
FEEDBACK_PATH = os.path.join(DATA_DIR, "feedback.txt")
VECTOR_SAVE_EVERY = 20
POSTPROCESS_SUMMARIZE = False
POSTPROCESS_MAX_CHARS = 600
MIN_SOURCE_RELEVANCE = 0.25
ENABLE_PII_REDACTION = True
ALLOW_EMBEDDINGS_FALLBACK = True
ALLOW_HASH_EMBEDDINGS = False
EMBEDDINGS_DISABLED = False
PROMPT_MAX_WORDS = 140
HF_USE_PIPELINE = True
HF_GENERATION_TASK = "text2text-generation"
HF_GENERATION_MODEL = os.path.join(BASE_DIR, "models", "hf_finetuned")
HF_FALLBACK_MODEL = "google/flan-t5-base"
HF_SUMMARY_MODEL = "google/flan-t5-base"
HF_MAX_NEW_TOKENS = 80
HF_TEMPERATURE = 0.7
HF_TOP_P = 0.9
HF_USE_SUMMARY = False
HF_SUMMARY_MAX_NEW_TOKENS = 80
HF_CONTEXT_SENTENCES = 3
SEARCH_EXTRACTIVE_ONLY = True
SEARCH_CONTEXT_TOP_K = 3
RERANK_CANDIDATES = 40
RERANK_BM25_WEIGHT = 0.55
RERANK_EMBED_WEIGHT = 0.45
ACADEMIC_CITATION_STYLE = "APA"
ACADEMIC_STRICT_SOURCES = False
ACADEMIC_TEMPLATE_DEFAULT = "summary"
HUMANIZER_ENABLED = False
HUMANIZER_USE_MODEL = False
HUMANIZER_PERSONALITY = "confident, calm, slightly witty, mentor-like"
HUMANIZER_MAX_CHARS = 500
HUMANIZER_MODEL_PATH = os.path.join(BASE_DIR, "humanizer", "model")
HUMANIZER_BASE_MODEL = "sshleifer/tiny-gpt2"
COGNITIVE_ADAPTER_ENABLED = True
COGNITIVE_FORCE_WEB = False
DECISION_MODEL_ENABLED = True
DECISION_MODEL_NAME = "google/flan-t5-small"
PLANNER_MAX_STEPS = 6
TOOLCHAIN_ENABLED = True
AGENT_ENABLED = False
RETRAIN_EVERY_INTERACTIONS = 250
RETRAIN_SIGNAL_PATH = os.path.join(DATA_DIR, "retrain.signal")
ALLOW_CODE_EXECUTION = True
CODE_EXEC_TIMEOUT = 4
MATH_RENDER = True
NER_USE_SPACY = False
NER_SPACY_MODEL = "en_core_web_sm"
SAFETY_USE_MODEL = False
SAFETY_MODEL_NAME = "unitary/toxic-bert"
SAFETY_BLOCK_THRESHOLD = 0.75
SAFETY_BLOCKLIST = [
    "kill",
    "harm",
    "suicide",
    "terrorist",
    "bomb",
    "weapon",
    "hate",
]
FACT_CHECK_ALWAYS = False
VISION_ENABLED = False
SPEECH_ENABLED = False
OCR_ENABLED = False
LOG_TIMINGS = True
LOG_PATH = os.path.join(DATA_DIR, "app.log")

# Profile DB (sqlite)
PROFILE_DB_PATH = os.path.join(DATA_DIR, "memory.db")
