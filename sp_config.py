"""
BFCL multi_turn_long_context semi-prefill benchmark configuration.

This bench targets cw=32000, threshold=30000, and C1 retention in the
2000-3000 token range.  It uses the global Python environment and writes every
run into a named subdirectory under results/.
"""

from pathlib import Path

BENCH_ROOT = Path(__file__).parent.resolve()
RESULTS_DIR = BENCH_ROOT / "results"
SAMPLE_IDS_FILE = BENCH_ROOT / "sample_ids.json"
AUGMENTED_DATA_FILE = BENCH_ROOT / "data" / "BFCL_v4_multi_turn_long_context_sp.json"

# ============================================================
# Model / endpoint
# ============================================================

MODEL_REGISTRY = {
    "GLM-4-9B-0414": {
        "model_path": "/root/share/models/GLM-4-9B-0414",
        "tokenizer_path": "/root/share/models/GLM-4-9B-0414",
        "notes": "Use vllm_tool_proxy --tool-parser auto --native-template.",
    },
    "Llama-3.3-70B-Instruct": {
        "model_path": "/root/share/models/Llama-3.3-70B-Instruct",
        "tokenizer_path": "/root/share/models/Llama-3.3-70B-Instruct",
        "notes": "Default model for this bench.",
    },
    "Qwen3-235B-A22B": {
        "model_path": "/root/share/models/Qwen3-235B-A22B",
        "tokenizer_path": "/root/share/models/Qwen3-235B-A22B",
        "notes": "Native tool template is expected through vllm_tool_proxy.",
    },
}

DEFAULT_MODEL = "Llama-3.3-70B-Instruct"
MODEL_KEY = DEFAULT_MODEL
MODEL_PATH = MODEL_REGISTRY[DEFAULT_MODEL]["model_path"]
TOKENIZER_PATH = MODEL_REGISTRY[DEFAULT_MODEL]["tokenizer_path"]

VLLM_URL = "http://localhost:8005/v1"
PROXY_URL = "http://localhost:6003/v1"
API_KEY = "EMPTY"
TEMPERATURE = 0.001

# ============================================================
# Data source: BFCL multi_turn_long_context
# ============================================================

BFCL_ROOT = Path("/root/gorilla/berkeley-function-call-leaderboard")
TEST_CATEGORY = "multi_turn_long_context"
ORIGINAL_DATA_FILE = BFCL_ROOT / "bfcl_eval" / "data" / f"BFCL_v4_{TEST_CATEGORY}.json"
# Runner default output/input path. run.py falls back to ORIGINAL_DATA_FILE only
# when this augmented file has not been generated and --prepare-data is not used.
DATA_FILE = AUGMENTED_DATA_FILE

FUNC_DOC_DIR = BFCL_ROOT / "bfcl_eval" / "data" / "multi_turn_func_doc"

# ============================================================
# Compression presets
# ============================================================
# Each preset: (name, context_window, reserve_tokens, keep_recent_tokens_budget,
#              summary_max_tokens). threshold = context_window - reserve_tokens.
# The primary preset is exactly cw32000/thr30000/C1_budget=2600.

PRESETS = [
    ("cw32k_thr30000_c12600", 32000, 2000, 2600, 1024),
    ("cw32k_thr30000_c12200", 32000, 2000, 2200, 1024),
    ("cw32k_thr30000_c13000", 32000, 2000, 3000, 1024),
]
PRIMARY_PRESET = "cw32k_thr30000_c12600"

C1_MIN_TOKENS = 2000
C1_MAX_TOKENS = 3000
P_TARGET_1_5 = 0.2
P_TARGET_1_6 = 1.0 / 6.0

# Dataset augmentation defaults. The initial target includes message + tool
# schema tokens; it stays below 30000 so turn 1 can enter the model. The per-turn
# growth target is close to the primary C1 budget so the current turn can form
# a 2000-3000 token retained segment when boundary compression triggers.
AUGMENT_INITIAL_TARGET_TOKENS = 26000
AUGMENT_TURN_GROWTH_TOKENS = 2600
AUGMENT_MIN_TURNS = 6
AUGMENT_MAX_SAMPLES = 0
AUGMENT_SEED = 20260505

# A small search subset: worst-case IDs observed in prior BFCL semi-prefill runs.
SEARCH_SAMPLE_IDS = [
    "multi_turn_long_context_110",
    "multi_turn_long_context_196",
    "multi_turn_long_context_174",
    "multi_turn_long_context_184",
    "multi_turn_long_context_125",
    "multi_turn_long_context_131",
]

SUMMARY_MAX_RETRIES = 0
USE_STRUCTURED_INSTRUCTIONS = True
PRESERVED_RECENT_TURNS = 1
STREAM_MAX_RETRIES = 2
STREAM_RETRY_BACKOFF_S = 1.0
NUM_THREADS = 1
MAX_SAMPLES_DEFAULT = None
