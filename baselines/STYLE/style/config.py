"""
Configuration settings for STYLE (paper-aligned defaults, Appendix F.1).
"""

import torch
import os
from dotenv import load_dotenv


class Config:
    """STYLE configuration aligned with the original paper."""

    _env_loaded = False

    # Action space: 0 = answer, 1 = ask (binary strategy planner)
    ACTION_ANSWER = 0
    ACTION_ASK = 1
    NUM_ACTIONS = 2

    # Model / DISP
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    MEMORY_SIZE = 10000
    NUM_EPOCHS = 1800
    HIDDEN_DIM = 256
    DISP_HIDDEN_SIZE = 256
    DISP_DROPOUT = 0.1
    NUM_LAYERS = 2
    DROPOUT = 0.1

    # BERT encoder (frozen, first 3 layers — Section 3.3)
    BERT_MODEL_NAME = "OpenMatch/cocodr-base-msmarco"
    BERT_LAYERS = 3
    BERT_HIDDEN_SIZE = 768
    MAX_QUERY_LENGTH = 512
    MAX_DOC_LENGTH = 512

    # Domains
    DOMAINS = ["clariq", "opendialkg"]

    # RL training
    GAMMA = 0.99
    EPSILON = 1.0
    EPSILON_START = 1.0
    EPSILON_END = 0.1
    EPSILON_MIN = 0.05
    EPSILON_DECAY = 0.995
    TARGET_UPDATE = 50
    RANDOM_SEED = 42
    NUM_WORKERS = 4
    MAX_TURNS = 10
    MAX_STEPS = 10

    # Paper reward structure (sparse)
    SUCCESS_REWARD = 1.0
    REWARD_TIMEOUT = -0.5
    INTERMEDIATE_REWARD = 0.0

    # Legacy aliases kept for compatibility
    FAILURE_REWARD = REWARD_TIMEOUT
    REWARD_SUCCESS = SUCCESS_REWARD
    MAX_TURN_PENALTY = REWARD_TIMEOUT

    # Retrieval
    TOP_K = 5
    TOP_K_DOCS = 5
    NUM_RETRIEVED_DOCS = 5
    SIMILARITY_THRESHOLD = 0.7
    RETRIEVAL_THRESHOLD = 0.5

    # LLM (inference / user simulation — Section 3.5)
    OPENAI_API_KEY = None
    MODEL_NAME = "gpt-3.5-turbo"
    MAX_TOKENS = 150
    TEMPERATURE = 0.7

    # Monitoring
    WANDB_PROJECT = "STYLE"
    WANDB_ENTITY = None
    LOG_INTERVAL = 100
    DEBUG = False

    # Paths
    CHECKPOINT_DIR = "checkpoints"
    MODEL_SAVE_PATH = "models/saved"
    MODEL_SAVE_DIR = "saved_models"
    MODEL_DIR = "models"
    DATA_DIR = "data"
    LOG_DIR = "logs"

    # Evaluation
    EVAL_INTERVAL = 100
    EVAL_EPISODES = 10
    METRICS = ["sr@k", "recall@5", "avg_turns", "strategy_diversity"]
    METRICS_WINDOW = 100

    # Dataset splits
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1

    DQN_BUFFER_SIZE = 10000
    DISCOUNT_FACTOR = 0.99
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def state_input_dim(cls, hidden_size=None, top_k=None):
        """Input dim: H_t + D_t (k docs) + k scores."""
        h = hidden_size or cls.BERT_HIDDEN_SIZE
        k = top_k or cls.TOP_K_DOCS
        return h + h * k + k

    def __init__(self):
        if not Config._env_loaded:
            env_path = os.path.join(os.getcwd(), ".env")
            if os.path.exists(env_path):
                load_dotenv(env_path, override=True)
            Config._env_loaded = True

        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not self.OPENAI_API_KEY:
            print(
                "Warning: OPENAI_API_KEY not set. "
                "LLM question generation and user simulation will fail until configured."
            )

        self.MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", self.MODEL_NAME)
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.WANDB_ENTITY = os.getenv("WANDB_ENTITY", self.WANDB_ENTITY)
        self.STATE_INPUT_DIM = self.state_input_dim()
