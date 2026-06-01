"""
Configuration settings for diagnosis STYLE training.
"""

import torch
import os


class DiagnosisConfig:
    """Configuration class for diagnosis STYLE training."""
    
    # Device settings
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    MEMORY_SIZE = 10000
    NUM_EPOCHS = 100
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.1
    
    # DISP model parameters
    DISP_HIDDEN_SIZE = 256
    NUM_ACTIONS = 2  # 0: Ask follow-up question, 1: Provide diagnosis
    
    # Training parameters
    GAMMA = 0.99
    EPSILON = 1.0
    EPSILON_START = 1.0
    EPSILON_END = 0.1
    EPSILON_MIN = 0.05
    EPSILON_DECAY = 0.999
    TARGET_UPDATE = 10
    GRADIENT_ACCUMULATION_STEPS = 1
    RANDOM_SEED = 42
    NUM_WORKERS = 4
    MAX_TURNS = 10
    MAX_STEPS = 10
    
    # Diagnosis-specific reward parameters
    SUCCESS_REWARD = 2.0  # Large reward for correct diagnosis
    FAILURE_REWARD = -0.5  # Penalty for incorrect diagnosis
    STEP_PENALTY = -0.01
    REWARD_ASK = 0.5  # Reward for asking follow-up when needed
    REWARD_CLARIFY = 0.3  # Reward for clarifying questions
    REWARD_ANSWER = 1.0  # Reward for providing diagnosis
    REWARD_INVALID = -0.3  # Penalty for asking follow-up when not needed
    REWARD_TIMEOUT = -0.5  # Penalty for timing out
    REWARD_SUCCESS = 3.0  # Large reward for successful completion
    ENTROPY_BONUS = 0.01
    
    # Diagnosis-specific parameters
    DIAGNOSIS_TOP_K = 5
    SYMPTOM_SIMILARITY_THRESHOLD = 0.7
    FOLLOWUP_THRESHOLD = 0.3  # Threshold for determining if follow-up is needed
    
    # Retrieval parameters
    TOP_K = 5
    SIMILARITY_THRESHOLD = 0.7
    NUM_RETRIEVED_DOCS = 5
    RETRIEVAL_THRESHOLD = 0.5
    
    # LLM integration
    OPENAI_API_KEY = None
    MODEL_NAME = "gpt-3.5-turbo"
    MAX_TOKENS = 150
    TEMPERATURE = 0.7
    
    # Monitoring
    WANDB_PROJECT = "STYLE_Diagnosis"
    WANDB_ENTITY = "ashteam"
    LOG_INTERVAL = 100
    
    # File paths
    CHECKPOINT_DIR = "checkpoints"
    DATA_DIR = "data"
    MODEL_DIR = "models"
    
    # Diagnosis-specific paths
    DIAGNOSIS_DATA_PATH = "/home/ailaty3088@id.sdsu.edu/STYLE/data/train/diagnosis_train.csv"
    DIAGNOSIS_KB_PATH = "data/diagnosis_documents.csv"
    OUTPUT_DIR = "diagnosis_training_output"
    
    # Training monitoring
    SAVE_EVERY_EPOCH = 10
    VALIDATION_EVERY_EPOCH = 5
    EARLY_STOPPING_PATIENCE = 20
    
    def __init__(self):
        """Initialize configuration with environment variables."""
        # Load environment variables if available
        if os.path.exists('.env'):
            from dotenv import load_dotenv
            load_dotenv()
        
        # Override with environment variables if they exist
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', self.OPENAI_API_KEY)
        self.WANDB_ENTITY = os.getenv('WANDB_ENTITY', self.WANDB_ENTITY)
        
        # Set random seed for reproducibility
        torch.manual_seed(self.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.RANDOM_SEED)
    
    def update_from_args(self, args):
        """Update configuration from command line arguments."""
        if hasattr(args, 'batch_size'):
            self.BATCH_SIZE = args.batch_size
        if hasattr(args, 'learning_rate'):
            self.LEARNING_RATE = args.learning_rate
        if hasattr(args, 'epochs'):
            self.NUM_EPOCHS = args.epochs
        if hasattr(args, 'output_dir'):
            self.OUTPUT_DIR = args.output_dir
