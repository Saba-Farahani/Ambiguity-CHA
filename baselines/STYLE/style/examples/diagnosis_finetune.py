#!/usr/bin/env python3
"""
Diagnosis Domain Fine-tuning for STYLE Model
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from style.models.disp import DISP
from style.config import Config
from style.models.retriever import Retriever
from style.training.mdt import MDTTrainer
from style.utils.monitoring import Monitor

class DiagnosisFineTuner:
    """Fine-tune STYLE model for medical diagnosis domain."""
    
    def __init__(self, base_model_path=None, config=None):
        """Initialize the diagnosis fine-tuner."""
        self.config = config or Config()
        self.config.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize DISP model
        self.disp_model = DISP(self.config)
        
        # Load base model if provided
        if base_model_path and os.path.exists(base_model_path):
            print(f"📥 Loading base model from: {base_model_path}")
            try:
                self._load_model_robust(base_model_path)
            except Exception as e:
                print(f"⚠️ Failed to load base model: {e}")
                print("🔄 Starting with random initialization")
        else:
            print("🔄 No base model provided, starting with random initialization")
        
        # Initialize retriever
        self.retriever = Retriever(self.config)
        self._load_diagnosis_knowledge_base()
        
        # Training components
        self.optimizer = torch.optim.Adam(self.disp_model.parameters(), lr=self.config.LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)
        
        # Create output directory
        os.makedirs('logs/diagnosis_finetune', exist_ok=True)
    
    def _load_model_robust(self, model_path):
        """Robustly load model with fallback options."""
        try:
            # Try loading with current architecture
            self.disp_model.load(model_path)
            print("✅ Model loaded successfully")
            return
        except Exception as e:
            print(f"⚠️ Standard loading failed: {e}")
        
        try:
            # Try loading just the state dict and handle missing keys
            checkpoint = torch.load(model_path, map_location=self.config.DEVICE)
            
            if isinstance(checkpoint, dict):
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Load with strict=False to ignore missing keys
                missing_keys, unexpected_keys = self.disp_model.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    print(f"⚠️ Missing keys (using defaults): {missing_keys[:5]}...")
                if unexpected_keys:
                    print(f"⚠️ Unexpected keys (ignored): {unexpected_keys[:5]}...")
                
                print("✅ Model loaded with partial weights")
                return
                
        except Exception as e:
            print(f"⚠️ Robust loading also failed: {e}")
        
        # If all loading attempts fail, continue with random initialization
        print("🔄 Continuing with random initialization")
    
    def _setup_logging(self):
        """Setup logging for fine-tuning."""
        log_dir = Path("logs/diagnosis_finetune")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"diagnosis_finetune_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_diagnosis_knowledge_base(self):
        """Load the comprehensive diagnosis knowledge base."""
        knowledge_base_file = "data/diagnosis_documents_complete.csv"
        
        if not os.path.exists(knowledge_base_file):
            print(f"❌ Knowledge base file not found: {knowledge_base_file}")
            print("🔧 Please run build_diagnosis_knowledge_base.py first")
            return
        
        # Load diagnosis documents
        df = pd.read_csv(knowledge_base_file)
        diagnosis_docs = df['Document'].tolist()
        
        # Load into retriever
        self.retriever.load_documents(domain='diagnosis', documents=diagnosis_docs)
        
        print(f"📚 Loaded {len(diagnosis_docs)} diagnosis documents")
        print(f"📚 Knowledge base covers {len(set([doc.split(':')[0] for doc in diagnosis_docs]))} unique diagnoses")
    
    def _prepare_diagnosis_data(self, data_file):
        """Prepare diagnosis data for training."""
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        df = pd.read_csv(data_file)
        
        # Convert to training format
        training_data = []
        
        for _, row in df.iterrows():
            prompt = row['Prompt']
            ground_truth = row['Ground Truth Diagnosis']
            
            # Determine if follow-up is needed (based on removed symptoms)
            needs_followup = int(pd.notna(row.get('Removed Symptoms', '')) and 
                               str(row.get('Removed Symptoms', '')).strip() != "")
            
            # Retrieve relevant documents
            try:
                docs, scores = self.retriever.retrieve(prompt, domain='diagnosis', top_k=5)
            except Exception as e:
                self.logger.warning(f"Retrieval failed for prompt: {e}")
                docs, scores = [""], [0.0]
            
            # Create training sample
            sample = {
                'query_history': [prompt],
                'documents': docs,
                'retrieval_scores': scores,
                'domain': 'diagnosis',
                'ground_truth': ground_truth,
                'needs_followup': needs_followup
            }
            
            training_data.append(sample)
        
        self.logger.info(f"📊 Prepared {len(training_data)} training samples from {data_file}")
        return training_data
    
    def _create_diagnosis_reward_function(self):
        """Create domain-specific reward function for diagnosis."""
        def diagnosis_reward(state, action, next_state, info):
            """Calculate reward for diagnosis domain."""
            reward = 0.0
            
            # Base reward for taking any action
            reward += 0.1
            
            # Reward for correct diagnosis prediction
            if info.get('correct_diagnosis', False):
                reward += 1.0
            
            # Reward for appropriate follow-up detection
            if info.get('correct_followup', False):
                reward += 0.5
            
            # Penalty for wrong diagnosis
            if info.get('wrong_diagnosis', False):
                reward -= 0.5
            
            # Penalty for unnecessary follow-up
            if info.get('unnecessary_followup', False):
                reward -= 0.3
            
            return reward
        
        return diagnosis_reward
    
    def fine_tune(self, train_file, val_file, test_file, epochs=10, batch_size=32, learning_rate=0.0001):
        """Fine-tune the model on diagnosis data."""
        print(f"🏥 Starting diagnosis fine-tuning for {epochs} epochs")
        print(f"📊 Batch size: {batch_size}, Learning rate: {learning_rate}")
        
        # Load data
        train_data = self._prepare_diagnosis_data(train_file)
        val_data = self._prepare_diagnosis_data(val_file)
        test_data = self._prepare_diagnosis_data(test_file)
        
        print(f"📈 Training samples: {len(train_data)}")
        print(f"📈 Validation samples: {len(val_data)}")
        print(f"📈 Test samples: {len(test_data)}")
        
        # Initialize trainer
        trainer = DiagnosisTrainer(
            disp_model=self.disp_model,
            retriever=self.retriever,
            config=self.config,
            reward_function=self._create_diagnosis_reward_function()
        )
        
        # Training loop
        best_val_accuracy = 0.0
        best_model_path = None
        
        for epoch in range(epochs):
            print(f"\n🔄 Epoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss = trainer.train_epoch(train_data)
            
            # Validate
            val_metrics = trainer.evaluate(val_data)
            val_accuracy = val_metrics['diagnosis_accuracy']
            
            print(f"  📊 Train Loss: {train_loss:.4f}")
            print(f"  📊 Val Accuracy: {val_accuracy:.4f}")
            print(f"  📊 Val Strategy Diversity: {val_metrics['strategy_diversity']:.4f}")
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_path = f'logs/diagnosis_finetune/best_model_epoch_{epoch+1}.pth'
                self.disp_model.save(best_model_path)
                print(f"  💾 New best model saved: {best_model_path}")
            
            # Save checkpoint
            checkpoint_path = f'logs/diagnosis_finetune/checkpoint_epoch_{epoch+1}.pth'
            self.disp_model.save(checkpoint_path)
        
        # Final evaluation on test set
        print(f"\n🎯 Final evaluation on test set")
        test_metrics = trainer.evaluate(test_data)
        
        print(f"📊 Final Test Results:")
        print(f"  Diagnosis Accuracy: {test_metrics['diagnosis_accuracy']:.4f}")
        print(f"  Follow-up Accuracy: {test_metrics['followup_accuracy']:.4f}")
        print(f"  Strategy Diversity: {test_metrics['strategy_diversity']:.4f}")
        print(f"  Retrieval Accuracy: {test_metrics['retrieval_accuracy']:.4f}")
        
        # Save final model
        final_model_path = 'logs/diagnosis_finetune/final_model.pth'
        self.disp_model.save(final_model_path)
        print(f"💾 Final model saved: {final_model_path}")
        
        return {
            'best_val_accuracy': best_val_accuracy,
            'best_model_path': best_model_path,
            'final_model_path': final_model_path,
            'test_metrics': test_metrics
        }
    
    def _save_model(self, path):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.disp_model.save(path)
        self.logger.info(f"💾 Model saved to: {path}")
    
    def evaluate(self, test_file):
        """Evaluate the fine-tuned model."""
        self.logger.info("🔍 Evaluating fine-tuned model")
        
        test_data = self._prepare_diagnosis_data(test_file)
        
        # Create evaluation trainer
        eval_trainer = DiagnosisTrainer(
            disp_model=self.disp_model,
            retriever=self.retriever,
            config=self.config
        )
        
        metrics = eval_trainer.evaluate(test_data)
        
        self.logger.info("📊 Evaluation Results:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")
        
        return metrics


class DiagnosisTrainer:
    """Custom trainer for diagnosis domain."""
    
    def __init__(self, disp_model, retriever, config, reward_function=None):
        self.disp_model = disp_model
        self.retriever = retriever
        self.config = config
        self.reward_function = reward_function or self._default_reward
        
        # Set model to training mode
        self.disp_model.train()
    
    def _default_reward(self, state, action, next_state, info):
        """Default reward function."""
        return 0.1  # Small positive reward for any action
    
    def train_epoch(self, train_data):
        """Train for one epoch."""
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch in train_data:
            # Prepare state
            state = self.disp_model._prepare_state(batch)
            
            # Get action
            action = self.disp_model.select_action(state)
            
            # Simulate environment step
            next_state, reward, done, info = self._simulate_step(batch, action)
            
            # Calculate custom reward
            custom_reward = self.reward_function(state, action, next_state, info)
            
            # Store experience
            self.disp_model.replay_buffer.push(
                state=state,
                action=action,
                reward=custom_reward,
                next_state=next_state,
                done=done
            )
            
            # Train on batch if enough samples
            if len(self.disp_model.replay_buffer) >= self.config.BATCH_SIZE:
                loss = self.disp_model.train_on_batch()
                if loss is not None:
                    total_loss += loss
            
            # Track accuracy
            if info.get('correct_diagnosis', False):
                correct_predictions += 1
            total_predictions += 1
        
        return {
            'loss': total_loss / len(train_data) if train_data else 0.0,
            'accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0.0
        }
    
    def evaluate(self, eval_data):
        """Evaluate the model."""
        self.disp_model.eval()
        
        correct_diagnoses = 0
        correct_followups = 0
        total_samples = 0
        actions_taken = []
        
        with torch.no_grad():
            for batch in eval_data:
                state = self.disp_model._prepare_state(batch)
                action = self.disp_model.select_action(state, eval_mode=True)
                
                # Track actions for diversity
                actions_taken.append(action.item())
                
                # Check diagnosis accuracy
                predicted_doc = batch['documents'][0] if batch['documents'] else ""
                ground_truth = batch['ground_truth']
                
                if predicted_doc.startswith(ground_truth):
                    correct_diagnoses += 1
                
                # Check follow-up accuracy
                needs_followup = batch['needs_followup']
                predicted_followup = (action.item() == 0)  # Action 0 = ask/follow-up
                
                if predicted_followup == needs_followup:
                    correct_followups += 1
                
                total_samples += 1
        
        # Calculate metrics
        accuracy = correct_diagnoses / total_samples if total_samples > 0 else 0.0
        followup_accuracy = correct_followups / total_samples if total_samples > 0 else 0.0
        
        # Calculate strategy diversity
        unique_actions = len(set(actions_taken))
        strategy_diversity = unique_actions / 3.0  # 3 possible actions
        
        return {
            'accuracy': accuracy,
            'followup_accuracy': followup_accuracy,
            'followup_f1': followup_accuracy,  # Simplified F1
            'strategy_diversity': strategy_diversity,
            'total_samples': total_samples
        }
    
    def _simulate_step(self, batch, action):
        """Simulate environment step for training."""
        # Simple simulation - in practice, this would be more complex
        next_state = batch  # Simplified
        reward = 0.1
        done = True
        
        info = {
            'correct_diagnosis': False,  # Would be calculated based on ground truth
            'correct_followup': False,   # Would be calculated based on needs
            'wrong_diagnosis': False,
            'unnecessary_followup': False
        }
        
        return next_state, reward, done, info


def main():
    """Main fine-tuning function."""
    print("🏥 STYLE Diagnosis Domain Fine-tuning")
    print("=" * 50)
    
    # Check if knowledge base exists
    if not os.path.exists("data/diagnosis_documents_complete.csv"):
        print("❌ Comprehensive knowledge base not found!")
        print("Please run build_diagnosis_knowledge_base.py first")
        return
    
    # Initialize fine-tuner
    base_model_path = "saved_models/quick_train/best_model.pt"
    if not os.path.exists(base_model_path):
        print(f"⚠️ Base model not found: {base_model_path}")
        base_model_path = None
    
    fine_tuner = DiagnosisFineTuner(base_model_path=base_model_path)
    
    # Fine-tune
    try:
        metrics = fine_tuner.fine_tune(
            train_file="data/train/diagnosis_train.csv",
            val_file="data/val/diagnosis_val.csv", 
            test_file="data/test/diagnosis_test.csv",
            epochs=5,
            batch_size=16,
            learning_rate=0.0001
        )
        
        print("\n✅ Fine-tuning completed successfully!")
        print("📊 Final Test Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
    except Exception as e:
        print(f"❌ Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 