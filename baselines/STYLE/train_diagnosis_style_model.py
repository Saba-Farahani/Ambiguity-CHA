#!/usr/bin/env python3
"""
Train STYLE model from scratch on diagnosis dataset with follow-up question capability.

This script trains a STYLE model specifically for medical diagnosis scenarios where
the model can ask follow-up questions when symptoms are removed and more information
is needed for accurate diagnosis.
"""

import os
import sys
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings("ignore")

# Add the style package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'style'))

from style.models.disp import DISP
from style.models.retriever import Retriever
from style.utils.llm_integration import LLMIntegration
from style.config import Config
from style.utils.replay_memory import ReplayMemory


class DiagnosisDataset(Dataset):
    """Dataset class for diagnosis training data."""
    
    def __init__(self, data_file: str, tokenizer=None):
        """
        Initialize diagnosis dataset.
        
        Args:
            data_file: Path to diagnosis_train.csv
            tokenizer: BERT tokenizer for text processing
        """
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.samples = self._load_and_process_data()
        
    def _load_and_process_data(self) -> List[Dict]:
        """Load and process the diagnosis training data."""
        print(f"📊 Loading diagnosis data from {self.data_file}")
        
        df = pd.read_csv(self.data_file)
        samples = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing diagnosis data"):
            # Extract information from the row
            patient_id = row['PATIENT']
            pathology = row['PATHOLOGY']
            full_symptoms = eval(row['FULL_SYMPTOMS']) if isinstance(row['FULL_SYMPTOMS'], str) else row['FULL_SYMPTOMS']
            remaining_symptoms = eval(row['REMAINING_SYMPTOMS']) if isinstance(row['REMAINING_SYMPTOMS'], str) else row['REMAINING_SYMPTOMS']
            removed_symptoms = eval(row['REMOVED_SYMPTOMS']) if isinstance(row['REMOVED_SYMPTOMS'], str) else row['REMOVED_SYMPTOMS']
            conversation = row['Conversation']
            
            # Create training sample
            sample = {
                'patient_id': patient_id,
                'pathology': pathology,
                'full_symptoms': full_symptoms,
                'remaining_symptoms': remaining_symptoms,
                'removed_symptoms': removed_symptoms,
                'conversation': conversation,
                'needs_followup': len(removed_symptoms) > 0,  # Needs followup if symptoms were removed
                'symptom_removal_ratio': len(removed_symptoms) / len(full_symptoms) if len(full_symptoms) > 0 else 0
            }
            samples.append(sample)
        
        print(f"✅ Loaded {len(samples)} diagnosis samples")
        print(f"📈 {sum(s['needs_followup'] for s in samples)} samples need follow-up questions")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


class DiagnosisStyleTrainer:
    """Trainer for STYLE model on diagnosis dataset."""
    
    def __init__(self, config: Config, data_file: str, output_dir: str = "diagnosis_training_output"):
        """
        Initialize the diagnosis trainer.
        
        Args:
            config: Configuration object
            data_file: Path to diagnosis_train.csv
            output_dir: Directory to save outputs
        """
        self.config = config
        self.data_file = data_file
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Using device: {self.device}")
        
        # Initialize dataset
        self.dataset = DiagnosisDataset(data_file)
        self.dataloader = DataLoader(self.dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        
        # Initialize models
        self.disp_model = DISP(config).to(self.device)
        self.retriever = Retriever(config)
        self.llm = LLMIntegration(config)
        
        # Initialize knowledge base for diagnosis
        self._setup_diagnosis_knowledge_base()
        
        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        self.training_history = {
            'epoch': [],
            'loss': [],
            'accuracy': [],
            'followup_accuracy': [],
            'timestamp': []
        }
        
        # Initialize replay buffer
        self.replay_buffer = ReplayMemory(config.MEMORY_SIZE)
        
        print("✅ Diagnosis STYLE trainer initialized")
    
    def _setup_diagnosis_knowledge_base(self):
        """Setup knowledge base for diagnosis domain."""
        print("🏥 Setting up diagnosis knowledge base...")
        
        # Get unique pathologies from the dataset
        pathologies = list(set(sample['pathology'] for sample in self.dataset.samples))
        
        # Create knowledge base entries
        self.knowledge_base = {}
        for pathology in pathologies:
            # Get all symptoms associated with this pathology
            all_symptoms = []
            for sample in self.dataset.samples:
                if sample['pathology'] == pathology:
                    all_symptoms.extend(sample['full_symptoms'])
            
            # Create knowledge entry
            symptoms_text = ", ".join(set(all_symptoms))
            self.knowledge_base[pathology] = {
                'description': f"Medical condition: {pathology}",
                'symptoms': symptoms_text,
                'full_text': f"Diagnosis: {pathology}. Common symptoms include: {symptoms_text}"
            }
        
        print(f"📚 Knowledge base contains {len(self.knowledge_base)} diagnoses")
        
        # Save knowledge base
        kb_path = os.path.join(self.output_dir, "diagnosis_knowledge_base.json")
        with open(kb_path, 'w') as f:
            json.dump(self.knowledge_base, f, indent=2)
        print(f"💾 Knowledge base saved to {kb_path}")
    
    def _prepare_state(self, batch: Dict) -> torch.Tensor:
        """Prepare state tensor for the model."""
        # For now, create a simple state representation
        # In a full implementation, this would include BERT embeddings
        
        batch_size = len(batch['pathology'])
        state_dim = 768 + 768 + 1  # query_embedding + doc_embedding + score
        
        # Create placeholder state (replace with actual BERT embeddings)
        state = torch.randn(batch_size, state_dim).to(self.device)
        return state
    
    def _get_reward(self, action: int, batch: Dict, retrieved_docs: List[str]) -> float:
        """
        Calculate reward based on action and retrieved documents.
        
        Args:
            action: Action taken (0: ask follow-up, 1: provide diagnosis)
            batch: Current batch data
            retrieved_docs: Retrieved diagnostic documents
            
        Returns:
            Reward value
        """
        ground_truth = batch['pathology'][0]  # Assuming single sample in batch
        needs_followup = batch['needs_followup'][0]
        
        # Base rewards
        if action == 0:  # Ask follow-up
            if needs_followup:
                return self.config.REWARD_CLARIFY  # Positive reward for asking when needed
            else:
                return self.config.REWARD_INVALID  # Negative reward for asking when not needed
        else:  # Provide diagnosis
            # Check if correct diagnosis is in retrieved documents
            correct_retrieval = any(ground_truth.lower() in doc.lower() for doc in retrieved_docs)
            
            if correct_retrieval:
                return self.config.REWARD_SUCCESS
            else:
                return self.config.FAILURE_REWARD
    
    def _retrieve_documents(self, query: str, top_k: int = 5) -> Tuple[List[str], List[float]]:
        """Retrieve relevant diagnostic documents."""
        try:
            docs, scores = self.retriever.retrieve(query, domain='diagnosis', top_k=top_k)
            return docs, scores
        except:
            # Fallback to simple keyword matching
            query_lower = query.lower()
            matching_docs = []
            matching_scores = []
            
            for pathology, kb_entry in self.knowledge_base.items():
                if any(symptom.lower() in query_lower for symptom in kb_entry['symptoms'].split(', ')):
                    matching_docs.append(kb_entry['full_text'])
                    matching_scores.append(0.8)  # Placeholder score
            
            return matching_docs[:top_k], matching_scores[:top_k]
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.disp_model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_followup_accuracy = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.dataloader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Prepare state
                state = self._prepare_state(batch)
                
                # Select action
                action = self.disp_model.select_action(state, eval_mode=False)
                
                # Retrieve documents based on conversation
                conversation = batch['conversation'][0]  # Assuming single sample
                retrieved_docs, scores = self._retrieve_documents(conversation)
                
                # Calculate reward
                reward = self._get_reward(action.item(), batch, retrieved_docs)
                
                # Prepare next state (simplified)
                next_state = torch.randn_like(state)
                done = torch.tensor([True], dtype=torch.bool)  # Simplified
                
                # Store in replay buffer
                self.replay_buffer.push(state, action, next_state, torch.tensor([reward]), done)
                
                # Train if we have enough samples
                if len(self.replay_buffer) >= self.config.BATCH_SIZE:
                    loss = self.disp_model.train_on_batch()
                    epoch_loss += loss
                
                # Calculate accuracy metrics
                ground_truth = batch['pathology'][0]
                needs_followup = batch['needs_followup'][0]
                
                # Check if action was correct
                if action.item() == 0 and needs_followup:  # Correctly asked follow-up
                    epoch_followup_accuracy += 1
                elif action.item() == 1 and not needs_followup:  # Correctly provided diagnosis
                    if any(ground_truth.lower() in doc.lower() for doc in retrieved_docs):
                        epoch_accuracy += 1
                
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{epoch_loss / num_batches:.4f}',
                    'Acc': f'{epoch_accuracy / num_batches:.4f}',
                    'Followup': f'{epoch_followup_accuracy / num_batches:.4f}'
                })
                
            except Exception as e:
                print(f"⚠️ Error in batch {batch_idx}: {e}")
                continue
        
        # Calculate epoch metrics
        epoch_metrics = {
            'loss': epoch_loss / num_batches if num_batches > 0 else 0,
            'accuracy': epoch_accuracy / num_batches if num_batches > 0 else 0,
            'followup_accuracy': epoch_followup_accuracy / num_batches if num_batches > 0 else 0
        }
        
        return epoch_metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.disp_model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        val_followup_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.dataloader:
                try:
                    state = self._prepare_state(batch)
                    action = self.disp_model.select_action(state, eval_mode=True)
                    
                    conversation = batch['conversation'][0]
                    retrieved_docs, _ = self._retrieve_documents(conversation)
                    
                    ground_truth = batch['pathology'][0]
                    needs_followup = batch['needs_followup'][0]
                    
                    # Calculate metrics
                    if action.item() == 0 and needs_followup:
                        val_followup_accuracy += 1
                    elif action.item() == 1 and not needs_followup:
                        if any(ground_truth.lower() in doc.lower() for doc in retrieved_docs):
                            val_accuracy += 1
                    
                    num_batches += 1
                    
                except Exception as e:
                    print(f"⚠️ Validation error: {e}")
                    continue
        
        return {
            'val_loss': val_loss / num_batches if num_batches > 0 else 0,
            'val_accuracy': val_accuracy / num_batches if num_batches > 0 else 0,
            'val_followup_accuracy': val_followup_accuracy / num_batches if num_batches > 0 else 0
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.disp_model.state_dict(),
            'optimizer_state_dict': self.disp_model.optimizer.state_dict(),
            'metrics': metrics,
            'config': vars(self.config),
            'training_history': self.training_history
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.output_dir, "checkpoints", f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.output_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"🏆 New best model saved with loss: {metrics['loss']:.4f}")
        
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def save_training_history(self):
        """Save training history."""
        history_path = os.path.join(self.output_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"📊 Training history saved: {history_path}")
    
    def train(self, num_epochs: int = 100, save_every: int = 10):
        """
        Train the model.
        
        Args:
            num_epochs: Number of training epochs
            save_every: Save checkpoint every N epochs
        """
        print(f"🚀 Starting training for {num_epochs} epochs...")
        print(f"📊 Dataset size: {len(self.dataset)} samples")
        print(f"💾 Checkpoints will be saved every {save_every} epochs")
        
        start_time = datetime.now()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train for one epoch
            epoch_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Combine metrics
            all_metrics = {**epoch_metrics, **val_metrics}
            
            # Update training history
            self.training_history['epoch'].append(epoch)
            self.training_history['loss'].append(epoch_metrics['loss'])
            self.training_history['accuracy'].append(epoch_metrics['accuracy'])
            self.training_history['followup_accuracy'].append(epoch_metrics['followup_accuracy'])
            self.training_history['timestamp'].append(datetime.now().isoformat())
            
            # Print epoch results
            print(f"\n📈 Epoch {epoch}/{num_epochs}")
            print(f"   Train Loss: {epoch_metrics['loss']:.4f}")
            print(f"   Train Accuracy: {epoch_metrics['accuracy']:.4f}")
            print(f"   Train Followup Accuracy: {epoch_metrics['followup_accuracy']:.4f}")
            print(f"   Val Accuracy: {val_metrics['val_accuracy']:.4f}")
            print(f"   Val Followup Accuracy: {val_metrics['val_followup_accuracy']:.4f}")
            
            # Save checkpoint if needed
            is_best = epoch_metrics['loss'] < self.best_loss
            if is_best:
                self.best_loss = epoch_metrics['loss']
            
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, all_metrics, is_best)
            
            # Save training history
            self.save_training_history()
        
        # Final save
        self.save_checkpoint(epoch, all_metrics, False)
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        print(f"\n✅ Training completed!")
        print(f"⏱️ Total training time: {training_time}")
        print(f"🏆 Best loss: {self.best_loss:.4f}")
        print(f"💾 All outputs saved to: {self.output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train STYLE model on diagnosis dataset")
    parser.add_argument("--data-file", type=str, 
                       default="/home/ailaty3088@id.sdsu.edu/STYLE/data/train/diagnosis_train.csv",
                       help="Path to diagnosis_train.csv")
    parser.add_argument("--output-dir", type=str, default="diagnosis_training_output",
                       help="Output directory for training results")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--save-every", type=int, default=10,
                       help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config()
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate
    config.NUM_EPOCHS = args.epochs
    config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize trainer
    trainer = DiagnosisStyleTrainer(config, args.data_file, args.output_dir)
    
    # Start training
    trainer.train(num_epochs=args.epochs, save_every=args.save_every)


if __name__ == "__main__":
    main()
