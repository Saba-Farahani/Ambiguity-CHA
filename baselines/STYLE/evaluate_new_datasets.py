#!/usr/bin/env python3
"""
Evaluate STYLE Models on New Datasets
======================================

This script evaluates all STYLE model variants on the new mental health and food datasets.
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime
import argparse

# Import STYLE components
from style.config import Config
from style.training.mdt import MDTTrainer
from style.models.disp import DISP


class NewDatasetEvaluator:
    """Evaluator for new datasets (mental health and food)."""
    
    def __init__(self, force_cpu=False):
        """Initialize evaluator."""
        self.config = Config()
        # Force CPU if CUDA is not compatible or if requested
        if force_cpu or not torch.cuda.is_available():
            self.device = torch.device("cpu")
            self.config.DEVICE = torch.device("cpu")
            print("⚠️  Using CPU device (CUDA not available or forced)")
        else:
            self.device = self.config.DEVICE
        
        # Model paths for all variants (only include models with valid paths)
        self.model_paths = {
            'finetuned_style': 'saved_models/quick_train/best_model.pt',
            'custom_style': 'checkpoints/quick_train/checkpoint_step_55333.pt',
            'scratch_style': 'diagnosis_training_20251001_104635/best_model.pth'
        }
        
        print(f"🔧 Initialized evaluator")
        print(f"   Device: {self.device}")
        print(f"   Available models: {list(self.model_paths.keys())}")
    
    def load_new_dataset(self, json_file: str) -> List[Dict[str, Any]]:
        """Load new dataset from JSON file."""
        print(f"📊 Loading dataset from: {json_file}")
        
        if not os.path.exists(json_file):
            print(f"❌ File not found: {json_file}")
            return []
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to evaluation format
        test_samples = []
        for item in data:
            original_query = item.get('original_query', '')
            ground_truth = item.get('ground_truth', '')
            ambiguous_queries = item.get('ambiguous_queries', [])
            
            # Process original query
            test_samples.append({
                'query': original_query,
                'ground_truth': ground_truth,
                'query_type': 'original',
                'masking_strategy': None
            })
            
            # Process ambiguous queries
            for amb_query in ambiguous_queries:
                query_text = amb_query.get('query', '').strip('"')
                masking_strategy = amb_query.get('masking_strategy', 'unknown')
                
                test_samples.append({
                    'query': query_text,
                    'ground_truth': ground_truth,
                    'query_type': 'ambiguous',
                    'masking_strategy': masking_strategy
                })
        
        print(f"✅ Loaded {len(test_samples)} test samples ({len(data)} original entries)")
        return test_samples
    
    def load_model(self, model_path: str, model_key: str) -> Tuple[Any, str]:
        """Load model with fallback strategies."""
        if not model_path or not os.path.exists(model_path):
            print(f"⚠️  Model not found: {model_path}")
            return None, "model_not_found"
        
        try:
            print(f"🔧 Loading {model_key} from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Initialize trainer
            trainer = MDTTrainer(self.config)
            
            # Try different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
            else:
                state_dict = checkpoint
            
            # Load with strict=False to handle missing keys
            trainer.disp_model.load_state_dict(state_dict, strict=False)
            trainer.disp_model.to(self.device)
            trainer.disp_model.eval()
            
            print(f"   ✅ Successfully loaded {model_key}")
            return trainer, 'success'
            
        except Exception as e:
            print(f"❌ Error loading {model_key}: {e}")
            import traceback
            traceback.print_exc()
            return None, str(e)
    
    def create_state_from_query(self, query: str) -> torch.Tensor:
        """Create paper-aligned state tensor from a single query."""
        try:
            if not query or not query.strip() or len(query.strip()) < 2:
                query = "no_query"
            else:
                query = query.strip()

            temp_disp = DISP(self.config).to(self.device)
            temp_disp.eval()

            with torch.no_grad():
                docs = [""] * self.config.TOP_K_DOCS
                scores = [0.0] * self.config.TOP_K_DOCS
                state = temp_disp.construct_features([query], docs, scores)

            del temp_disp
            return state

        except Exception as e:
            print(f"⚠️  Error creating state: {e}")
            dim = Config.state_input_dim()
            return torch.zeros((1, dim), device=self.device)
    
    def predict_with_model(self, trainer, query: str) -> Dict[str, Any]:
        """Make prediction using the model."""
        try:
            # Create state from query
            state = self.create_state_from_query(query)
            
            # Get model prediction
            with torch.no_grad():
                try:
                    action_probs = trainer.disp_model(state)
                    action = torch.argmax(action_probs, dim=-1).item()

                    # 0 = answer, 1 = ask (paper action space)
                    if action == Config.ACTION_ANSWER:
                        return {
                            'action': 'answer',
                            'response': 'MODEL_ANSWER',
                            'action_prob': float(action_probs[0, action].item())
                        }
                    else:
                        return {
                            'action': 'ask_clarification',
                            'response': 'ASK_FOR_CLARIFICATION',
                            'action_prob': float(action_probs[0, action].item())
                        }
                
                except Exception as e:
                    print(f"⚠️  Error in model forward pass: {e}")
                    return {
                        'action': 'error',
                        'response': 'ERROR',
                        'action_prob': 0.0
                    }
        
        except Exception as e:
            print(f"⚠️  Error predicting: {e}")
            return {
                'action': 'error',
                'response': 'ERROR',
                'action_prob': 0.0
            }
    
    def evaluate_model(self, model_key: str, test_samples: List[Dict[str, Any]], 
                      max_samples: int = None) -> Dict[str, Any]:
        """Evaluate a model on test samples."""
        print(f"\n🔧 Evaluating {model_key}...")
        
        model_path = self.model_paths.get(model_key)
        if not model_path:
            print(f"⚠️  No model path for {model_key} - skipping evaluation")
            return self._create_fallback_results(model_key)
        
        # Load model
        trainer, load_status = self.load_model(model_path, model_key)
        if trainer is None:
            print(f"❌ Failed to load {model_key}: {load_status}")
            return self._create_fallback_results(model_key)
        
        # Limit samples if specified
        if max_samples:
            test_samples = test_samples[:max_samples]
        
        print(f"📊 Testing on {len(test_samples)} samples...")
        
        # Initialize metrics
        results = {
            'predictions': [],
            'ground_truths': [],
            'actions': [],
            'action_probs': [],
            'query_types': [],
            'masking_strategies': []
        }
        
        # Evaluate each sample
        for i, sample in enumerate(test_samples):
            if (i + 1) % 100 == 0:
                print(f"   Processing sample {i+1}/{len(test_samples)}")
            
            try:
                query = sample['query']
                ground_truth = sample['ground_truth']
                
                # Get prediction
                prediction = self.predict_with_model(trainer, query)
                
                # Store results
                results['predictions'].append(prediction['response'])
                results['ground_truths'].append(ground_truth)
                results['actions'].append(prediction['action'])
                results['action_probs'].append(prediction['action_prob'])
                results['query_types'].append(sample['query_type'])
                results['masking_strategies'].append(sample.get('masking_strategy', 'N/A'))
            
            except Exception as e:
                print(f"⚠️  Error processing sample {i}: {e}")
                results['predictions'].append('ERROR')
                results['ground_truths'].append(sample.get('ground_truth', ''))
                results['actions'].append('error')
                results['action_probs'].append(0.0)
                results['query_types'].append(sample.get('query_type', 'unknown'))
                results['masking_strategies'].append(sample.get('masking_strategy', 'N/A'))
        
        # Calculate metrics
        metrics = self._calculate_metrics(results)
        
        print(f"✅ Evaluation results for {model_key}:")
        print(f"   Action Distribution: Ask={metrics['action_distribution']['ask']:.2%}, Answer={metrics['action_distribution']['answer']:.2%}")
        print(f"   Average Action Probability: {metrics['avg_action_prob']:.4f}")
        
        return {
            'model_key': model_key,
            'load_status': load_status,
            'metrics': metrics,
            'results': results,
            'total_samples': len(test_samples)
        }
    
    def _calculate_metrics(self, results: Dict[str, List]) -> Dict[str, Any]:
        """Calculate evaluation metrics."""
        total = len(results['actions'])
        if total == 0:
            return {}
        
        # Action distribution
        actions = results['actions']
        ask_count = sum(1 for a in actions if a == 'ask_clarification')
        answer_count = sum(1 for a in actions if a == 'answer')
        error_count = sum(1 for a in actions if a == 'error')
        
        # Action probabilities
        action_probs = [p for p in results['action_probs'] if p > 0]
        avg_action_prob = np.mean(action_probs) if action_probs else 0.0
        
        # Query type distribution
        query_types = results['query_types']
        original_count = sum(1 for qt in query_types if qt == 'original')
        ambiguous_count = sum(1 for qt in query_types if qt == 'ambiguous')
        
        # Masking strategy distribution (for ambiguous queries)
        masking_strategies = [ms for ms in results['masking_strategies'] if ms != 'N/A' and ms is not None]
        strategy_counts = {}
        for ms in masking_strategies:
            strategy_counts[ms] = strategy_counts.get(ms, 0) + 1
        
        return {
            'action_distribution': {
                'ask': ask_count / total if total > 0 else 0.0,
                'answer': answer_count / total if total > 0 else 0.0,
                'error': error_count / total if total > 0 else 0.0
            },
            'action_counts': {
                'ask': ask_count,
                'answer': answer_count,
                'error': error_count,
                'total': total
            },
            'avg_action_prob': avg_action_prob,
            'query_type_distribution': {
                'original': original_count / total if total > 0 else 0.0,
                'ambiguous': ambiguous_count / total if total > 0 else 0.0
            },
            'query_type_counts': {
                'original': original_count,
                'ambiguous': ambiguous_count,
                'total': total
            },
            'masking_strategy_distribution': strategy_counts
        }
    
    def _create_fallback_results(self, model_key: str) -> Dict[str, Any]:
        """Create fallback results when model fails or is not available.
        
        Returns zero metrics since no evaluation was performed.
        """
        print(f"⚠️  Model {model_key} not available - returning zero metrics")
        return {
            'model_key': model_key,
            'load_status': 'not_available',
            'metrics': {
                'action_distribution': {
                    'ask': 0.0,
                    'answer': 0.0,
                    'error': 0.0
                },
                'avg_action_prob': 0.0
            },
            'results': {
                'predictions': [],
                'ground_truths': [],
                'actions': [],
                'action_probs': [],
                'query_types': [],
                'masking_strategies': []
            },
            'total_samples': 0
        }
    
    def run_evaluation(self, dataset_path: str, dataset_name: str, 
                      max_samples: int = None) -> Dict[str, Any]:
        """Run evaluation on a dataset."""
        print(f"\n🚀 Running evaluation on {dataset_name}")
        print(f"   Dataset path: {dataset_path}")
        
        # Load dataset
        test_samples = self.load_new_dataset(dataset_path)
        if not test_samples:
            print("❌ No test samples loaded")
            return {}
        
        # Evaluate all models (only those with valid paths)
        all_results = {}
        for model_key, model_path in self.model_paths.items():
            if model_path is None:
                print(f"\n{'='*60}")
                print(f"⚠️  Skipping {model_key} - no model path provided")
                continue
            print(f"\n{'='*60}")
            results = self.evaluate_model(model_key, test_samples, max_samples)
            all_results[model_key] = results
        
        return {
            'dataset_name': dataset_name,
            'dataset_path': dataset_path,
            'total_samples': len(test_samples),
            'model_results': all_results,
            'timestamp': datetime.now().isoformat()
        }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Evaluate STYLE models on new datasets')
    parser.add_argument('--mental-health', type=str, 
                       default='Datasets - Mental and Food/aim_output_mental_health_multi_strategy_20251210_143131.json',
                       help='Path to mental health dataset JSON file')
    parser.add_argument('--food', type=str,
                       default='Datasets - Mental and Food/aim_output_food_multi_strategy_20251210_151422.json',
                       help='Path to food dataset JSON file')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to evaluate per dataset')
    parser.add_argument('--output-dir', type=str, default='new_datasets_evaluation',
                       help='Output directory for results')
    parser.add_argument('--force-cpu', action='store_true',
                       help='Force CPU usage (useful for CUDA compatibility issues)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = NewDatasetEvaluator(force_cpu=args.force_cpu)
    
    # Evaluate mental health dataset
    print("\n" + "="*60)
    print("MENTAL HEALTH DATASET EVALUATION")
    print("="*60)
    mental_health_results = evaluator.run_evaluation(
        args.mental_health,
        'mental_health',
        args.max_samples
    )
    
    # Save mental health results
    if mental_health_results:
        output_file = os.path.join(args.output_dir, 
                                   f'mental_health_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(mental_health_results, f, indent=2, default=str)
        print(f"\n💾 Mental health results saved to: {output_file}")
    
    # Evaluate food dataset
    print("\n" + "="*60)
    print("FOOD DATASET EVALUATION")
    print("="*60)
    food_results = evaluator.run_evaluation(
        args.food,
        'food',
        args.max_samples
    )
    
    # Save food results
    if food_results:
        output_file = os.path.join(args.output_dir,
                                   f'food_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(food_results, f, indent=2, default=str)
        print(f"\n💾 Food results saved to: {output_file}")
    
    # Create summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'datasets_evaluated': ['mental_health', 'food'],
        'models_evaluated': list(evaluator.model_paths.keys()),
        'mental_health_summary': {
            'total_samples': mental_health_results.get('total_samples', 0) if mental_health_results else 0,
            'models': {k: v.get('load_status', 'unknown') for k, v in mental_health_results.get('model_results', {}).items()} if mental_health_results else {}
        },
        'food_summary': {
            'total_samples': food_results.get('total_samples', 0) if food_results else 0,
            'models': {k: v.get('load_status', 'unknown') for k, v in food_results.get('model_results', {}).items()} if food_results else {}
        }
    }
    
    summary_file = os.path.join(args.output_dir, 
                                f'evaluation_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n💾 Summary saved to: {summary_file}")
    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()

