"""
Dataset management for STYLE implementation.
Handles dataset splits, domain organization, and data loading.
"""

import os
import json
import random
import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from ..config import Config
from torch.utils.data import Dataset, DataLoader
import traceback
import pickle

def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function to handle variable length sequences."""
    if not batch:
        raise ValueError("Empty batch received")
    
    # Get max lengths for padding
    max_query_len = max(len(item['query_history']) for item in batch)
    max_doc_len = max(len(item['documents']) for item in batch)
    
    # Initialize batch dictionary
    batch_dict = {
        'query_history': [],
        'documents': [],
        'retrieval_scores': [],
        'target_action': [],
        'success': [],
        'num_turns': [],
        'target_doc': []
    }
    
    # Process each item
    for item in batch:
        # Pad query history
        query_history = item['query_history'].copy()
        if len(query_history) < max_query_len:
            query_history.extend([''] * (max_query_len - len(query_history)))
        
        # Pad documents
        documents = item['documents'].copy()
        if len(documents) < max_doc_len:
            documents.extend([''] * (max_doc_len - len(documents)))
        
        # Pad retrieval scores
        scores = item['retrieval_scores'].copy()
        if len(scores) < max_doc_len:
            scores.extend([0.0] * (max_doc_len - len(scores)))
        
        # Add to batch
        batch_dict['query_history'].append(query_history)
        batch_dict['documents'].append(documents)
        batch_dict['retrieval_scores'].append(scores)
        batch_dict['target_action'].append(item['target_action'])
        batch_dict['success'].append(item['success'])
        batch_dict['num_turns'].append(item['num_turns'])
        batch_dict['target_doc'].append(item['target_doc'])
    
    # Convert to tensors where appropriate
    for key in batch_dict:
        if key in ['query_history', 'documents', 'target_doc']:
            # These are text fields, keep as lists
            continue
        elif key == 'retrieval_scores':
            batch_dict[key] = torch.tensor(batch_dict[key], dtype=torch.float)
        elif key in ['success']:
            batch_dict[key] = torch.tensor(batch_dict[key], dtype=torch.bool)
        else:
            batch_dict[key] = torch.tensor(batch_dict[key], dtype=torch.long)
    
    return batch_dict

class ConversationDataset(Dataset):
    """Dataset for conversation data."""
    
    def __init__(self, data: List[Dict[str, Any]]):
        """Initialize dataset.
        
        Args:
            data: List of conversation examples
        """
        self.data = []
        
        # Process and validate each item
        for item in data:
            # Clean and validate query history
            query_history = [q for q in item.get('query_history', []) if q and not pd.isna(q)]
            if not query_history:  # Skip if no valid queries
                continue
                
            # Clean and validate documents
            documents = [d for d in item.get('documents', []) if d and not pd.isna(d)]
            if not documents:  # Skip if no valid documents
                continue
                
            # Clean and validate retrieval scores
            scores = item.get('retrieval_scores', [])
            if len(scores) != len(documents):
                scores = [1.0] * len(documents)  # Default score if mismatch
            
            # Create processed item with fixed structure
            processed_item = {
                'query_history': query_history,
                'documents': documents,
                'retrieval_scores': scores,
                'target_action': int(item.get('target_action', 0)),  # Ensure integer
                'success': bool(item.get('success', False)),  # Ensure boolean
                'num_turns': int(item.get('num_turns', 0)),  # Ensure integer
                'target_doc': str(documents[0] if documents else '')  # Use first document as target
            }
            
            self.data.append(processed_item)
        
        print(f"Processed {len(self.data)} valid examples")
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item at index."""
        return self.data[idx]

class DatasetManager:
    def __init__(self):
        """Initialize the dataset manager."""
        self.datasets = {}
        self.domain_stats = {}
        self._load_datasets()
    
    def _load_datasets(self):
        """Load and preprocess datasets for each domain."""
        # Supported domains from paper
        domains = ['clariq', 'opendialkg']  # Removed 'faqant' and 'msdialog'
        
        for domain in domains:
            try:
                # Load raw data for the domain
                raw_data = self._load_domain_data(domain)
                
                # Convert to paper's format
                formatted_data = self._convert_to_paper_format(raw_data, domain)
                
                # Balance ambiguity
                balanced_data = self._balance_ambiguity(formatted_data)
                
                # Create dataset
                dataset = ConversationDataset(balanced_data)
                
                # Calculate split sizes
                total_size = len(dataset)
                train_size = int(0.7 * total_size)
                val_size = int(0.15 * total_size)
                test_size = total_size - train_size - val_size
                
                # Split dataset
                train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                    dataset,
                    [train_size, val_size, test_size],
                    generator=torch.Generator().manual_seed(Config.RANDOM_SEED)
                )
                
                # Create data loaders with custom collate function
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=Config.BATCH_SIZE,
                    shuffle=True,
                    collate_fn=custom_collate_fn
                )
                
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=Config.BATCH_SIZE,
                    shuffle=False,
                    collate_fn=custom_collate_fn
                )
                
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=Config.BATCH_SIZE,
                    shuffle=False,
                    collate_fn=custom_collate_fn
                )
                
                # Store datasets
                self.datasets[domain] = {
                    'train': train_loader,
                    'val': val_loader,
                    'test': test_loader
                }
                
                # Update domain statistics
                self.domain_stats[domain] = {
                    'train': len(train_dataset),
                    'val': len(val_dataset),
                    'test': len(test_dataset)
                }
            except Exception as e:
                print(f"Error loading {domain} dataset: {e}")
                continue
    
    def _convert_to_paper_format(self, data: List[Dict[str, Any]], domain: str) -> List[Dict[str, Any]]:
        """Convert data to the paper's format."""
        formatted_data = []
        
        for item in data:
            # Get user ID
            user_id = item.get('user_id', str(len(formatted_data)))
            
            # Get initial query
            query_initial = item.get('query_initial', item.get('query_history', [''])[0])
            
            # Get target document
            target_doc = item.get('target_doc', item.get('documents', [''])[0])
            
            # Paraphrase target document for intent info
            intent_info = self._paraphrase_document(target_doc)
            
            formatted_item = {
                'user_id': user_id,
                'query_initial': query_initial,
                'intent_info': intent_info,
                'target_doc': target_doc
            }
            
            formatted_data.append(formatted_item)
        
        return formatted_data
    
    def _paraphrase_document(self, doc: str) -> str:
        """Paraphrase document using ChatGPT."""
        try:
            from ..utils.llm_integration import LLMIntegration
            llm = LLMIntegration(Config())
            
            prompt = f"""Paraphrase the following text while maintaining its meaning:
            {doc}
            
            Paraphrase:"""
            
            response = llm.generate_text(prompt)
            return response.strip()
        except Exception as e:
            print(f"Error paraphrasing document: {e}")
            return doc
    
    def _load_domain_data(self, domain: str) -> List[Dict[str, Any]]:
        """Load raw data for a specific domain."""
        if domain == 'clariq':
            return self.load_clariq_dataset()
        elif domain == 'faqant':
            return self.load_faqant_dataset()
        elif domain == 'msdialog':
            return self.load_msdialog_dataset()
        elif domain == 'opendialkg':
            return self.load_opendialkg_dataset()
        else:
            raise ValueError(f"Unknown domain: {domain}")
    
    def load_clariq_dataset(self) -> List[Dict[str, Any]]:
        """Load ClariQ dataset from cache."""
        try:
            cache_path = os.path.join('cache', 'clariq_all_cache.pkl')
            if os.path.exists(cache_path):
                print(f"Loading cached data for clariq all...")
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Convert cached format to our format
                processed_data = []
                for item in cached_data:
                    processed_item = {
                        'query_history': [item['query_initial']],  # Start with initial query
                        'documents': [item['target_doc']],  # Use target doc as document
                        'retrieval_scores': item['retrieval_scores'],
                        'target_action': 0,  # Default action
                        'success': True,  # Default success
                        'num_turns': 1,  # Default turns
                        'target_doc': item['target_doc']
                    }
                    processed_data.append(processed_item)
                
                print(f"Loaded {len(processed_data)} samples from cache")
                return processed_data
            else:
                print(f"Cache file not found: {cache_path}")
                return []
        except Exception as e:
            print(f"Error loading ClariQ dataset from cache: {str(e)}")
            print("Stack trace:", traceback.format_exc())
            return []
    
    def load_faqant_dataset(self) -> List[Dict[str, Any]]:
        """Load FaqAnt dataset."""
        data = []
        
        # Load FAQ data
        faq_df = pd.read_csv('FaqAnt/data/faq.csv')
        
        # Process FAQ data
        for _, row in faq_df.iterrows():
            item = {
                'user_id': row.get('user_id', str(len(data))),
                'query_initial': row['question'],
                'target_doc': row['answer']
            }
            data.append(item)
        
        return data
    
    def load_msdialog_dataset(self) -> List[Dict[str, Any]]:
        """Load MSDialog dataset."""
        data = []
        
        # Load MSDialog data
        dialog_df = pd.read_csv('MSDialog/data/dialog.csv')
        
        # Process dialog data
        for _, row in dialog_df.iterrows():
            item = {
                'user_id': row.get('user_id', str(len(data))),
                'query_initial': row['query'],
                'target_doc': row['response']
            }
            data.append(item)
        
        return data
    
    def load_opendialkg_dataset(self) -> List[Dict[str, Any]]:
        """Load OpenDialKG dataset from cache."""
        try:
            cache_path = os.path.join('cache', 'opendialkg_all_cache.pkl')
            if os.path.exists(cache_path):
                print(f"Loading cached data for opendialkg all...")
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Convert cached format to our format
                processed_data = []
                for item in cached_data:
                    processed_item = {
                        'query_history': [item['query_initial']],  # Start with initial query
                        'documents': [item['target_doc']],  # Use target doc as document
                        'retrieval_scores': item['retrieval_scores'],
                        'target_action': 0,  # Default action
                        'success': True,  # Default success
                        'num_turns': 1,  # Default turns
                        'target_doc': item['target_doc']
                    }
                    processed_data.append(processed_item)
                
                print(f"Loaded {len(processed_data)} samples from cache")
                return processed_data
            else:
                print(f"Cache file not found: {cache_path}")
                return []
        except Exception as e:
            print(f"Error loading OpenDialKG dataset from cache: {str(e)}")
            print("Stack trace:", traceback.format_exc())
            return []
    
    def _balance_ambiguity(self, data: List[Dict[str, Any]], target_ambiguous_ratio: float = 0.4) -> List[Dict[str, Any]]:
        """Balance ambiguity in the dataset."""
        if not data:
            print("Warning: Empty data provided to _balance_ambiguity. Returning empty list.")
            return []
        
        # Separate ambiguous and unambiguous examples
        ambiguous_examples = [item for item in data if item.get('is_ambiguous', False)]
        unambiguous_examples = [item for item in data if not item.get('is_ambiguous', False)]
        
        # Calculate current ratio
        current_ratio = len(ambiguous_examples) / len(data)
        
        # Balance the dataset
        if current_ratio < target_ambiguous_ratio:
            # Need more ambiguous examples
            num_to_add = int((target_ambiguous_ratio * len(data) - len(ambiguous_examples)) / (1 - target_ambiguous_ratio))
            if num_to_add > 0:
                # Duplicate some ambiguous examples
                ambiguous_examples.extend(random.sample(ambiguous_examples, min(num_to_add, len(ambiguous_examples))))
        else:
            # Need more unambiguous examples
            num_to_add = int((len(ambiguous_examples) - target_ambiguous_ratio * len(data)) / target_ambiguous_ratio)
            if num_to_add > 0:
                # Duplicate some unambiguous examples
                unambiguous_examples.extend(random.sample(unambiguous_examples, min(num_to_add, len(unambiguous_examples))))
        
        # Combine and shuffle
        balanced_data = ambiguous_examples + unambiguous_examples
        random.shuffle(balanced_data)
        
        return balanced_data
    
    def get_domain_stats(self):
        """Get statistics for each domain."""
        return self.domain_stats
    
    def get_domain_data(self, domain: str, split: str = 'train') -> DataLoader:
        """
        Get DataLoader for a specific domain and split.
        
        Args:
            domain (str): Domain name
            split (str): Data split ('train', 'val', or 'test')
            
        Returns:
            DataLoader: DataLoader for the specified domain and split
        """
        if domain not in self.datasets:
            raise ValueError(f"Domain {domain} not found")
        
        if split not in self.datasets[domain]:
            raise ValueError(f"Split {split} not found for domain {domain}")
        
        return self.datasets[domain][split]
    
    def get_domain_batch(self, domain: str, split: str = 'train', batch_size: int = None) -> Dict[str, Any]:
        """
        Get a single batch of data for a specific domain and split.
        
        Args:
            domain (str): Domain name
            split (str): Data split ('train', 'val', or 'test')
            batch_size (int, optional): Batch size. If None, uses Config.BATCH_SIZE
            
        Returns:
            Dict[str, Any]: Batch of data
        """
        dataloader = self.get_domain_data(domain, split)
        batch_size = batch_size or Config.BATCH_SIZE
        
        # Get a single batch
        for batch in dataloader:
            return batch
        
        raise ValueError(f"No data available for domain {domain} and split {split}")
    
    def load_domains(self, domain_files: Dict[str, str]):
        """
        Load domain data from files.
        
        Args:
            domain_files (Dict[str, str]): Dictionary mapping domain names to their data file paths
        """
        for domain_name, file_path in domain_files.items():
            with open(file_path, 'r') as f:
                domain_data = json.load(f)
            self.domain_info[domain_name] = {
                'total_samples': len(domain_data),
                'file_path': file_path
            }
            
    def create_splits(self, 
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.1,
                     test_ratio: float = 0.1,
                     seed: int = 42):
        """
        Create train/val/test splits for each domain.
        
        Args:
            train_ratio (float): Ratio of data for training
            val_ratio (float): Ratio of data for validation
            test_ratio (float): Ratio of data for testing
            seed (int): Random seed for reproducibility
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1"
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        for domain_name, info in self.domain_info.items():
            # Load domain data
            with open(info['file_path'], 'r') as f:
                domain_data = json.load(f)
            
            # Shuffle data
            random.shuffle(domain_data)
            
            # Calculate split indices
            n_samples = len(domain_data)
            train_end = int(n_samples * train_ratio)
            val_end = train_end + int(n_samples * val_ratio)
            
            # Split data
            self.train_data[domain_name] = domain_data[:train_end]
            self.val_data[domain_name] = domain_data[train_end:val_end]
            self.test_data[domain_name] = domain_data[val_end:]
            
            # Save split information
            split_info = {
                'domain': domain_name,
                'total_samples': n_samples,
                'train_samples': len(self.train_data[domain_name]),
                'val_samples': len(self.val_data[domain_name]),
                'test_samples': len(self.test_data[domain_name])
            }
            
            # Save split info to file
            split_file = os.path.join(self.data_dir, 'splits', f'{domain_name}_splits.json')
            with open(split_file, 'w') as f:
                json.dump(split_info, f, indent=2)
            
            print(f"\nDomain: {domain_name}")
            print(f"Total samples: {n_samples}")
            print(f"Train samples: {split_info['train_samples']}")
            print(f"Val samples: {split_info['val_samples']}")
            print(f"Test samples: {split_info['test_samples']}")
    
    def save_splits(self):
        """Save all splits to disk."""
        for split_name in ['train', 'val', 'test']:
            split_data = getattr(self, f'{split_name}_data')
            for domain_name, data in split_data.items():
                file_path = os.path.join(self.data_dir, 'splits', f'{domain_name}_{split_name}.json')
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
    
    def load_splits(self):
        """Load all splits from disk."""
        for split_name in ['train', 'val', 'test']:
            for domain_name in self.domain_info:
                file_path = os.path.join(self.data_dir, 'splits', f'{domain_name}_{split_name}.json')
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        setattr(self, f'{split_name}_data', {domain_name: json.load(f)}) 