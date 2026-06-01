"""
Dataset management for STYLE implementation.
Handles dataset splits, domain organization, and data loading.
"""

import os
import json
import random
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from ..config import Config
from torch.utils.data import Dataset, DataLoader

class ConversationDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

class DatasetManager:
    def __init__(self):
        """Initialize the dataset manager."""
        self.datasets = {}
        self.domain_stats = {}
        self._load_datasets()
    
    def _load_datasets(self):
        """Load and preprocess datasets for each domain."""
        # Example domains (replace with your actual domains)
        domains = ['travel', 'restaurant', 'movie']
        
        for domain in domains:
            # Load raw data for the domain
            raw_data = self._load_domain_data(domain)
            
            # Convert to dataset format
            dataset = ConversationDataset(raw_data)
            
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
            
            # Store datasets
            self.datasets[domain] = {
                'train': train_dataset,
                'val': val_dataset,
                'test': test_dataset
            }
            
            # Update domain statistics
            self.domain_stats[domain] = {
                'train': len(train_dataset),
                'val': len(val_dataset),
                'test': len(test_dataset)
            }
    
    def _load_domain_data(self, domain: str) -> List[Dict[str, Any]]:
        """
        Load raw data for a specific domain.
        Replace this with your actual data loading logic.
        """
        # Example data structure (replace with your actual data)
        if domain == 'travel':
            return [
                {
                    'query_history': ['What are some good places to visit in San Diego?'],
                    'documents': ['Balboa Park is a great place to visit', 'The San Diego Zoo is world famous', 'La Jolla Cove offers beautiful views'],
                    'retrieval_scores': [0.9, 0.8, 0.7],
                    'target_action': 1.0,  # 1.0 for ask, 0.0 for answer
                    'success': True,
                    'num_turns': 1
                },
                {
                    'query_history': ['What is the best time to visit San Diego?', 'I prefer warm weather'],
                    'documents': ['Summer months are warm and dry', 'Spring offers mild temperatures', 'Winter is mild but can be rainy'],
                    'retrieval_scores': [0.85, 0.75, 0.65],
                    'target_action': 0.0,
                    'success': True,
                    'num_turns': 2
                },
                {
                    'query_history': ['What hotels do you recommend?', 'I want something near the beach'],
                    'documents': ['Hotel del Coronado is a historic beachfront hotel', 'The Grand Hyatt is downtown', 'The La Jolla Shores Hotel is beachfront'],
                    'retrieval_scores': [0.95, 0.7, 0.9],
                    'target_action': 0.0,
                    'success': True,
                    'num_turns': 2
                }
            ]
        elif domain == 'restaurant':
            return [
                {
                    'query_history': ['What restaurants are in downtown San Diego?'],
                    'documents': ['The Lionfish serves seafood', 'Civico 1845 offers Italian cuisine', 'The Prado is in Balboa Park'],
                    'retrieval_scores': [0.9, 0.8, 0.7],
                    'target_action': 1.0,
                    'success': True,
                    'num_turns': 1
                },
                {
                    'query_history': ['Do you have any vegetarian options?', 'I prefer Italian food'],
                    'documents': ['Civico 1845 has vegetarian pasta', 'The Prado has a vegetarian menu', 'Lionfish has seafood options'],
                    'retrieval_scores': [0.95, 0.85, 0.6],
                    'target_action': 0.0,
                    'success': True,
                    'num_turns': 2
                }
            ]
        else:  # movie domain
            return [
                {
                    'query_history': ['What movies are playing this weekend?'],
                    'documents': ['Avengers: Endgame is showing', 'The Lion King is a new release', 'Toy Story 4 is family friendly'],
                    'retrieval_scores': [0.9, 0.8, 0.7],
                    'target_action': 1.0,
                    'success': True,
                    'num_turns': 1
                },
                {
                    'query_history': ['What time is the 7pm showing?', 'I need tickets for 2'],
                    'documents': ['7pm showing is at AMC', 'Tickets are $12 each', 'Online booking available'],
                    'retrieval_scores': [0.95, 0.9, 0.8],
                    'target_action': 0.0,
                    'success': True,
                    'num_turns': 2
                }
            ]
    
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
        
        return DataLoader(
            self.datasets[domain][split],
            batch_size=Config.BATCH_SIZE,
            shuffle=(split == 'train'),
            num_workers=Config.NUM_WORKERS
        )
    
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