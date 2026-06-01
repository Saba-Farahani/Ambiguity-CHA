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
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from ..config import Config
from torch.utils.data import Dataset, DataLoader
from ..utils.llm_integration import LLMIntegration
from ..models.retriever import Retriever

def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function to handle variable-length sequences.
    
    Args:
        batch: List of dictionaries containing the data
        
    Returns:
        Dictionary containing batched data
    """
    # Initialize batched data
    batched_data = {
        'query_history': [],
        'documents': [],
        'retrieval_scores': [],
        'target_action': [],
        'success': [],
        'num_turns': [],
        'target_document': []  # Add target document
    }
    
    # Collect all data
    for item in batch:
        for key in batched_data:
            if key in item:  # Check if key exists in item
                batched_data[key].append(item[key])
            else:
                print(f"Warning: Missing key {key} in item")
                if key == 'target_document':
                    # Use first document as target if not specified
                    batched_data[key].append(item['documents'][0] if item['documents'] else None)
                else:
                    batched_data[key].append(None)
    
    # Convert lists to tensors where appropriate
    batched_data['target_action'] = torch.tensor(batched_data['target_action'], dtype=torch.float)
    batched_data['success'] = torch.tensor(batched_data['success'], dtype=torch.bool)
    batched_data['num_turns'] = torch.tensor(batched_data['num_turns'], dtype=torch.long)
    
    # Convert retrieval scores to tensor with consistent size
    max_docs = max(len(scores) for scores in batched_data['retrieval_scores'])
    max_docs = min(max_docs, Config.TOP_K_DOCS)  # Ensure we don't exceed TOP_K_DOCS
    
    padded_scores = []
    for scores in batched_data['retrieval_scores']:
        # Pad or truncate scores to max_docs length
        if len(scores) < max_docs:
            scores = scores + [0.0] * (max_docs - len(scores))
        else:
            scores = scores[:max_docs]
        padded_scores.append(scores)
    
    batched_data['retrieval_scores'] = torch.tensor(padded_scores, dtype=torch.float)
    
    # Ensure all tensors are on the same device
    device = batched_data['target_action'].device
    for key in batched_data:
        if isinstance(batched_data[key], torch.Tensor):
            batched_data[key] = batched_data[key].to(device)
    
    return batched_data

class ConversationDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

class CachedDataset(Dataset):
    """Dataset that caches processed batches."""
    
    def __init__(self, data: List[Dict[str, Any]], cache_dir: Path, domain: str, split: str):
        self.data = data
        self.cache_dir = cache_dir
        self.domain = domain
        self.split = split
        self.cache_path = cache_dir / f"{domain}_{split}_batches.pkl"
        self.processed_batches = self._load_or_create_cache()
    
    def _load_or_create_cache(self) -> Dict[int, Dict[str, Any]]:
        """Load cached batches or create new cache."""
        if self.cache_path.exists():
            print(f"Loading cached batches for {self.domain} {self.split}...")
            try:
                with open(self.cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
        
        print(f"Creating cache for {self.domain} {self.split}...")
        processed_batches = {}
        return processed_batches
    
    def _save_cache(self):
        """Save processed batches to cache."""
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.processed_batches, f)
            print(f"Saved batch cache for {self.domain} {self.split}")
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Check if batch is in cache
        if idx in self.processed_batches:
            return self.processed_batches[idx]
        
        # Process batch
        item = self.data[idx]
        processed_item = {
            'state': self._process_state(item),
            'target_doc': item['target_doc'],
            'retrieval_scores': item['retrieval_scores']
        }
        
        # Cache processed batch
        self.processed_batches[idx] = processed_item
        
        # Save cache periodically
        if idx % 10 == 0:
            self._save_cache()
        
        return processed_item
    
    def _process_state(self, item: Dict[str, Any]) -> torch.Tensor:
        """Process state for a single item."""
        # Convert state to tensor
        state = torch.tensor([
            item['retrieval_scores'][0] if item['retrieval_scores'] else 0.0
        ], dtype=torch.float32)
        return state

class DatasetManager:
    """Manages datasets for different domains."""
    
    def __init__(self):
        """Initialize dataset manager."""
        self.datasets = {}  # domain -> {train, val, test}
        self.doc_text_map = {}  # doc_id -> text
        self.domain_stats = {}
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self._load_datasets()
    
    def _get_cache_path(self, domain: str, split: str) -> Path:
        """Get cache file path for a domain and split."""
        return self.cache_dir / f"{domain}_{split}_cache.pkl"
    
    def _load_from_cache(self, domain: str, split: str) -> Optional[List[Dict[str, Any]]]:
        """Load processed data from cache."""
        cache_path = self._get_cache_path(domain, split)
        if cache_path.exists():
            print(f"Loading cached data for {domain} {split}...")
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
        return None
    
    def _save_to_cache(self, data: List[Dict[str, Any]], domain: str, split: str):
        """Save processed data to cache."""
        cache_path = self._get_cache_path(domain, split)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved cache for {domain} {split}")
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def _load_datasets(self):
        """Load datasets for all domains."""
        for domain in Config().DOMAINS:
            print(f"\nLoading {domain} dataset...")
            
            # Try to load from cache first
            cached_data = self._load_from_cache(domain, 'all')
            if cached_data is not None:
                self.datasets[domain] = self._create_datasets(cached_data, domain)
                continue
            
            # Load raw data
            raw_data = self._load_domain_data(domain)
            
            # Convert to paper format
            formatted_data = self._convert_to_paper_format(raw_data, domain)
            
            # Balance ambiguity
            balanced_data = self._balance_ambiguity(formatted_data)
            
            # Save to cache
            self._save_to_cache(balanced_data, domain, 'all')
            
            # Create datasets
            self.datasets[domain] = self._create_datasets(balanced_data, domain)
    
    def _create_datasets(self, data: List[Dict[str, Any]], domain: str) -> Dict[str, Dataset]:
        """Create train/val/test datasets."""
        # Split data
        n = len(data)
        train_size = int(n * 0.8)
        val_size = int(n * 0.1)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        # Create datasets
        return {
            'train': CachedDataset(train_data, self.cache_dir, domain, 'train'),
            'val': CachedDataset(val_data, self.cache_dir, domain, 'val'),
            'test': CachedDataset(test_data, self.cache_dir, domain, 'test')
        }
    
    def _convert_to_paper_format(self, raw_data: List[Dict[str, Any]], domain: str) -> List[Dict[str, Any]]:
        """Convert raw data to paper format."""
        # Try to load from cache first
        cache_path = self._get_cache_path(domain, 'formatted')
        if cache_path.exists():
            print(f"Loading formatted data from cache for {domain}...")
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
        
        print(f"Processing and caching data for {domain}...")
        formatted_data = []
        
        # Batch documents for paraphrasing
        documents_to_paraphrase = []
        doc_to_idx = {}
        
        # First pass: collect unique documents
        for i, item in enumerate(raw_data):
            target_doc = item['target_doc']
            if target_doc not in doc_to_idx:
                doc_to_idx[target_doc] = len(documents_to_paraphrase)
                documents_to_paraphrase.append(target_doc)
        
        # Batch paraphrase documents
        print(f"Paraphrasing {len(documents_to_paraphrase)} unique documents...")
        paraphrased_docs = self._batch_paraphrase_documents(documents_to_paraphrase)
        
        # Initialize retriever for computing scores
        retriever = Retriever(Config())
        retriever.load_documents(domain, documents_to_paraphrase)
        
        # Second pass: create formatted data
        for item in raw_data:
            target_doc = item['target_doc']
            idx = doc_to_idx[target_doc]
            intent_info = paraphrased_docs[idx]
            
            # Get retrieval scores
            _, retrieval_scores = retriever.retrieve(target_doc, domain)
            
            formatted_item = {
                'user_id': item['user_id'],
                'query_initial': item['query_initial'],
                'intent_info': intent_info,
                'target_doc': target_doc,
                'retrieval_scores': retrieval_scores
            }
            formatted_data.append(formatted_item)
        
        # Save to cache
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(formatted_data, f)
            print(f"Saved formatted data to cache for {domain}")
        except Exception as e:
            print(f"Error saving cache: {e}")
        
        return formatted_data
    
    def _batch_paraphrase_documents(self, documents: List[str]) -> List[str]:
        """Paraphrase a batch of documents efficiently."""
        llm = LLMIntegration(Config())
        paraphrased_docs = []
        
        # Process in smaller batches to avoid rate limits
        batch_size = 5
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
            
            for doc in batch:
                try:
                    paraphrased = llm.generate_text(doc)
                    paraphrased_docs.append(paraphrased)
                except Exception as e:
                    print(f"Error paraphrasing document: {e}")
                    # Use original document as fallback
                    paraphrased_docs.append(doc)
            
            # Add a small delay between batches to avoid rate limits
            if i + batch_size < len(documents):
                import time
                time.sleep(1)
        
        return paraphrased_docs
    
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
        """Load ClariQ dataset."""
        data = []
        
        # Load train data
        train_df = pd.read_csv('ClariQ/data/train.tsv', sep='\t')
        dev_df = pd.read_csv('ClariQ/data/dev.tsv', sep='\t')
        
        # Process train data
        for _, row in train_df.iterrows():
            # Get the target document from facet_desc
            target_doc = row['facet_desc']
            
            # Get the initial query
            query = row['initial_request']
            
            # Get the topic description as intent info
            intent_info = row['topic_desc']
            
            item = {
                'user_id': str(row['topic_id']),
                'query_initial': query,
                'intent_info': intent_info,
                'target_doc': target_doc
            }
            data.append(item)
        
        return data
    
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
        """Load OpenDialKG dataset."""
        data = []
        
        # Load OpenDialKG data
        df = pd.read_csv('opendialkg/data/opendialkg.csv')
        
        # Process each conversation
        for _, row in df.iterrows():
            try:
                # Parse the JSON string in the Messages column
                conversation = json.loads(row['Messages'])
                
                # Extract messages and metadata
                messages = []
                metadata = []
                
                for turn in conversation:
                    if turn['type'] == 'chat':
                        messages.append(turn['message'])
                    elif turn['type'] == 'action' and 'metadata' in turn:
                        if 'path' in turn['metadata']:
                            metadata.append(str(turn['metadata']['path']))
                
                if len(messages) >= 2:  # At least one turn of conversation
                    item = {
                        'user_id': row.get('user_id', str(len(data))),
                        'query_initial': messages[0],
                        'target_doc': messages[-1] if messages else ''
                    }
                    data.append(item)
            except Exception as e:
                print(f"Error processing OpenDialKG row: {e}")
                continue
        
        return data
    
    def _balance_ambiguity(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Balance data based on ambiguity scores."""
        # Try to load from cache first
        cache_path = self._get_cache_path('all', 'balanced')
        if cache_path.exists():
            print("Loading balanced data from cache...")
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
        
        print("Balancing data and caching...")
        # Compute ambiguity scores
        ambiguity_scores = []
        for example in data:
            scores = example['retrieval_scores']
            # Compute ambiguity as variance of scores
            ambiguity = np.var(scores) if len(scores) > 1 else 0.0
            ambiguity_scores.append(ambiguity)
        
        # Split into high and low ambiguity
        median_ambiguity = np.median(ambiguity_scores)
        high_amb = [d for d, s in zip(data, ambiguity_scores) if s > median_ambiguity]
        low_amb = [d for d, s in zip(data, ambiguity_scores) if s <= median_ambiguity]
        
        # Balance the dataset
        min_size = min(len(high_amb), len(low_amb))
        balanced_data = high_amb[:min_size] + low_amb[:min_size]
        
        # Save to cache
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(balanced_data, f)
            print("Saved balanced data to cache")
        except Exception as e:
            print(f"Error saving cache: {e}")
        
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
        
        return DataLoader(
            self.datasets[domain][split],
            batch_size=Config.BATCH_SIZE,
            shuffle=(split == 'train'),
            num_workers=Config.NUM_WORKERS,
            collate_fn=custom_collate_fn  # Use custom collate function
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
    
    def load_document_texts(self, file_path: str):
        """Load document text mapping from a JSON file.
        
        Args:
            file_path: Path to JSON file containing document texts
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Handle different JSON formats
            if isinstance(data, dict):
                # Format: {"doc_id": "text", ...}
                self.doc_text_map.update(data)
            elif isinstance(data, list):
                # Format: [{"doc_id": "...", "text": "..."}, ...]
                for item in data:
                    if isinstance(item, dict) and 'doc_id' in item and 'text' in item:
                        self.doc_text_map[item['doc_id']] = item['text']
            
            print(f"Loaded {len(self.doc_text_map)} document texts")
            
        except Exception as e:
            print(f"Error loading document texts: {e}")
            raise
    
    def get_document_text(self, doc_id: str) -> str:
        """Get text content for a document ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document text or empty string if not found
        """
        text = self.doc_text_map.get(doc_id, "")
        if not text:
            print(f"⚠️ Could not find text for doc_id: {doc_id}")
        return text
    
    def load_dataset(self, domain: str, split: str, file_path: str):
        """Load dataset for a domain and split.
        
        Args:
            domain: Domain name (e.g., 'clariq', 'opendialkg')
            split: Dataset split ('train', 'val', 'test')
            file_path: Path to dataset file
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Process each example
            processed_data = []
            for example in data:
                # Get document texts
                doc_ids = example.get('documents', [])
                doc_texts = [self.get_document_text(doc_id) for doc_id in doc_ids]
                
                # Update example with document texts
                example['documents'] = doc_texts
                processed_data.append(example)
            
            # Store processed data
            if domain not in self.datasets:
                self.datasets[domain] = {}
            self.datasets[domain][split] = processed_data
            
            print(f"Loaded {len(processed_data)} examples for {domain} {split}")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    
    def get_dataset(self, domain: str, split: str) -> List[Dict[str, Any]]:
        """Get dataset for a domain and split.
        
        Args:
            domain: Domain name
            split: Dataset split
            
        Returns:
            List of examples
        """
        if domain not in self.datasets or split not in self.datasets[domain]:
            raise ValueError(f"No dataset found for {domain} {split}")
        return self.datasets[domain][split]
    
    def get_all_domains(self) -> List[str]:
        """Get list of all domains."""
        return list(self.datasets.keys())
    
    def get_domain_splits(self, domain: str) -> List[str]:
        """Get list of splits available for a domain."""
        if domain not in self.datasets:
            raise ValueError(f"No dataset found for domain: {domain}")
        return list(self.datasets[domain].keys()) 