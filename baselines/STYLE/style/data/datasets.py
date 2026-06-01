"""
Dataset handling for STYLE implementation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle
import tarfile
import os
import json
from typing import List, Dict, Any
from ..config import Config
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

def pad_sequence(sequence: List[str], max_length: int, pad_value: str = "") -> List[str]:
    """Pad or truncate a sequence to a fixed length."""
    if len(sequence) > max_length:
        return sequence[:max_length]
    return sequence + [pad_value] * (max_length - len(sequence))

class DomainDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], domain_name: str):
        self.data = data
        self.domain_name = domain_name
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Pad query history and documents to fixed lengths
        query_history = pad_sequence(item['query_history'], Config.MAX_QUERY_LENGTH)
        documents = pad_sequence(item['documents'], Config.TOP_K_DOCS)
        
        # Ensure retrieval scores match TOP_K_DOCS
        retrieval_scores = item['retrieval_scores']
        if len(retrieval_scores) < Config.TOP_K_DOCS:
            retrieval_scores = retrieval_scores + [0.0] * (Config.TOP_K_DOCS - len(retrieval_scores))
        elif len(retrieval_scores) > Config.TOP_K_DOCS:
            retrieval_scores = retrieval_scores[:Config.TOP_K_DOCS]
        
        # Convert to tensors with proper shapes
        retrieval_scores = torch.tensor(retrieval_scores, dtype=torch.float).view(1, -1)  # [1, TOP_K_DOCS]
        target_action = torch.tensor(item['target_action'], dtype=torch.float).view(1)  # [1]
        
        return {
            'query_history': query_history,
            'documents': documents,
            'retrieval_scores': retrieval_scores,
            'target_action': target_action,
            'success': item['success'],
            'num_turns': item['num_turns'],
            'domain': self.domain_name
        }

class MultiDomainDataset:
    def __init__(self, domains: Dict[str, List[Dict[str, Any]]]):
        self.domains = {
            domain_name: DomainDataset(data, domain_name)
            for domain_name, data in domains.items()
        }
        
    def sample_domains(self, num_domains: int, samples_per_domain: int):
        """Sample domains and create data loaders."""
        selected_domains = np.random.choice(
            list(self.domains.keys()),
            size=num_domains,
            replace=False
        )
        
        domain_loaders = []
        for domain_name in selected_domains:
            dataset = self.domains[domain_name]
            # Sample indices
            indices = np.random.choice(
                len(dataset),
                size=min(samples_per_domain, len(dataset)),
                replace=False
            )
            
            # Create subset
            subset = torch.utils.data.Subset(dataset, indices)
            loader = DataLoader(
                subset,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                drop_last=True,  # Drop last incomplete batch
                collate_fn=self._collate_fn  # Use custom collate function
            )
            domain_loaders.append(loader)
        
        return domain_loaders
    
    def _collate_fn(self, batch):
        """Custom collate function to handle batching properly."""
        # Get batch size
        batch_size = len(batch)
        
        # Collate query histories
        query_histories = [item['query_history'] for item in batch]
        
        # Collate documents
        documents = [item['documents'] for item in batch]
        
        # Stack retrieval scores and target actions
        retrieval_scores = torch.cat([item['retrieval_scores'] for item in batch], dim=0)  # [batch_size, TOP_K_DOCS]
        target_actions = torch.cat([item['target_action'] for item in batch], dim=0)  # [batch_size]
        
        return {
            'query_history': query_histories,
            'documents': documents,
            'retrieval_scores': retrieval_scores,
            'target_action': target_actions,
            'success': [item['success'] for item in batch],
            'num_turns': [item['num_turns'] for item in batch],
            'domain': batch[0]['domain']  # All items in batch are from same domain
        }
    
    def get_domain_loader(self, domain_name: str, batch_size: int = None):
        """Get DataLoader for a specific domain."""
        if domain_name not in self.domains:
            raise ValueError(f"Domain {domain_name} not found")
        
        return DataLoader(
            self.domains[domain_name],
            batch_size=batch_size or Config.BATCH_SIZE,
            shuffle=True,
            drop_last=True  # Drop last incomplete batch
        )
    
    def get_all_domains(self):
        """Get list of all domain names."""
        return list(self.domains.keys())

def load_clariq_data():
    """Load ClariQ dataset."""
    data = []
    
    # Load train data
    train_df = pd.read_csv('ClariQ/data/train.tsv', sep='\t')
    dev_df = pd.read_csv('ClariQ/data/dev.tsv', sep='\t')
    
    # Load question bank
    question_bank = pd.read_csv('ClariQ/data/question_bank.tsv', sep='\t')
    
    # Process train data
    for _, row in train_df.iterrows():
        item = {
            'query_history': [row['initial_request']],
            'documents': question_bank['question'].tolist()[:Config.TOP_K_DOCS],
            'retrieval_scores': np.random.uniform(0, 1, size=Config.TOP_K_DOCS).tolist(),
            'target_action': 1 if row['clarification_need'] > 2 else 0,  # Convert to binary
            'success': True,  # We don't have this information in the dataset
            'num_turns': 1
        }
        data.append(item)
    
    return data

def load_opendialkg_data():
    """Load OpenDialKG dataset."""
    data = []
    
    # Load conversations from OpenDialKG
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
                # Ensure we have enough documents
                if len(metadata) < Config.TOP_K_DOCS:
                    metadata.extend([''] * (Config.TOP_K_DOCS - len(metadata)))
                elif len(metadata) > Config.TOP_K_DOCS:
                    metadata = metadata[:Config.TOP_K_DOCS]
                
                item = {
                    'query_history': messages[:2],  # First two turns
                    'documents': metadata,
                    'retrieval_scores': np.random.uniform(0, 1, size=Config.TOP_K_DOCS).tolist(),
                    'target_action': 1 if len(messages) > 2 else 0,  # If there are more turns, we needed clarification
                    'success': True,  # We don't have this information in the dataset
                    'num_turns': len(messages)
                }
                data.append(item)
        except json.JSONDecodeError:
            continue  # Skip malformed JSON
    
    return data

def load_benchmark_datasets():
    """Load ClariQ and OpenDialKG datasets."""
    domains = {
        'clariq': load_clariq_data(),
        'opendialkg': load_opendialkg_data()
    }
    return MultiDomainDataset(domains)

class ConversationDataset(Dataset):
    """Dataset class for conversation data following paper format."""
    
    def __init__(self, data: List[Dict[str, Any]]):
        """Initialize dataset with conversation data."""
        self.data = data
        
        # Initialize BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Initialize BERT model for embeddings
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model.eval()  # Set to evaluation mode
        
        # Move model to device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert_model = self.bert_model.to(self.device)
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example from the dataset."""
        example = self.data[idx]
        
        # Get BERT embeddings for query and documents
        query_embedding = self._get_bert_embedding(example['query_initial'])
        intent_embedding = self._get_bert_embedding(example['intent_info'])
        target_embedding = self._get_bert_embedding(example['target_doc'])
        
        # Calculate similarity scores
        query_intent_sim = self._cosine_similarity(query_embedding, intent_embedding)
        query_target_sim = self._cosine_similarity(query_embedding, target_embedding)
        
        # Create state representation
        state = torch.cat([
            query_embedding,
            intent_embedding,
            target_embedding,
            query_intent_sim.unsqueeze(0),
            query_target_sim.unsqueeze(0)
        ])
        
        return {
            'state': state,
            'user_id': example['user_id'],
            'query_initial': example['query_initial'],
            'intent_info': example['intent_info'],
            'target_doc': example['target_doc']
        }
    
    def _get_bert_embedding(self, text: str) -> torch.Tensor:
        """Get BERT embedding for a text."""
        # Tokenize text
        tokens = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Move tokens to device
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.bert_model(**tokens)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :]
        
        return embedding.squeeze(0)
    
    def _cosine_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate cosine similarity between two tensors."""
        return F.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0)) 