"""
Data conversion script for ClariQ and OpenDialKG datasets.
Converts the datasets into the format required by our training system.
"""

import os
import json
import pickle
import tarfile
import pandas as pd
import numpy as np
import argparse
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from ..config import Config

class DataConverter:
    def __init__(self, output_dir: str = Config.DATA_DIR):
        """
        Initialize the data converter.
        
        Args:
            output_dir (str): Directory to save converted data
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def convert_clariq(self, clariq_dir: str):
        """
        Convert ClariQ dataset to our format.
        
        Args:
            clariq_dir (str): Directory containing ClariQ dataset files
        """
        print("Converting ClariQ dataset...")
        
        # Load and process train data
        train_data = self._load_clariq_data(
            os.path.join(clariq_dir, 'train.tsv'),
            os.path.join(clariq_dir, 'train.qrel')
        )
        
        # Load and process dev data
        dev_data = self._load_clariq_data(
            os.path.join(clariq_dir, 'dev.tsv'),
            os.path.join(clariq_dir, 'dev.qrel')
        )
        
        # Load and process test data
        test_data = self._load_clariq_data(
            os.path.join(clariq_dir, 'test.tsv'),
            os.path.join(clariq_dir, 'test.qrel')
        )
        
        # Save converted data
        self._save_converted_data('clariq', {
            'train': train_data,
            'dev': dev_data,
            'test': test_data
        })
        
        print(f"ClariQ conversion complete. Data saved to {self.output_dir}")
        
    def _load_clariq_data(self, tsv_file: str, qrel_file: str) -> List[Dict]:
        """
        Load and process ClariQ data from TSV and QREL files.
        
        Args:
            tsv_file (str): Path to TSV file containing queries and documents
            qrel_file (str): Path to QREL file containing relevance scores
            
        Returns:
            List[Dict]: List of processed samples
        """
        print(f"Loading TSV file: {tsv_file}")
        
        # Check if this is the test set
        is_test = 'test' in tsv_file
        
        if is_test:
            # For test set, load both test.tsv and test_with_labels.tsv
            initial_df = pd.read_csv(tsv_file, sep='\t', header=None, 
                                   names=['topic_id', 'initial_request'])
            labels_df = pd.read_csv(tsv_file.replace('test.tsv', 'test_with_labels.tsv'), 
                                  sep='\t')
            print(f"Loaded {len(initial_df)} initial queries and {len(labels_df)} labeled samples")
            # Load the full test_with_labels.tsv for topic_id to facet_id mapping
            df = pd.read_csv(tsv_file.replace('test.tsv', 'test_with_labels.tsv'), sep='\t')
        else:
            # For train/dev, load the regular TSV with all columns
            df = pd.read_csv(tsv_file, sep='\t', header=None, 
                           names=['topic_id', 'initial_request', 'topic_desc', 'clarification_need',
                                 'facet_id', 'facet_desc', 'question_id', 'question', 'answer'])
            print(f"Loaded {len(df)} rows from TSV file")
        
        print(f"Loading QREL file: {qrel_file}")
        # Load QREL data with correct format: query_id 0 doc_id relevance
        qrel_df = pd.read_csv(qrel_file, sep=' ', header=None,
                             names=['facet_id', 'zero', 'doc_id', 'relevance'])
        print(f"Loaded {len(qrel_df)} rows from QREL file")
        
        samples = []
        debug_prints = 0
        DEBUG_LIMIT = 5  # Only print for the first 5 queries
        
        if is_test:
            # Process test data
            for _, row in tqdm(initial_df.iterrows(), desc="Processing test queries"):
                topic_id = row['topic_id']
                initial_query = row['initial_request']
                
                # Get all clarifications for this topic
                topic_clarifications = labels_df[labels_df['topic_id'] == topic_id]
                
                # Build query history
                query_history = [initial_query]
                if not topic_clarifications.empty:
                    # Add clarifications to history
                    clarifications = topic_clarifications['question'].tolist()
                    query_history.extend(clarifications)
                
                # Get documents and scores from QREL for this topic
                topic_facets = df[df['topic_id'].astype(str) == str(topic_id)]['facet_id'].unique()
                if len(topic_facets) == 0:
                    print(f"Warning: No facet_id found for topic {topic_id}")
                    continue
                
                # Use the first facet_id for this topic
                facet_id = topic_facets[0]
                topic_docs = qrel_df[qrel_df['facet_id'] == facet_id].sort_values('relevance', ascending=False)
                
                # Take top-k documents
                docs = topic_docs['doc_id'].tolist()[:Config.TOP_K_DOCS]
                scores = topic_docs['relevance'].tolist()[:Config.TOP_K_DOCS]
                
                # Pad if needed
                if len(docs) < Config.TOP_K_DOCS:
                    docs.extend([''] * (Config.TOP_K_DOCS - len(docs)))
                    scores.extend([0.0] * (Config.TOP_K_DOCS - len(scores)))
                
                # Debug print for first few queries
                if debug_prints < DEBUG_LIMIT:
                    print(f"[DEBUG] TEST Query ID: {topic_id}")
                    print(f"[DEBUG]   Facet ID: {facet_id}")
                    print(f"[DEBUG]   Query: {initial_query}")
                    print(f"[DEBUG]   Docs: {docs}")
                    print(f"[DEBUG]   Scores: {scores}")
                    debug_prints += 1
                
                # Create sample
                sample = {
                    'query_history': query_history,
                    'documents': docs,
                    'retrieval_scores': scores,
                    'target_action': 1 if any(s > 0 for s in scores) else 0,
                    'success': any(s > 0 for s in scores),
                    'num_turns': len(query_history)
                }
                samples.append(sample)
        else:
            # Process train/dev data
            for topic_id, topic_group in tqdm(df.groupby('topic_id'), desc="Processing queries"):
                # Get all facets for this topic
                topic_facets = topic_group['facet_id'].unique()
                
                # Process each facet
                for facet_id in topic_facets:
                    # Get all questions for this facet
                    facet_questions = topic_group[topic_group['facet_id'] == facet_id]
                    
                    # Build query history
                    query_history = [facet_questions.iloc[0]['initial_request']]  # Start with initial request
                    query_history.extend(facet_questions['question'].tolist())  # Add all questions
                    
                    # Get documents and scores from QREL for this facet
                    facet_docs = qrel_df[qrel_df['facet_id'] == facet_id].sort_values('relevance', ascending=False)
                    
                    # Take top-k documents
                    docs = facet_docs['doc_id'].tolist()[:Config.TOP_K_DOCS]
                    scores = facet_docs['relevance'].tolist()[:Config.TOP_K_DOCS]
                    
                    # Pad if needed
                    if len(docs) < Config.TOP_K_DOCS:
                        docs.extend([''] * (Config.TOP_K_DOCS - len(docs)))
                        scores.extend([0.0] * (Config.TOP_K_DOCS - len(scores)))
                    
                    # Debug print for first few queries
                    if debug_prints < DEBUG_LIMIT:
                        print(f"[DEBUG] TRAIN/DEV Topic ID: {topic_id}")
                        print(f"[DEBUG]   Facet ID: {facet_id}")
                        print(f"[DEBUG]   Initial Query: {query_history[0]}")
                        print(f"[DEBUG]   Docs: {docs}")
                        print(f"[DEBUG]   Scores: {scores}")
                        debug_prints += 1
                    
                    # Create sample
                    sample = {
                        'query_history': query_history,
                        'documents': docs,
                        'retrieval_scores': scores,
                        'target_action': 1 if any(s > 0 for s in scores) else 0,
                        'success': any(s > 0 for s in scores),
                        'num_turns': len(query_history)
                    }
                    samples.append(sample)
        
        print(f"Processed {len(samples)} samples")
        return samples
    
    def convert_opendialkg(self, opendialkg_dir: str):
        """
        Convert OpenDialKG dataset to our format.
        
        Args:
            opendialkg_dir (str): Directory containing OpenDialKG dataset files
        """
        print("Converting OpenDialKG dataset...")
        
        # Load the main dialogue data
        dialogue_file = os.path.join(opendialkg_dir, 'data', 'opendialkg.csv')
        print(f"Loading dialogue data from: {dialogue_file}")
        
        # Load and process the data
        data = self._load_opendialkg_data(dialogue_file)
        
        # Split into train/dev/test (80/10/10 split)
        total_samples = len(data)
        train_size = int(0.8 * total_samples)
        dev_size = int(0.1 * total_samples)
        
        train_data = data[:train_size]
        dev_data = data[train_size:train_size + dev_size]
        test_data = data[train_size + dev_size:]
        
        # Save converted data
        self._save_converted_data('opendialkg', {
            'train': train_data,
            'dev': dev_data,
            'test': test_data
        })
        
        print(f"OpenDialKG conversion complete. Data saved to {self.output_dir}")
    
    def _load_opendialkg_data(self, file_path: str) -> List[Dict]:
        """
        Load and process OpenDialKG data from CSV file.
        
        Args:
            file_path (str): Path to OpenDialKG data file
            
        Returns:
            List[Dict]: List of processed samples
        """
        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows from CSV file")
        print("CSV columns:", df.columns.tolist())
        
        # Check if we have the required columns
        if 'Messages' not in df.columns:
            raise ValueError(f"Expected 'Messages' column in CSV file. Found columns: {df.columns.tolist()}")
        
        samples = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing dialogs"):
            try:
                # Parse the JSON-formatted Messages column
                messages = json.loads(row['Messages'])
                
                # Extract user and assistant messages
                query_history = []
                documents = []
                
                for msg in messages:
                    if msg['type'] == 'chat':
                        if msg['sender'] == 'user':
                            query_history.append(msg['message'])
                        elif msg['sender'] == 'assistant':
                            documents.append(msg['message'])
                
                # Skip if no valid dialogue
                if not query_history or not documents:
                    continue
                
                # Pad or truncate documents to TOP_K_DOCS
                if len(documents) < Config.TOP_K_DOCS:
                    documents.extend([''] * (Config.TOP_K_DOCS - len(documents)))
                else:
                    documents = documents[:Config.TOP_K_DOCS]
                
                # Generate dummy retrieval scores (since OpenDialKG doesn't have them)
                # We'll use a simple decreasing score based on position
                scores = [1.0 - (i * 0.1) for i in range(Config.TOP_K_DOCS)]
                
                # Get ratings if available, default to 5 if missing
                try:
                    user_rating = json.loads(row['User Rating'])['dialog_rating'] if pd.notna(row['User Rating']) else '5'
                    assistant_rating = json.loads(row['Assistant Rating'])['dialog_rating'] if pd.notna(row['Assistant Rating']) else '5'
                except (KeyError, json.JSONDecodeError):
                    # If ratings are missing or invalid, use default
                    user_rating = '5'
                    assistant_rating = '5'
                
                # Calculate success based on ratings (consider it successful if both ratings are 4 or higher)
                success = float(user_rating) >= 4 and float(assistant_rating) >= 4
                
                sample = {
                    'query_history': query_history,
                    'documents': documents,
                    'retrieval_scores': scores,
                    'target_action': 1 if success else 0,
                    'success': success,
                    'num_turns': len(query_history)
                }
                samples.append(sample)
                
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping row due to JSON decode error: {e}")
                continue
            except Exception as e:
                print(f"Warning: Skipping row due to error: {e}")
                continue
        
        print(f"Processed {len(samples)} samples")
        return samples
    
    def _save_converted_data(self, dataset_name: str, data: Dict[str, List[Dict]]):
        """
        Save converted data to JSON files.
        
        Args:
            dataset_name (str): Name of the dataset
            data (Dict[str, List[Dict]]): Dictionary containing train/dev/test data
        """
        for split_name, split_data in data.items():
            output_file = os.path.join(self.output_dir, f'{dataset_name}_{split_name}.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, indent=2)
            print(f"Saved {len(split_data)} samples to {output_file}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert datasets to required format')
    parser.add_argument('--dataset', type=str, choices=['clariq', 'opendialkg', 'all'],
                      default='all', help='Dataset to convert (default: all)')
    args = parser.parse_args()
    
    # Initialize converter
    converter = DataConverter()
    
    # Convert selected dataset(s)
    if args.dataset in ['clariq', 'all']:
        clariq_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ClariQ', 'data')
        if os.path.exists(clariq_dir):
            converter.convert_clariq(clariq_dir)
        else:
            print(f"ClariQ directory not found at {clariq_dir}")
    
    if args.dataset in ['opendialkg', 'all']:
        opendialkg_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'opendialkg')
        if os.path.exists(opendialkg_dir):
            converter.convert_opendialkg(opendialkg_dir)
        else:
            print(f"OpenDialKG directory not found at {opendialkg_dir}")

if __name__ == "__main__":
    main() 