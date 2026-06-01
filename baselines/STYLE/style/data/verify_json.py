"""
Script to verify the structure and validity of generated JSON files.
"""

import os
import json
from typing import Dict, List
import pandas as pd

def verify_json_file(file_path: str) -> Dict:
    """
    Verify a JSON file's structure and content.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        Dict: Statistics about the file
    """
    print(f"\nAnalyzing {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Basic validation
        if not isinstance(data, list):
            raise ValueError("Root element must be a list")
        
        # Collect statistics
        stats = {
            'total_samples': len(data),
            'fields': set(),
            'query_history_lengths': [],
            'document_lengths': [],
            'success_count': 0,
            'target_action_1_count': 0,
            'invalid_samples': []
        }
        
        # Analyze each sample
        for i, sample in enumerate(data):
            # Check required fields
            required_fields = {'query_history', 'documents', 'retrieval_scores', 
                             'target_action', 'success', 'num_turns'}
            sample_fields = set(sample.keys())
            stats['fields'].update(sample_fields)
            
            if not required_fields.issubset(sample_fields):
                stats['invalid_samples'].append({
                    'index': i,
                    'reason': f"Missing required fields. Found: {sample_fields}"
                })
                continue
            
            # Validate field types and values
            if not isinstance(sample['query_history'], list):
                stats['invalid_samples'].append({
                    'index': i,
                    'reason': "query_history must be a list"
                })
                continue
                
            if not isinstance(sample['documents'], list):
                stats['invalid_samples'].append({
                    'index': i,
                    'reason': "documents must be a list"
                })
                continue
                
            if not isinstance(sample['retrieval_scores'], list):
                stats['invalid_samples'].append({
                    'index': i,
                    'reason': "retrieval_scores must be a list"
                })
                continue
            
            # Collect statistics
            stats['query_history_lengths'].append(len(sample['query_history']))
            stats['document_lengths'].append(len(sample['documents']))
            if sample['success']:
                stats['success_count'] += 1
            if sample['target_action'] == 1:
                stats['target_action_1_count'] += 1
        
        # Calculate additional statistics
        stats['avg_query_history_length'] = sum(stats['query_history_lengths']) / len(stats['query_history_lengths'])
        stats['avg_document_length'] = sum(stats['document_lengths']) / len(stats['document_lengths'])
        stats['success_rate'] = stats['success_count'] / stats['total_samples']
        stats['target_action_1_rate'] = stats['target_action_1_count'] / stats['total_samples']
        
        # Print results
        print(f"Total samples: {stats['total_samples']}")
        print(f"Fields found: {sorted(stats['fields'])}")
        print(f"Average query history length: {stats['avg_query_history_length']:.2f}")
        print(f"Average document length: {stats['avg_document_length']:.2f}")
        print(f"Success rate: {stats['success_rate']:.2%}")
        print(f"Target action 1 rate: {stats['target_action_1_rate']:.2%}")
        
        if stats['invalid_samples']:
            print(f"\nFound {len(stats['invalid_samples'])} invalid samples:")
            for invalid in stats['invalid_samples']:
                print(f"  Sample {invalid['index']}: {invalid['reason']}")
        else:
            print("\nAll samples are valid!")
        
        return stats
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {str(e)}")
        return None
    except Exception as e:
        print(f"Error analyzing file: {str(e)}")
        return None

def main():
    # Get the data directory (root STYLE directory)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    
    # List of files to verify
    files = [
        'clariq_train.json',
        'clariq_dev.json',
        'clariq_test.json',
        'opendialkg_train.json',
        'opendialkg_dev.json',
        'opendialkg_test.json'
    ]
    
    # Verify each file
    all_stats = {}
    for file_name in files:
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            stats = verify_json_file(file_path)
            if stats:
                all_stats[file_name] = stats
        else:
            print(f"\nFile not found: {file_path}")
    
    # Compare statistics across files
    if all_stats:
        print("\nComparing statistics across files:")
        comparison = pd.DataFrame({
            'total_samples': [stats['total_samples'] for stats in all_stats.values()],
            'avg_query_length': [stats['avg_query_history_length'] for stats in all_stats.values()],
            'avg_doc_length': [stats['avg_document_length'] for stats in all_stats.values()],
            'success_rate': [stats['success_rate'] for stats in all_stats.values()],
            'target_action_1_rate': [stats['target_action_1_rate'] for stats in all_stats.values()]
        }, index=all_stats.keys())
        print("\n", comparison)

if __name__ == "__main__":
    main() 