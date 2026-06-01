"""
Document text loading and management utilities.
"""

import json
import csv
import os
from typing import Dict, Optional, List
from pathlib import Path

class DocumentLoader:
    """Handles loading and managing document texts from various sources."""
    
    def __init__(self, config):
        """Initialize document loader.
        
        Args:
            config: Configuration object containing paths and settings
        """
        self.config = config
        self.doc_text_map = {}  # doc_id -> text
        self.loaded_sources = set()
        
    def load_from_json(self, file_path: str) -> Dict[str, str]:
        """Load document texts from a JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Dictionary mapping document IDs to their text
            
        Raises:
            ValueError: If file is invalid or contains invalid data
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            doc_text_map = {}
            
            # Handle different JSON formats
            if isinstance(data, dict):
                # Format: {"doc_id": "text", ...}
                for doc_id, text in data.items():
                    if not isinstance(doc_id, str):
                        raise ValueError(f"Invalid document ID type in JSON: {type(doc_id)}")
                    if not isinstance(text, str):
                        raise ValueError(f"Invalid text type in JSON for {doc_id}: {type(text)}")
                    if not text.strip():
                        raise ValueError(f"Empty text in JSON for {doc_id}")
                    doc_text_map[doc_id] = text
                    
            elif isinstance(data, list):
                # Format: [{"doc_id": "...", "text": "..."}, ...]
                for item in data:
                    if not isinstance(item, dict):
                        raise ValueError(f"Invalid item type in JSON: {type(item)}")
                    if 'doc_id' not in item or 'text' not in item:
                        raise ValueError(f"Missing doc_id or text in JSON item: {item}")
                    if not isinstance(item['doc_id'], str):
                        raise ValueError(f"Invalid document ID type in JSON: {type(item['doc_id'])}")
                    if not isinstance(item['text'], str):
                        raise ValueError(f"Invalid text type in JSON for {item['doc_id']}: {type(item['text'])}")
                    if not item['text'].strip():
                        raise ValueError(f"Empty text in JSON for {item['doc_id']}")
                    doc_text_map[item['doc_id']] = item['text']
            
            else:
                raise ValueError(f"Invalid JSON format: {type(data)}")
            
            return doc_text_map
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file {file_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading JSON from {file_path}: {e}")
    
    def load_from_tsv(self, file_path: str) -> Dict[str, str]:
        """Load document texts from a TSV file.
        
        Args:
            file_path: Path to TSV file
            
        Returns:
            Dictionary mapping document IDs to their text
            
        Raises:
            ValueError: If file is invalid or contains invalid data
        """
        try:
            doc_text_map = {}
            with open(file_path, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                for i, row in enumerate(reader, 1):
                    if len(row) < 2:
                        raise ValueError(f"Invalid row {i} in TSV: {row}")
                    doc_id, text = row[0], row[1]
                    if not isinstance(doc_id, str):
                        raise ValueError(f"Invalid document ID type in TSV row {i}: {type(doc_id)}")
                    if not isinstance(text, str):
                        raise ValueError(f"Invalid text type in TSV row {i} for {doc_id}: {type(text)}")
                    if not text.strip():
                        raise ValueError(f"Empty text in TSV row {i} for {doc_id}")
                    doc_text_map[doc_id] = text
            return doc_text_map
            
        except Exception as e:
            raise ValueError(f"Error loading TSV from {file_path}: {e}")
    
    def load_from_directory(self, directory: str, pattern: str = "*.json") -> Dict[str, str]:
        """Load document texts from all matching files in a directory.
        
        Args:
            directory: Directory containing document files
            pattern: File pattern to match (e.g., "*.json", "*.tsv")
            
        Returns:
            Dictionary mapping document IDs to their text
        """
        doc_text_map = {}
        directory = Path(directory)
        
        for file_path in directory.glob(pattern):
            if file_path.suffix == '.json':
                doc_text_map.update(self.load_from_json(str(file_path)))
            elif file_path.suffix == '.tsv':
                doc_text_map.update(self.load_from_tsv(str(file_path)))
        
        return doc_text_map
    
    def load_all_sources(self) -> None:
        """Load document texts from all configured sources.
        
        Raises:
            ValueError: If any source contains invalid data
        """
        try:
            # Load from main document text file
            main_path = os.path.join(self.config.DATA_DIR, "document_texts.json")
            if os.path.exists(main_path):
                print(f"Loading main document texts from {main_path}")
                self.doc_text_map.update(self.load_from_json(main_path))
                self.loaded_sources.add(main_path)
            
            # Load from ClueWeb09 directory if configured
            clueweb_dir = os.path.join(self.config.DATA_DIR, "clueweb09")
            if os.path.exists(clueweb_dir):
                print(f"Loading ClueWeb09 documents from {clueweb_dir}")
                clueweb_texts = self.load_from_directory(clueweb_dir)
                self.doc_text_map.update(clueweb_texts)
                self.loaded_sources.add(clueweb_dir)
            
            # Load from OpenDialKG directory if configured
            opendialkg_dir = os.path.join(self.config.DATA_DIR, "opendialkg")
            if os.path.exists(opendialkg_dir):
                print(f"Loading OpenDialKG documents from {opendialkg_dir}")
                opendialkg_texts = self.load_from_directory(opendialkg_dir)
                self.doc_text_map.update(opendialkg_texts)
                self.loaded_sources.add(opendialkg_dir)
            
            if not self.doc_text_map:
                raise ValueError("No document texts loaded from any source")
                
            print(f"Loaded {len(self.doc_text_map)} document texts from {len(self.loaded_sources)} sources")
            
            # Validate all loaded texts
            invalid_docs = []
            for doc_id, text in self.doc_text_map.items():
                if not isinstance(text, str) or not text.strip():
                    invalid_docs.append(doc_id)
            
            if invalid_docs:
                raise ValueError(f"Found {len(invalid_docs)} invalid documents: {invalid_docs[:5]}...")
            
        except Exception as e:
            raise ValueError(f"Error loading document texts: {e}")
    
    def get_document_text(self, doc_id: str) -> str:
        """Get text content for a document ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document text
            
        Raises:
            ValueError: If document ID is invalid or text is missing
        """
        if not doc_id:
            raise ValueError("Empty document ID")
            
        if not isinstance(doc_id, str):
            raise ValueError(f"Invalid document ID type: {type(doc_id)}")
        
        # Try to get actual text
        text = self.doc_text_map.get(doc_id)
        
        # Validate text
        if text is None:
            raise ValueError(f"Missing text for document {doc_id}")
            
        if not isinstance(text, str):
            raise ValueError(f"Invalid text type for document {doc_id}: {type(text)}")
            
        if not text.strip():
            raise ValueError(f"Empty text for document {doc_id}")
        
        return text
    
    def get_document_texts(self, doc_ids: List[str]) -> List[str]:
        """Get text content for multiple document IDs.
        
        Args:
            doc_ids: List of document IDs
            
        Returns:
            List of document texts
        """
        return [self.get_document_text(doc_id) for doc_id in doc_ids]
    
    def verify_document_texts(self, doc_ids: List[str]) -> Dict[str, bool]:
        """Verify which document IDs have text content.
        
        Args:
            doc_ids: List of document IDs to check
            
        Returns:
            Dictionary mapping document IDs to whether they have text
        """
        return {doc_id: bool(self.doc_text_map.get(doc_id, "")) 
                for doc_id in doc_ids}
    
    def validate_document_ids(self, doc_ids: List[str]) -> List[str]:
        """Validate a list of document IDs and return missing ones.
        
        Args:
            doc_ids: List of document IDs to validate
            
        Returns:
            List of missing document IDs
        """
        missing = []
        for doc_id in doc_ids:
            if not doc_id or not isinstance(doc_id, str):
                continue
            if doc_id not in self.doc_text_map:
                missing.append(doc_id)
        return missing 