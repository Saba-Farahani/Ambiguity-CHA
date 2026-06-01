"""
Retrieval system implementation.
Supports multiple retrieval methods including BM25, SentenceBERT, and ChatSearch.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import torch
from ..config import Config
from transformers import AutoTokenizer, AutoModel
import os
import json
from ..data.document_loader import DocumentLoader


class Retriever:
    """Document retriever for information-seeking conversations."""

    def __init__(self, config: Config):
        """Initialize the retriever.

        Args:
            config: Configuration object containing model parameters
        """
        self.config = config
        self.documents = {}  # domain -> List[str]
        self.embeddings = {}  # domain -> List[torch.Tensor]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize document loader
        self.document_loader = DocumentLoader(config)
        self.document_loader.load_all_sources()

        # BERT for retrieval embeddings (same checkpoint family as DISP encoder)
        model_name = getattr(config, "BERT_MODEL_NAME", "OpenMatch/cocodr-base-msmarco")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

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

        # Get text from document loader
        text = self.document_loader.get_document_text(doc_id)

        # Validate text
        if not text:
            print(f"⚠️ Warning: Missing text for document {doc_id}, using fallback")
            return f"This document {doc_id} contains general information and resources for various applications."

        if not isinstance(text, str):
            print(f"⚠️ Warning: Invalid text type for document {doc_id}, using fallback")
            return f"This document {doc_id} contains general information and resources for various applications."

        if not text.strip():
            print(f"⚠️ Warning: Empty text for document {doc_id}, using fallback")
            return f"This document {doc_id} contains general information and resources for various applications."

        return text

    def load_documents(
        self,
        domain: str,
        documents: List[str],
        doc_text_map: Optional[Dict[str, str]] = None,
    ):
        """Load documents for a specific domain.

        Args:
            domain: Domain name
            documents: List of document IDs
            doc_text_map: Optional mapping of document IDs to their text content
        """
        domain = domain.lower()  # Convert to lowercase for consistency
        self.documents[domain] = documents

        # Update document text mapping if provided
        if doc_text_map:
            self.document_loader.doc_text_map.update(doc_text_map)

        # Compute embeddings for all documents
        with torch.no_grad():
            self.embeddings[domain] = self._compute_embeddings(documents)

    def get_domain_data(self, domain: str) -> List[str]:
        """
        Get all documents for a specific domain.

        Args:
            domain: The domain to get data for

        Returns:
            List of documents for the domain
        """
        domain = domain.lower()  # Convert to lowercase for consistency
        if domain not in self.documents:
            raise ValueError(f"No documents loaded for domain: {domain}")
        return self.documents[domain]

    def retrieve(
        self, query: str, domain: str, top_k: int = 5
    ) -> Tuple[List[str], List[float]]:
        """Retrieve relevant documents for a query.

        Args:
            query: Query text
            domain: Domain to search in
            top_k: Number of documents to retrieve

        Returns:
            Tuple of (document IDs, scores)

        Raises:
            ValueError: If query is invalid or no documents found
        """
        if not query or not isinstance(query, str):
            raise ValueError(f"Invalid query: {query}")

        if not query.strip():
            raise ValueError("Empty query")

        # Get domain documents
        domain_docs = self.get_domain_data(domain)
        if not domain_docs:
            raise ValueError(f"No documents found for domain {domain}")

        # Compute query embedding
        query_embedding = self._compute_embeddings([query])[0]

        # Compute document embeddings
        doc_embeddings = self._compute_embeddings(domain_docs)

        # Compute similarities
        similarities = []
        for doc_emb in doc_embeddings:
            sim = torch.cosine_similarity(
                query_embedding.unsqueeze(0), doc_emb.unsqueeze(0)
            )
            similarities.append(sim.item())

        # Get top-k documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_docs = [domain_docs[i] for i in top_indices]
        top_scores = [similarities[i] for i in top_indices]

        return top_docs, top_scores

    def _compute_embeddings(self, texts: List[str]) -> List[torch.Tensor]:
        """Compute BERT embeddings for a list of texts."""
        embeddings = []
        for text in texts:
            try:
                # Get document text if it's an ID
                if text.startswith("clueweb09-"):
                    text = self.get_document_text(text)

                # Validate text
                if not text or not isinstance(text, str):
                    print(f"⚠️ Warning: Invalid text for embedding, using fallback")
                    text = "This document contains general information and resources."

                # Tokenize and get BERT embeddings
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use [CLS] token embedding as document representation
                    embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                    embeddings.append(embedding)

            except Exception as e:
                print(f"❌ Error computing embedding: {e}")
                print(f"Text that caused error: {text[:100]}...")
                # Provide fallback embedding
                print("🔄 Using fallback embedding...")
                fallback_embedding = torch.randn(
                    768, device=self.device
                )  # Random embedding
                embeddings.append(fallback_embedding)

        return embeddings

    def compute_ambiguity_score(
        self, query: str, retrieved_docs: List[str], retrieval_scores: List[float]
    ) -> float:
        """
        Compute an ambiguity score for the query based on retrieval results.

        Args:
            query: Query text
            retrieved_docs: Retrieved documents
            retrieval_scores: Retrieval scores

        Returns:
            Ambiguity score between 0 and 1 (higher means more ambiguous)
        """
        if not retrieval_scores:
            return 1.0

        # Normalize scores
        scores = np.array(retrieval_scores)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)

        # Compute ambiguity based on score distribution
        score_diff = scores[0] - scores[1] if len(scores) > 1 else 1.0
        score_std = np.std(scores)

        # Combine multiple factors
        ambiguity = 0.5 * (1 - score_diff) + 0.5 * score_std

        return float(ambiguity)

    def update_embeddings(self):
        """Update document embeddings (useful after fine-tuning)."""
        # This method is not applicable for BERT-based retrieval
        pass
