"""
BERT-based feature extractor for domain-invariant representations.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from ..config import Config


class BERTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def _encode_text(self, texts, max_length=512):
        """Encode a list of texts using BERT."""
        # Handle both single string and list of strings
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize and encode
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # Move to device
        device = next(self.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.bert(**encoded)
            embeddings = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token embedding

        return embeddings

    def forward(self, query_history, documents):
        """
        Encode query history and documents.

        Args:
            query_history: List of query strings or list of lists of query strings
            documents: List of document strings or list of lists of document strings

        Returns:
            Dict containing query and document embeddings
        """
        # Handle batch processing
        batch_size = len(query_history)

        # Process query history
        if isinstance(query_history[0], list):
            # If query_history is a list of lists (batch of queries)
            query_embeddings = []
            for queries in query_history:
                # Join queries with [SEP] token
                combined_query = " [SEP] ".join(queries)
                query_emb = self._encode_text(combined_query)
                query_embeddings.append(query_emb)
            query_embedding = torch.cat(query_embeddings, dim=0)
        else:
            # If query_history is a single list of queries
            combined_query = " [SEP] ".join(query_history)
            query_embedding = self._encode_text(combined_query)
            # Repeat for batch size if needed
            if batch_size > 1:
                query_embedding = query_embedding.repeat(batch_size, 1)

        # Process documents
        if isinstance(documents[0], list):
            # If documents is a list of lists (batch of documents)
            doc_embeddings = []
            for docs in documents:
                # Encode each document
                doc_embs = self._encode_text(docs)
                doc_embeddings.append(doc_embs)
            doc_embeddings = torch.stack(doc_embeddings)
        else:
            # If documents is a single list of documents
            doc_embeddings = self._encode_text(documents)
            # Repeat for batch size if needed
            if batch_size > 1:
                doc_embeddings = doc_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)

        return {
            "query_embedding": query_embedding,  # [batch_size, bert_dim]
            "doc_embeddings": doc_embeddings,  # [batch_size, num_docs, bert_dim]
        }
