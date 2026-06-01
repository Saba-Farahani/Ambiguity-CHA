"""
Frozen BERT encoder for domain-invariant representations (STYLE Section 3.3).
Uses OpenMatch/cocodr-base-msmarco, first N transformer layers, mean pooling.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from ..config import Config
from typing import List


class BERTEncoder(nn.Module):
    """Frozen truncated BERT encoder; weights are never updated during training."""

    def __init__(self, config: Config = None, device=None):
        super().__init__()
        config = config or Config()
        model_name = config.BERT_MODEL_NAME
        n_layers = config.BERT_LAYERS

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        full_model = AutoModel.from_pretrained(model_name)

        self.embeddings = full_model.embeddings
        self.encoder_layers = nn.ModuleList(full_model.encoder.layer[:n_layers])
        self.hidden_size = full_model.config.hidden_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        for param in self.parameters():
            param.requires_grad = False

        self.to(self.device)
        self.eval()

    def encode(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        """
        Encode strings with mean pooling over non-padding tokens.

        Returns:
            Tensor of shape (len(texts), hidden_size).
        """
        if not texts:
            raise ValueError("Empty texts list provided to encode")

        valid_texts = []
        for text in texts:
            if isinstance(text, str) and text.strip() and text.lower() not in ("nan", "none", "null"):
                valid_texts.append(text.strip())
            else:
                valid_texts.append("")

        encoded = self.tokenizer(
            valid_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            hidden = self.embeddings(
                input_ids=encoded["input_ids"],
                token_type_ids=encoded.get("token_type_ids"),
            )
            attention_mask = encoded["attention_mask"]
            extended_mask = attention_mask[:, None, None, :]
            extended_mask = (1.0 - extended_mask) * -10000.0

            for layer in self.encoder_layers:
                layer_outputs = layer(hidden, extended_mask)
                hidden = layer_outputs[0]

            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        return pooled

    def forward(self, query_history: List[str], documents: List[str]):
        """
        Backward-compatible interface used by retriever/eval paths.

        Returns history embedding (full dialogue) and per-document embeddings.
        """
        history_text = " [SEP] ".join(query_history) if query_history else ""
        query_embedding = self.encode([history_text])
        doc_embeddings = self.encode(documents if documents else [""])
        if doc_embeddings.dim() == 2:
            doc_embeddings = doc_embeddings.unsqueeze(0)
        return {"query_embedding": query_embedding, "doc_embeddings": doc_embeddings}
