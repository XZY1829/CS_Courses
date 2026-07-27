from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.crf import CRF


def _parse_bio_label(label: str) -> Tuple[str, Optional[str]]:
    if label == "O":
        return "O", None
    if label.startswith("B-"):
        return "B", label[2:]
    if label.startswith("I-"):
        return "I", label[2:]
    return "OTHER", label


def _build_bio_constraints(label_names: Sequence[str]) -> Dict[str, torch.Tensor]:
    num_labels = len(label_names)
    allowed_start = torch.ones(num_labels, dtype=torch.bool)
    allowed_end = torch.ones(num_labels, dtype=torch.bool)
    allowed_transitions = torch.ones((num_labels, num_labels), dtype=torch.bool)

    parsed = [_parse_bio_label(label) for label in label_names]
    for i, (prefix_i, type_i) in enumerate(parsed):
        if prefix_i == "I":
            allowed_start[i] = False

    for prev_idx, (prev_prefix, prev_type) in enumerate(parsed):
        for curr_idx, (curr_prefix, curr_type) in enumerate(parsed):
            if curr_prefix == "I":
                is_valid = (prev_prefix in {"B", "I"}) and (prev_type == curr_type)
                allowed_transitions[prev_idx, curr_idx] = is_valid

    return {
        "allowed_start": allowed_start,
        "allowed_end": allowed_end,
        "allowed_transitions": allowed_transitions,
    }


class CharCNNEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_filters: int,
        kernel_sizes: Sequence[int],
        pad_idx: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                    padding=k // 2,
                )
                for k in kernel_sizes
            ]
        )
        self.output_dim = num_filters * len(kernel_sizes)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        # char_ids: [batch, seq_len, max_char_len]
        batch_size, seq_len, max_char_len = char_ids.shape
        x = char_ids.view(batch_size * seq_len, max_char_len)
        x = self.embedding(x)  # [B*L, C, E]
        x = x.transpose(1, 2)  # [B*L, E, C]

        conv_outputs = []
        for conv in self.convs:
            y = F.relu(conv(x))
            y = torch.max(y, dim=2).values
            conv_outputs.append(y)

        x = torch.cat(conv_outputs, dim=1)
        return x.view(batch_size, seq_len, self.output_dim)


class BiLSTMCRF(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        word_pad_idx: int,
        word_unk_idx: int,
        embedding_dim: int = 100,
        hidden_dim: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.33,
        word_dropout: float = 0.05,
        use_char_cnn: bool = False,
        char_vocab_size: int = 0,
        char_pad_idx: int = 0,
        char_embedding_dim: int = 30,
        char_num_filters: int = 50,
        char_kernel_sizes: Optional[Sequence[int]] = None,
        pretrained_word_embeddings: Optional[torch.Tensor] = None,
        freeze_word_embeddings: bool = False,
        crf_constraint: str = "none",
        label_names: Optional[Sequence[str]] = None,
        use_pos_chunk_aux: bool = False,
        num_pos_labels: int = 0,
        num_chunk_labels: int = 0,
    ) -> None:
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even for bidirectional LSTM")

        self.word_pad_idx = word_pad_idx
        self.word_unk_idx = word_unk_idx
        self.word_dropout = word_dropout
        self.use_char_cnn = use_char_cnn
        self.use_pos_chunk_aux = use_pos_chunk_aux

        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=word_pad_idx)
        if pretrained_word_embeddings is not None:
            if pretrained_word_embeddings.shape != self.word_embedding.weight.shape:
                raise ValueError(
                    "Shape mismatch for pretrained embeddings: "
                    f"expected {tuple(self.word_embedding.weight.shape)}, "
                    f"got {tuple(pretrained_word_embeddings.shape)}"
                )
            with torch.no_grad():
                self.word_embedding.weight.copy_(pretrained_word_embeddings)
        self.word_embedding.weight.requires_grad = not freeze_word_embeddings

        input_dim = embedding_dim
        if use_char_cnn:
            if char_kernel_sizes is None:
                char_kernel_sizes = [3, 4, 5]
            self.char_encoder = CharCNNEncoder(
                vocab_size=char_vocab_size,
                embedding_dim=char_embedding_dim,
                num_filters=char_num_filters,
                kernel_sizes=char_kernel_sizes,
                pad_idx=char_pad_idx,
            )
            input_dim += self.char_encoder.output_dim
        else:
            self.char_encoder = None

        self.embedding_dropout = nn.Dropout(dropout)
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)
        if self.use_pos_chunk_aux:
            if num_pos_labels <= 0 or num_chunk_labels <= 0:
                raise ValueError(
                    "num_pos_labels and num_chunk_labels must be positive when use_pos_chunk_aux=True"
                )
            self.pos_classifier = nn.Linear(hidden_dim, num_pos_labels)
            self.chunk_classifier = nn.Linear(hidden_dim, num_chunk_labels)
        else:
            self.pos_classifier = None
            self.chunk_classifier = None

        constraint_mode = str(crf_constraint).lower()
        if constraint_mode in {"none", ""}:
            constraints = None
        elif constraint_mode == "bio":
            if label_names is None:
                raise ValueError("label_names is required when crf_constraint='bio'")
            constraints = _build_bio_constraints(label_names)
        else:
            raise ValueError(f"Unsupported crf_constraint: {crf_constraint}")

        self.crf = CRF(num_labels, constraints=constraints)

    def _apply_word_dropout(self, word_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.word_dropout <= 0.0:
            return word_ids
        noise = torch.rand(word_ids.shape, device=word_ids.device)
        dropout_mask = (
            (noise < self.word_dropout)
            & mask
            & (word_ids != self.word_pad_idx)
            & (word_ids != self.word_unk_idx)
        )
        return word_ids.masked_fill(dropout_mask, self.word_unk_idx)

    def _encode_features(
        self, word_ids: torch.Tensor, mask: torch.Tensor, char_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        word_ids = self._apply_word_dropout(word_ids, mask)
        word_embed = self.word_embedding(word_ids)

        features = [word_embed]
        if self.use_char_cnn:
            if char_ids is None:
                raise ValueError("char_ids is required when use_char_cnn=True")
            char_features = self.char_encoder(char_ids)
            features.append(char_features)

        x = torch.cat(features, dim=-1)
        x = self.embedding_dropout(x)
        x, _ = self.encoder(x)
        x = self.output_dropout(x)
        return x

    def _compute_emissions(
        self, word_ids: torch.Tensor, mask: torch.Tensor, char_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        encoded = self._encode_features(word_ids, mask, char_ids)
        return self.classifier(encoded)

    def forward(
        self,
        word_ids: torch.Tensor,
        mask: torch.Tensor,
        tags: Optional[torch.Tensor] = None,
        char_ids: Optional[torch.Tensor] = None,
        pos_ids: Optional[torch.Tensor] = None,
        chunk_ids: Optional[torch.Tensor] = None,
        aux_loss_weight: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        encoded = self._encode_features(word_ids, mask, char_ids)
        emissions = self.classifier(encoded)
        if tags is None:
            raise ValueError("tags cannot be None in forward(); call decode() for inference")
        ner_loss = self.crf.neg_log_likelihood(emissions, tags, mask)

        pos_loss = torch.zeros((), dtype=ner_loss.dtype, device=ner_loss.device)
        chunk_loss = torch.zeros((), dtype=ner_loss.dtype, device=ner_loss.device)
        total_loss = ner_loss

        if self.use_pos_chunk_aux:
            if self.pos_classifier is None or self.chunk_classifier is None:
                raise RuntimeError("Auxiliary classifiers are not initialized.")
            if pos_ids is None or chunk_ids is None:
                raise ValueError("pos_ids and chunk_ids are required when use_pos_chunk_aux=True")

            pos_logits = self.pos_classifier(encoded)
            chunk_logits = self.chunk_classifier(encoded)
            pos_loss = F.cross_entropy(
                pos_logits.view(-1, pos_logits.size(-1)),
                pos_ids.view(-1),
                ignore_index=-100,
            )
            chunk_loss = F.cross_entropy(
                chunk_logits.view(-1, chunk_logits.size(-1)),
                chunk_ids.view(-1),
                ignore_index=-100,
            )
            total_loss = ner_loss + float(aux_loss_weight) * 0.5 * (pos_loss + chunk_loss)

        return {
            "total_loss": total_loss,
            "ner_loss": ner_loss,
            "pos_loss": pos_loss,
            "chunk_loss": chunk_loss,
        }

    def decode(
        self, word_ids: torch.Tensor, mask: torch.Tensor, char_ids: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        emissions = self._compute_emissions(word_ids, mask, char_ids)
        return self.crf.decode(emissions, mask)
