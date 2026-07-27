from collections import Counter
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.utils import load_json, save_json

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass
class NERExample:
    tokens: List[str]
    tags: List[str]
    pos_tags: Optional[List[int]] = None
    chunk_tags: Optional[List[int]] = None


class Vocab:
    def __init__(self, stoi: Dict[str, int], itos: List[str]) -> None:
        self.stoi = stoi
        self.itos = itos

    @classmethod
    def build(
        cls,
        sequences: Iterable[Sequence[str]],
        min_freq: int = 1,
        max_size: Optional[int] = None,
        specials: Optional[List[str]] = None,
    ) -> "Vocab":
        specials = specials or [PAD_TOKEN, UNK_TOKEN]
        counter = Counter()
        for seq in sequences:
            counter.update(seq)

        words = [w for w, c in counter.items() if c >= min_freq]
        words.sort(key=lambda w: (-counter[w], w))
        if max_size is not None:
            words = words[: max(0, max_size - len(specials))]

        itos = list(specials) + words
        stoi = {token: idx for idx, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_idx(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.stoi[UNK_TOKEN]

    def encode(self, tokens: Sequence[str], lowercase: bool = False) -> List[int]:
        ids = []
        for token in tokens:
            key = token.lower() if lowercase else token
            ids.append(self.stoi.get(key, self.unk_idx))
        return ids

    def to_dict(self) -> Dict[str, Any]:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Vocab":
        itos = data["itos"]
        stoi = {token: idx for idx, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)


class ConllNERDataset(Dataset):
    def __init__(
        self,
        examples: List[NERExample],
        word_vocab: Vocab,
        label2id: Dict[str, int],
        lowercase: bool = False,
        char_vocab: Optional[Vocab] = None,
    ) -> None:
        self.examples = examples
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.label2id = label2id
        self.lowercase = lowercase

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]
        word_ids = self.word_vocab.encode(example.tokens, lowercase=self.lowercase)
        tag_ids = [self.label2id[tag] for tag in example.tags]
        pos_ids = example.pos_tags
        chunk_ids = example.chunk_tags

        char_ids = None
        if self.char_vocab is not None:
            char_ids = [
                self.char_vocab.encode(list(token), lowercase=False) for token in example.tokens
            ]

        return {
            "tokens": example.tokens,
            "tags": example.tags,
            "word_ids": word_ids,
            "tag_ids": tag_ids,
            "char_ids": char_ids,
            "pos_ids": pos_ids,
            "chunk_ids": chunk_ids,
        }


class NERCollator:
    def __init__(self, word_pad_idx: int, char_pad_idx: int = 0, use_char: bool = False) -> None:
        self.word_pad_idx = word_pad_idx
        self.char_pad_idx = char_pad_idx
        self.use_char = use_char

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_size = len(batch)
        seq_lens = [len(item["word_ids"]) for item in batch]
        max_len = max(seq_lens)

        word_ids = torch.full((batch_size, max_len), self.word_pad_idx, dtype=torch.long)
        tag_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

        char_ids = None
        pos_ids = None
        chunk_ids = None
        if self.use_char:
            max_char_len = 1
            for item in batch:
                for token_char_ids in item["char_ids"]:
                    max_char_len = max(max_char_len, len(token_char_ids))

            char_ids = torch.full(
                (batch_size, max_len, max_char_len), self.char_pad_idx, dtype=torch.long
            )

        if any(item.get("pos_ids") is not None for item in batch):
            pos_ids = torch.full((batch_size, max_len), -100, dtype=torch.long)
        if any(item.get("chunk_ids") is not None for item in batch):
            chunk_ids = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for i, item in enumerate(batch):
            length = seq_lens[i]
            word_ids[i, :length] = torch.tensor(item["word_ids"], dtype=torch.long)
            tag_ids[i, :length] = torch.tensor(item["tag_ids"], dtype=torch.long)
            mask[i, :length] = True

            if self.use_char and char_ids is not None:
                for j, token_char_ids in enumerate(item["char_ids"]):
                    c_len = len(token_char_ids)
                    if c_len > 0:
                        char_ids[i, j, :c_len] = torch.tensor(token_char_ids, dtype=torch.long)
            if pos_ids is not None and item.get("pos_ids") is not None:
                pos_ids[i, :length] = torch.tensor(item["pos_ids"], dtype=torch.long)
            if chunk_ids is not None and item.get("chunk_ids") is not None:
                chunk_ids[i, :length] = torch.tensor(item["chunk_ids"], dtype=torch.long)

        return {
            "word_ids": word_ids,
            "tag_ids": tag_ids,
            "mask": mask,
            "char_ids": char_ids,
            "pos_ids": pos_ids,
            "chunk_ids": chunk_ids,
            "tokens": [item["tokens"] for item in batch],
            "tags": [item["tags"] for item in batch],
        }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return _project_root() / path


def _load_pretrained_embedding_matrix(
    embedding_path: str,
    word_vocab: Vocab,
    embedding_dim: int,
    lowercase: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    resolved_path = _resolve_path(embedding_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Pretrained embedding file not found: {resolved_path}")

    matrix = torch.empty((len(word_vocab), embedding_dim), dtype=torch.float)
    torch.nn.init.normal_(matrix, mean=0.0, std=0.1)
    matrix[word_vocab.pad_idx].zero_()

    specials = {PAD_TOKEN, UNK_TOKEN}
    target_vocab = len(word_vocab) - sum(1 for tok in word_vocab.itos if tok in specials)
    target_vocab = max(target_vocab, 1)

    hits = 0
    malformed_lines = 0
    with resolved_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) <= embedding_dim:
                # Skip likely header lines such as "400000 300".
                continue

            token = parts[0]
            vector_values = parts[1:]
            if len(vector_values) != embedding_dim:
                malformed_lines += 1
                continue

            key = token.lower() if lowercase else token
            idx = word_vocab.stoi.get(key)
            if idx is None:
                continue

            try:
                vector = torch.tensor([float(v) for v in vector_values], dtype=torch.float)
            except ValueError:
                malformed_lines += 1
                continue

            matrix[idx] = vector
            if key not in specials:
                hits += 1

    stats = {
        "pretrained_hits": float(hits),
        "pretrained_target_vocab": float(target_vocab),
        "pretrained_coverage": float(hits) / float(target_vocab),
        "pretrained_malformed_lines": float(malformed_lines),
        "pretrained_embedding_dim": float(embedding_dim),
    }
    return matrix, stats


def load_conll2003(
    dataset_name: str = "conll2003",
) -> Tuple[List[NERExample], List[NERExample], List[NERExample], List[str], List[str], List[str]]:
    # CoNLL-2003 currently ships with a dataset loading script on HF Hub.
    # trust_remote_code=True is required to execute that loader.
    print(
        f"  load_dataset({dataset_name!r}) starting - may download from HuggingFace or prepare cache (wait if it looks stuck)...",
        flush=True,
    )
    ds = load_dataset(dataset_name, trust_remote_code=True)
    print("  load_dataset finished; reading label names...", flush=True)
    label_names = ds["train"].features["ner_tags"].feature.names
    pos_label_names = ds["train"].features["pos_tags"].feature.names
    chunk_label_names = ds["train"].features["chunk_tags"].feature.names

    def convert(split: str) -> List[NERExample]:
        examples = []
        n = ds[split].num_rows
        print(f"  Materializing split {split!r} ({n} examples) into memory...", flush=True)
        for row in tqdm(
            ds[split],
            desc=f"  {split}",
            total=n,
            file=sys.stderr,
            disable=False,
            mininterval=0.3,
            leave=True,
            dynamic_ncols=True,
        ):
            tokens = row["tokens"]
            tags = [label_names[idx] for idx in row["ner_tags"]]
            pos_tags = [int(idx) for idx in row["pos_tags"]]
            chunk_tags = [int(idx) for idx in row["chunk_tags"]]
            examples.append(NERExample(tokens=tokens, tags=tags, pos_tags=pos_tags, chunk_tags=chunk_tags))
        return examples

    train_examples = convert("train")
    dev_examples = convert("validation")
    test_examples = convert("test")
    print("  All splits materialized.", flush=True)
    return train_examples, dev_examples, test_examples, label_names, pos_label_names, chunk_label_names


def build_label_maps(label_names: Sequence[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    label2id = {label: idx for idx, label in enumerate(label_names)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


@dataclass
class DataBundle:
    train_loader: DataLoader
    dev_loader: DataLoader
    test_loader: DataLoader
    word_vocab: Vocab
    char_vocab: Optional[Vocab]
    label2id: Dict[str, int]
    id2label: Dict[int, str]
    pretrained_word_embeddings: Optional[torch.Tensor] = None
    pretrained_stats: Optional[Dict[str, float]] = None
    num_pos_labels: int = 0
    num_chunk_labels: int = 0


def build_data_bundle(
    config: Dict[str, Any],
    word_vocab: Optional[Vocab] = None,
    char_vocab: Optional[Vocab] = None,
    label2id: Optional[Dict[str, int]] = None,
) -> DataBundle:
    data_cfg = config["data"]
    word_vocab_provided = word_vocab is not None
    train_examples, dev_examples, test_examples, label_names, pos_label_names, chunk_label_names = load_conll2003(
        dataset_name=data_cfg.get("dataset_name", "conll2003")
    )
    if label2id is None:
        label2id, id2label = build_label_maps(label_names)
    else:
        id2label = {idx: label for label, idx in label2id.items()}

    lowercase = bool(data_cfg.get("lowercase", False))
    use_char = bool(config["model"].get("use_char_cnn", False))
    min_freq = int(data_cfg.get("min_freq", 1))
    max_vocab_size = data_cfg.get("max_vocab_size", None)
    pretrained_word_embeddings = None
    pretrained_stats: Optional[Dict[str, float]] = None

    if word_vocab is None:
        print("  Building word vocabulary from training tokens...", flush=True)
        token_sequences = [
            [token.lower() if lowercase else token for token in ex.tokens] for ex in train_examples
        ]
        word_vocab = Vocab.build(
            token_sequences,
            min_freq=min_freq,
            max_size=max_vocab_size,
            specials=[PAD_TOKEN, UNK_TOKEN],
        )

    pretrained_path = data_cfg.get("pretrained_embeddings_path", None)
    if (
        pretrained_path
        and isinstance(pretrained_path, str)
        and pretrained_path.strip()
        and (not word_vocab_provided)
    ):
        embedding_dim = int(config["model"].get("embedding_dim", 100))
        print(f"  Loading pretrained embeddings from: {pretrained_path}", flush=True)
        pretrained_word_embeddings, pretrained_stats = _load_pretrained_embedding_matrix(
            embedding_path=pretrained_path,
            word_vocab=word_vocab,
            embedding_dim=embedding_dim,
            lowercase=lowercase,
        )
        if pretrained_stats is not None:
            coverage = pretrained_stats["pretrained_coverage"] * 100.0
            print(
                f"  Pretrained embedding coverage: {coverage:.2f}% "
                f"({int(pretrained_stats['pretrained_hits'])}/{int(pretrained_stats['pretrained_target_vocab'])})",
                flush=True,
            )

    if use_char and char_vocab is None:
        print("  Building character vocabulary...", flush=True)
        char_sequences = []
        for ex in train_examples:
            for token in ex.tokens:
                char_sequences.append(list(token))
        char_vocab = Vocab.build(
            char_sequences,
            min_freq=1,
            max_size=None,
            specials=[PAD_TOKEN, UNK_TOKEN],
        )

    train_ds = ConllNERDataset(
        train_examples, word_vocab=word_vocab, char_vocab=char_vocab, label2id=label2id, lowercase=lowercase
    )
    dev_ds = ConllNERDataset(
        dev_examples, word_vocab=word_vocab, char_vocab=char_vocab, label2id=label2id, lowercase=lowercase
    )
    test_ds = ConllNERDataset(
        test_examples, word_vocab=word_vocab, char_vocab=char_vocab, label2id=label2id, lowercase=lowercase
    )

    collator = NERCollator(
        word_pad_idx=word_vocab.pad_idx,
        char_pad_idx=char_vocab.pad_idx if char_vocab is not None else 0,
        use_char=use_char,
    )
    batch_size = int(config["training"].get("batch_size", 32))
    num_workers = int(config["training"].get("num_workers", 0))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )

    return DataBundle(
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        word_vocab=word_vocab,
        char_vocab=char_vocab,
        label2id=label2id,
        id2label=id2label,
        pretrained_word_embeddings=pretrained_word_embeddings,
        pretrained_stats=pretrained_stats,
        num_pos_labels=len(pos_label_names),
        num_chunk_labels=len(chunk_label_names),
    )


def save_preprocessor(
    path: str,
    word_vocab: Vocab,
    char_vocab: Optional[Vocab],
    label2id: Dict[str, int],
    lowercase: bool,
) -> None:
    payload = {
        "word_vocab": word_vocab.to_dict(),
        "char_vocab": char_vocab.to_dict() if char_vocab is not None else None,
        "label2id": label2id,
        "lowercase": lowercase,
    }
    save_json(path, payload)


def load_preprocessor(path: str) -> Dict[str, Any]:
    payload = load_json(path)
    word_vocab = Vocab.from_dict(payload["word_vocab"])
    char_vocab = (
        Vocab.from_dict(payload["char_vocab"]) if payload.get("char_vocab") is not None else None
    )
    label2id = {str(k): int(v) for k, v in payload["label2id"].items()}
    id2label = {idx: label for label, idx in label2id.items()}
    return {
        "word_vocab": word_vocab,
        "char_vocab": char_vocab,
        "label2id": label2id,
        "id2label": id2label,
        "lowercase": bool(payload.get("lowercase", False)),
    }
