import argparse
import csv
import importlib
import json
import math
import platform
import random
import re
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

"""
EN->ZH Transformer 训练/测试脚本（单文件版）。

核心流程：
1) 读取分词后的平行语料与词表（token 已被空格分开）。
2) 构造 Dataset/DataLoader，生成 src、tgt_in、tgt_out 三路张量。
3) 使用标准 Encoder-Decoder Transformer（PyTorch nn.Transformer）训练。
4) 验证阶段用 beam search 计算 BLEU；训练结束后可自动跑 test。
5) 将参数、日志、checkpoint、预测结果、指标统一写入时间戳目录。
"""


def set_seed(seed: int) -> None:
    """固定随机种子，尽量减少实验波动（仍可能受 CUDA 非确定性算子影响）。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_word_vocab(path: Path) -> Tuple[Dict[str, int], List[str]]:
    """
    读取 `token -> id` 词表，并重建 `id -> token` 列表。

    返回：
    - word2idx: Dict[str, int]
    - idx2word: List[str]，下标就是 token id，便于解码阶段反查
    """
    with path.open("r", encoding="utf-8") as f:
        word2idx = json.load(f)

    if not word2idx:
        raise ValueError(f"Empty vocabulary file: {path}")

    vocab_size = max(word2idx.values()) + 1
    idx2word = ["<UNK>"] * vocab_size
    for token, idx in word2idx.items():
        idx2word[idx] = token
    return word2idx, idx2word


def parse_parallel_file(path: Path) -> List[Tuple[List[str], List[str]]]:
    """读取 `src<TAB>tgt` 文件，输出 token 列表对。"""
    pairs: List[Tuple[List[str], List[str]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if "\t" not in line:
                raise ValueError(f"Missing tab separator at {path}:{line_no}")
            src, tgt = line.split("\t", maxsplit=1)
            src_tokens = src.strip().split()
            tgt_tokens = tgt.strip().split()
            if not src_tokens or not tgt_tokens:
                continue
            pairs.append((src_tokens, tgt_tokens))
    return pairs


class TranslationDataset(Dataset):
    """
    将 token 序列转成训练样本（id 序列）。

    对目标端，构造两条序列：
    - tgt_in : [BOS] + y1 y2 ... yn
    - tgt_out: y1 y2 ... yn + [EOS]

    这样 decoder 在时刻 t 输入前缀，预测下一个 token（teacher forcing）。
    """

    def __init__(
        self,
        pairs: Sequence[Tuple[List[str], List[str]]],
        src_vocab: Dict[str, int],
        tgt_vocab: Dict[str, int],
        max_src_len: int,
        max_tgt_len: int,
        word_dropout: float = 0.0,
    ) -> None:
        self.samples = []
        self.word_dropout = word_dropout

        # 特殊 token id 在训练过程中会频繁使用，缓存为成员变量减少查表开销。
        src_unk = src_vocab["<UNK>"]
        src_eos = src_vocab["<EOS>"]
        tgt_unk = tgt_vocab["<UNK>"]
        tgt_bos = tgt_vocab["<BOS>"]
        tgt_eos = tgt_vocab["<EOS>"]
        self._src_unk = src_unk
        self._tgt_unk = tgt_unk

        for src_tokens, tgt_tokens in pairs:
            # 预留一个位置给 EOS，防止长度溢出。
            src_tokens = src_tokens[: max_src_len - 1]
            tgt_tokens = tgt_tokens[: max_tgt_len - 1]

            src_ids = [src_vocab.get(tok, src_unk) for tok in src_tokens] + [src_eos]
            tgt_ids = [tgt_vocab.get(tok, tgt_unk) for tok in tgt_tokens]
            tgt_in_ids = [tgt_bos] + tgt_ids
            tgt_out_ids = tgt_ids + [tgt_eos]

            self.samples.append(
                {
                    "src_ids": src_ids,
                    "tgt_in_ids": tgt_in_ids,
                    "tgt_out_ids": tgt_out_ids,
                    "tgt_tokens": tgt_tokens,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_word_dropout(self, ids: List[int], unk_idx: int) -> List[int]:
        """按概率把 token 替换为 UNK，用于鲁棒性增强。"""
        if self.word_dropout <= 0.0:
            return ids
        return [unk_idx if random.random() < self.word_dropout else tok for tok in ids]

    def _apply_src_noise(self, ids: List[int]) -> List[int]:
        """Source-side noise: word dropout -> token delete -> token swap."""
        if self.word_dropout <= 0.0:
            return ids

        # 1) 先做 UNK 替换，让模型适应词汇缺失。
        dropped = self._apply_word_dropout(ids, self._src_unk)
        # 2) 再做随机删除，模拟输入缺词/漏词。
        result = [tok for tok in dropped if random.random() >= self.word_dropout]
        if not result:
            return ids

        # 3) 最后做相邻交换，模拟局部词序扰动。
        for i in range(len(result) - 1):
            if random.random() < self.word_dropout:
                result[i], result[i + 1] = result[i + 1], result[i]
        return result

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        src_ids = sample["src_ids"]
        tgt_in_ids = sample["tgt_in_ids"]

        if self.word_dropout > 0.0:
            # 源端使用更强噪声（UNK+删除+交换），EOS 不参与扰动。
            src_ids = self._apply_src_noise(src_ids[:-1]) + [src_ids[-1]]
            # 目标端只做 UNK dropout，不做删除/交换，保证监督信号稳定。
            tgt_in_ids = [tgt_in_ids[0]] + self._apply_word_dropout(tgt_in_ids[1:], self._tgt_unk)

        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_in_ids, dtype=torch.long),
            torch.tensor(sample["tgt_out_ids"], dtype=torch.long),
            sample["tgt_tokens"],
        )


class TranslationCollator:
    """Pickle-safe collator for Windows DataLoader workers."""

    def __init__(self, src_pad_idx: int, tgt_pad_idx: int) -> None:
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

    def __call__(self, batch):
        # 这里做“批内动态 padding”，把不同长度句子补齐成矩阵。
        src_seqs, tgt_in_seqs, tgt_out_seqs, tgt_tokens = zip(*batch)
        src_batch = pad_sequence(src_seqs, batch_first=True, padding_value=self.src_pad_idx)
        tgt_in_batch = pad_sequence(tgt_in_seqs, batch_first=True, padding_value=self.tgt_pad_idx)
        tgt_out_batch = pad_sequence(tgt_out_seqs, batch_first=True, padding_value=self.tgt_pad_idx)
        return src_batch, tgt_in_batch, tgt_out_batch, list(tgt_tokens)


class PositionalEncoding(nn.Module):
    """标准正弦位置编码：把“位置信息”加到 token embedding 上。"""

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    """
    标准 Encoder-Decoder Transformer。

    说明：
    - 多头注意力由 `nn.Transformer{Encoder,Decoder}Layer` 内部实现；
    - `nhead` 决定并行注意力头数量，每个头在不同子空间关注不同关系；
    - encoder 负责编码源句上下文，decoder 负责“看源句 + 看历史目标词”生成下一个词。
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        tie_embeddings: bool,
        activation_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        # 下面两层中最关键的是 Multi-Head Attention：
        # - Encoder layer: Self-Attention + FFN
        # - Decoder layer: Masked Self-Attention + Cross-Attention + FFN
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True,
            norm_first=True,
        )
        encoder_norm = nn.LayerNorm(d_model)
        decoder_norm = nn.LayerNorm(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            custom_encoder=nn.TransformerEncoder(
                encoder_layer, num_encoder_layers, encoder_norm, enable_nested_tensor=False,
            ),
            custom_decoder=nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm),
        )

        self.embed_dropout = nn.Dropout(dropout)
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        if tie_embeddings:
            # 输出层与目标 embedding 权重共享：参数更少，通常也更稳。
            self.generator.weight = self.tgt_tok_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        # 乘 sqrt(d_model) 是 Transformer 常见做法，稳定 embedding 数值尺度。
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.d_model))
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

    def decode(
        self,
        tgt_in: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt_in) * math.sqrt(self.d_model))
        return self.transformer.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt_in: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        # 训练时 forward 一次性返回每个位置的词表 logits，供 CE loss 计算。
        memory = self.encode(src, src_key_padding_mask=src_key_padding_mask)
        out = self.decode(
            tgt_in=tgt_in,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.generator(out)


class NoamLR:
    """Transformer 经典学习率调度：warmup 上升 + 之后按步数衰减。"""

    def __init__(self, optimizer: torch.optim.Optimizer, model_size: int, warmup_steps: int, factor: float = 1.0):
        self.optimizer = optimizer
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = 0

    def step(self) -> float:
        self._step += 1
        step = self._step
        lr = self.factor * (self.model_size ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    @property
    def current_step(self) -> int:
        return self._step


def generate_causal_mask(size: int, device: torch.device) -> torch.Tensor:
    # 上三角为 True，表示“未来位置不可见”（decoder 自回归约束）。
    return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)


def ids_to_tokens(ids: List[int], idx2word: List[str], eos_idx: int, pad_idx: int, bos_idx: int) -> List[str]:
    """把预测 id 序列转回 token 序列，并去掉特殊符号。"""
    out = []
    for idx in ids:
        if idx == eos_idx:
            break
        if idx in (pad_idx, bos_idx):
            continue
        if 0 <= idx < len(idx2word):
            out.append(idx2word[idx])
        else:
            out.append("<UNK>")
    return out


def length_norm(score: float, length: int, alpha: float) -> float:
    # GNMT style length normalization.
    return score / (((5 + length) ** alpha) / ((5 + 1) ** alpha))


@torch.inference_mode()
def beam_search_decode(
    model: Seq2SeqTransformer,
    src_ids: List[int],
    device: torch.device,
    src_pad_idx: int,
    tgt_pad_idx: int,
    tgt_bos_idx: int,
    tgt_eos_idx: int,
    beam_size: int,
    max_decode_len: int,
    alpha: float,
    no_repeat_ngram_size: int = 3,
) -> List[int]:
    """
    逐句 beam search 解码。

    beams 元素含义：(tokens, log_prob_sum, finished)。
    每一轮扩展后按长度归一化分数排序，只保留 top-k。
    """
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_key_padding_mask = src_tensor.eq(src_pad_idx)
    memory = model.encode(src_tensor, src_key_padding_mask=src_key_padding_mask)

    beams = [([tgt_bos_idx], 0.0, False)]

    for _ in range(max_decode_len):
        candidates = []
        all_finished = True

        for tokens, score, finished in beams:
            if finished:
                candidates.append((tokens, score, True))
                continue

            all_finished = False
            tgt_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
            tgt_key_padding_mask = tgt_tensor.eq(tgt_pad_idx)
            tgt_mask = generate_causal_mask(tgt_tensor.size(1), device=device)

            decoder_out = model.decode(
                tgt_in=tgt_tensor,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
            logits = model.generator(decoder_out[:, -1, :])  # [1, vocab]
            log_probs = F.log_softmax(logits, dim=-1).squeeze(0)

            # 这些 token 不应该被再次生成为普通词。
            log_probs[tgt_pad_idx] = -1e9
            log_probs[tgt_bos_idx] = -1e9

            if no_repeat_ngram_size > 0 and len(tokens) >= no_repeat_ngram_size:
                # 简单的 n-gram blocking，缓解“重复短语”。
                ngram_prefix_len = no_repeat_ngram_size - 1
                for i in range(len(tokens) - ngram_prefix_len):
                    ngram_prefix = tokens[i : i + ngram_prefix_len]
                    if tokens[-ngram_prefix_len:] == ngram_prefix:
                        banned_token = tokens[i + ngram_prefix_len]
                        log_probs[banned_token] = -1e9

            topk = torch.topk(log_probs, k=beam_size, dim=-1)
            for next_idx, next_logp in zip(topk.indices.tolist(), topk.values.tolist()):
                next_tokens = tokens + [int(next_idx)]
                next_finished = next_idx == tgt_eos_idx
                candidates.append((next_tokens, score + float(next_logp), next_finished))

        if all_finished:
            break

        candidates.sort(
            key=lambda x: length_norm(x[1], max(1, len(x[0]) - 1), alpha),
            reverse=True,
        )
        beams = candidates[:beam_size]

    best_tokens, best_score, _ = max(
        beams,
        key=lambda x: length_norm(x[1], max(1, len(x[0]) - 1), alpha),
    )
    _ = best_score
    return best_tokens[1:]  # remove BOS


def train_one_epoch(
    model: Seq2SeqTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: NoamLR,
    criterion: nn.Module,
    device: torch.device,
    src_pad_idx: int,
    tgt_pad_idx: int,
    grad_clip: float,
    use_amp: bool,
    scaler: torch.amp.GradScaler,
    max_batches: int,
    rdrop_alpha: float,
) -> Tuple[float, float]:
    """
    单个 epoch 训练。

    返回：
    - avg_loss: 按有效 token 加权平均的训练损失
    - running_lr: 该 epoch 最后一个 step 的学习率
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0
    running_lr = 0.0

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kwargs: x  # type: ignore

    iterator = tqdm(loader, desc="train", leave=False)
    for batch_idx, batch in enumerate(iterator, start=1):
        src, tgt_in, tgt_out, _ = batch
        src = src.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)

        src_key_padding_mask = src.eq(src_pad_idx)
        tgt_key_padding_mask = tgt_in.eq(tgt_pad_idx)
        tgt_mask = generate_causal_mask(tgt_in.size(1), device=device)

        optimizer.zero_grad(set_to_none=True)

        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
        )
        with amp_context:
            logits1 = model(
                src=src,
                tgt_in=tgt_in,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_mask=tgt_mask,
            )
            if rdrop_alpha > 0.0:
                # R-Drop：同一 batch 两次前向（不同 dropout mask）。
                logits2 = model(
                    src=src,
                    tgt_in=tgt_in,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    tgt_mask=tgt_mask,
                )
                vocab_size = logits1.size(-1)
                ce_loss = 0.5 * (
                    criterion(logits1.reshape(-1, vocab_size), tgt_out.reshape(-1))
                    + criterion(logits2.reshape(-1, vocab_size), tgt_out.reshape(-1))
                )

                # 对称 KL，只在非 PAD 位置计算一致性约束。
                log_p = F.log_softmax(logits1, dim=-1)
                log_q = F.log_softmax(logits2, dim=-1)
                kl_pq = F.kl_div(log_p, log_q.detach().exp(), reduction="none").sum(dim=-1)
                kl_qp = F.kl_div(log_q, log_p.detach().exp(), reduction="none").sum(dim=-1)
                non_pad_mask = tgt_out.ne(tgt_pad_idx).float()
                denom = non_pad_mask.sum().clamp_min(1.0)
                kl_loss = (0.5 * (kl_pq + kl_qp) * non_pad_mask).sum() / denom
                loss = ce_loss + rdrop_alpha * kl_loss
            else:
                loss = criterion(logits1.reshape(-1, logits1.size(-1)), tgt_out.reshape(-1))

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        running_lr = scheduler.step()

        token_count = int(tgt_out.ne(tgt_pad_idx).sum().item())
        total_tokens += token_count
        total_loss += float(loss.item()) * token_count

        if max_batches > 0 and batch_idx >= max_batches:
            break

    avg_loss = total_loss / max(1, total_tokens)
    return avg_loss, running_lr


@torch.inference_mode()
def evaluate_loss(
    model: Seq2SeqTransformer,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    src_pad_idx: int,
    tgt_pad_idx: int,
    max_batches: int,
) -> float:
    """验证集 loss（不反传、不更新参数）。"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch_idx, batch in enumerate(loader, start=1):
        src, tgt_in, tgt_out, _ = batch
        src = src.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)

        src_key_padding_mask = src.eq(src_pad_idx)
        tgt_key_padding_mask = tgt_in.eq(tgt_pad_idx)
        tgt_mask = generate_causal_mask(tgt_in.size(1), device=device)

        logits = model(
            src=src,
            tgt_in=tgt_in,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=tgt_mask,
        )
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        token_count = int(tgt_out.ne(tgt_pad_idx).sum().item())
        total_loss += float(loss.item()) * token_count
        total_tokens += token_count

        if max_batches > 0 and batch_idx >= max_batches:
            break

    return total_loss / max(1, total_tokens)


@torch.inference_mode()
def evaluate_bleu(
    model: Seq2SeqTransformer,
    dataset: TranslationDataset,
    idx2word_tgt: List[str],
    device: torch.device,
    src_pad_idx: int,
    tgt_pad_idx: int,
    tgt_bos_idx: int,
    tgt_eos_idx: int,
    beam_size: int,
    max_decode_len: int,
    beam_alpha: float,
    max_samples: int,
    no_repeat_ngram_size: int = 3,
) -> Tuple[float, List[str], List[str]]:
    """逐句 beam search，最终用 sacrebleu 计算 corpus BLEU。"""
    try:
        sacrebleu = importlib.import_module("sacrebleu")
    except ImportError as e:
        raise ImportError(
            "Missing dependency `sacrebleu`. Install with: pip install sacrebleu"
        ) from e

    model.eval()
    hypotheses: List[str] = []
    references: List[str] = []

    samples = dataset.samples
    if max_samples > 0:
        samples = samples[:max_samples]

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kwargs: x  # type: ignore

    for sample in tqdm(samples, desc="bleu", leave=False):
        pred_ids = beam_search_decode(
            model=model,
            src_ids=sample["src_ids"],
            device=device,
            src_pad_idx=src_pad_idx,
            tgt_pad_idx=tgt_pad_idx,
            tgt_bos_idx=tgt_bos_idx,
            tgt_eos_idx=tgt_eos_idx,
            beam_size=beam_size,
            max_decode_len=max_decode_len,
            alpha=beam_alpha,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        pred_tokens = ids_to_tokens(
            pred_ids,
            idx2word=idx2word_tgt,
            eos_idx=tgt_eos_idx,
            pad_idx=tgt_pad_idx,
            bos_idx=tgt_bos_idx,
        )

        hypotheses.append(" ".join(pred_tokens))
        references.append(" ".join(sample["tgt_tokens"]))

    bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="none").score if hypotheses else 0.0
    return bleu, hypotheses, references


def write_prediction_files(output_dir: Path, tokenized_preds: List[str], split_name: str) -> Tuple[Path, Path]:
    """
    写两份预测：
    - *.tok.txt: token 之间保留空格
    - *.txt    : 去掉空格，便于直接阅读中文句子
    """
    tok_path = output_dir / f"{split_name}_predictions.tok.txt"
    detok_path = output_dir / f"{split_name}_predictions.txt"

    with tok_path.open("w", encoding="utf-8") as f_tok:
        for line in tokenized_preds:
            f_tok.write(line + "\n")

    with detok_path.open("w", encoding="utf-8") as f_detok:
        for line in tokenized_preds:
            f_detok.write("".join(line.split()) + "\n")

    return tok_path, detok_path


def average_checkpoints(paths: List[Path], device: torch.device) -> Dict[str, torch.Tensor]:
    """Average model state dicts from multiple checkpoints."""
    state_dicts = []
    for p in paths:
        ckpt = torch.load(p, map_location=device, weights_only=True)
        state_dicts.append(ckpt["model_state_dict"])

    avg_state = {}
    for key in state_dicts[0]:
        tensors = [sd[key].float() for sd in state_dicts]
        avg_state[key] = torch.stack(tensors).mean(dim=0)
    return avg_state


def format_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_tag(tag: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", tag.strip())
    return cleaned.strip("_")


def make_run_dir(output_root: Path, mode: str, tag: str) -> Path:
    timestamp = format_timestamp()
    suffix = sanitize_tag(tag)
    run_name = f"{mode}_{timestamp}" if not suffix else f"{mode}_{timestamp}_{suffix}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def append_train_log_csv(csv_path: Path, row: Dict[str, Any]) -> None:
    fieldnames = ["epoch", "train_loss", "val_loss", "val_bleu", "lr", "epoch_seconds"]
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in fieldnames})


def save_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def collect_runtime_info(device: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["gpu_total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
    return info


def load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        yaml = importlib.import_module("yaml")
    except ImportError as e:
        raise ImportError("Missing dependency `pyyaml`. Install with: pip install pyyaml") from e

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        return {}
    if not isinstance(config, dict):
        raise TypeError(f"YAML config must be a dictionary, got: {type(config).__name__}")

    return config


def get_args() -> argparse.Namespace:
    # 设计原则：所有参数都可 CLI 覆盖；若给 --config，先加载 YAML 再被 CLI 覆盖。
    parser = argparse.ArgumentParser(description="Train Transformer for EN->ZH translation")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--config", type=str, default=None, help="YAML config path. CLI args override YAML values.")
    parser.add_argument("--data_dir", type=str, default="cmn-eng-simple")
    parser.add_argument("--output_dir", type=str, default="runs_transformer")
    parser.add_argument("--run_tag", type=str, default="", help="Optional tag added to timestamped run directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_src_len", type=int, default=64)
    parser.add_argument("--max_tgt_len", type=int, default=64)
    parser.add_argument("--max_train_batches", type=int, default=0, help="0 means no limit")
    parser.add_argument("--max_eval_batches", type=int, default=0, help="0 means no limit")
    parser.add_argument("--max_bleu_samples", type=int, default=0, help="0 means no limit")
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint path used in test mode.")

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=4)
    parser.add_argument("--num_decoder_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no_tie_embeddings", action="store_true", help="Disable decoder embedding weight tying")

    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lr_factor", type=float, default=2.0)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.2,
        help="Minimum BLEU improvement to reset early-stopping patience.",
    )
    parser.add_argument(
        "--loss_explosion_ratio",
        type=float,
        default=1.8,
        help="Stop if val_loss > best_val_loss * ratio for several epochs.",
    )
    parser.add_argument(
        "--loss_explosion_patience",
        type=int,
        default=2,
        help="Number of consecutive exploding-loss epochs before stopping.",
    )

    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--beam_alpha", type=float, default=0.6)
    parser.add_argument("--max_decode_len", type=int, default=64)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3, help="Block repeated n-grams during beam search (0=off)")

    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--word_dropout", type=float, default=0.0, help="Word dropout rate for data augmentation (0=off)")
    parser.add_argument("--rdrop_alpha", type=float, default=3.0, help="R-Drop KL loss weight (0=off)")
    parser.add_argument(
        "--run_test_after_train",
        action="store_true",
        help="If set, run full test evaluation after training. Default is train-only.",
    )
    parser.add_argument(
        "--top_k_checkpoints",
        type=int,
        default=5,
        help="Keep top-K best checkpoints for averaging at end of training.",
    )
    parser.add_argument("--test_split", type=str, default="test", choices=["test", "val"])

    pre_args, _ = parser.parse_known_args()
    if pre_args.config:
        config_path = Path(pre_args.config)
        config_dict = load_yaml_config(config_path)
        valid_keys = {action.dest for action in parser._actions}
        unknown_keys = sorted(set(config_dict.keys()) - valid_keys)
        if unknown_keys:
            raise ValueError(
                f"Unknown keys in config {config_path}: {unknown_keys}. "
                "Use argument names from `python train_transformer.py -h`."
            )
        parser.set_defaults(**config_dict)

    return parser.parse_args()


def main() -> None:
    args = get_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    # 每次运行创建独立时间戳目录，避免覆盖历史实验。
    run_dir = make_run_dir(output_root, args.mode, args.run_tag)
    save_json(run_dir / "args.json", vars(args))
    save_json(run_dir / "runtime.json", collect_runtime_info(device))

    use_amp = (not args.no_amp) and device.type == "cuda"
    print(f"Run directory: {run_dir}")
    print(f"Device: {device}, AMP: {use_amp}")

    data_dir = Path(args.data_dir)
    train_path = data_dir / "training.txt"
    val_path = data_dir / "validation.txt"
    test_path = data_dir / "testing.txt"
    src_vocab_path = data_dir / "word2int_en.json"
    tgt_vocab_path = data_dir / "word2int_cn.json"

    src_vocab, _ = load_word_vocab(src_vocab_path)
    tgt_vocab, idx2word_tgt = load_word_vocab(tgt_vocab_path)

    val_pairs = parse_parallel_file(val_path)
    test_pairs = parse_parallel_file(test_path)
    val_set = TranslationDataset(val_pairs, src_vocab, tgt_vocab, args.max_src_len, args.max_tgt_len)
    test_set = TranslationDataset(test_pairs, src_vocab, tgt_vocab, args.max_src_len, args.max_tgt_len)

    src_pad_idx = src_vocab["<PAD>"]
    tgt_pad_idx = tgt_vocab["<PAD>"]
    tgt_bos_idx = tgt_vocab["<BOS>"]
    tgt_eos_idx = tgt_vocab["<EOS>"]

    if args.mode == "test":
        # test 模式：只加载 checkpoint 并评估，不进入训练环节。
        if not args.checkpoint:
            raise ValueError("--checkpoint is required in test mode.")

        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        ckpt_args = checkpoint.get("args", {})
        model_hparams = checkpoint.get("model_hparams", {})
        d_model = int(model_hparams.get("d_model", ckpt_args.get("d_model", args.d_model)))
        nhead = int(model_hparams.get("nhead", ckpt_args.get("nhead", args.nhead)))
        num_encoder_layers = int(
            model_hparams.get("num_encoder_layers", ckpt_args.get("num_encoder_layers", args.num_encoder_layers))
        )
        num_decoder_layers = int(
            model_hparams.get("num_decoder_layers", ckpt_args.get("num_decoder_layers", args.num_decoder_layers))
        )
        dim_feedforward = int(
            model_hparams.get("dim_feedforward", ckpt_args.get("dim_feedforward", args.dim_feedforward))
        )
        dropout = float(model_hparams.get("dropout", ckpt_args.get("dropout", args.dropout)))
        tie_embeddings = bool(model_hparams.get("tie_embeddings", (not ckpt_args.get("no_tie_embeddings", False))))

        model = Seq2SeqTransformer(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            tie_embeddings=tie_embeddings,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        dataset = test_set if args.test_split == "test" else val_set
        bleu, hypotheses, _ = evaluate_bleu(
            model=model,
            dataset=dataset,
            idx2word_tgt=idx2word_tgt,
            device=device,
            src_pad_idx=src_pad_idx,
            tgt_pad_idx=tgt_pad_idx,
            tgt_bos_idx=tgt_bos_idx,
            tgt_eos_idx=tgt_eos_idx,
            beam_size=args.beam_size,
            max_decode_len=args.max_decode_len,
            beam_alpha=args.beam_alpha,
            max_samples=args.max_bleu_samples,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        tok_path, detok_path = write_prediction_files(run_dir, hypotheses, split_name=args.test_split)
        metrics = {
            "mode": "test",
            "checkpoint": str(checkpoint_path.resolve()),
            "split": args.test_split,
            "bleu": bleu,
            "num_predictions": len(hypotheses),
            "beam_size": args.beam_size,
            "beam_alpha": args.beam_alpha,
            "max_decode_len": args.max_decode_len,
        }
        save_json(run_dir / "metrics_test.json", metrics)
        print(f"[Test] split={args.test_split}, BLEU={bleu:.2f}, predictions={len(hypotheses)}")
        print(f"Predictions saved to: {tok_path} and {detok_path}")
        print(f"Metrics saved to: {run_dir / 'metrics_test.json'}")
        return

    train_pairs = parse_parallel_file(train_path)
    train_set = TranslationDataset(train_pairs, src_vocab, tgt_vocab, args.max_src_len, args.max_tgt_len, word_dropout=args.word_dropout)
    collate_fn = TranslationCollator(src_pad_idx=src_pad_idx, tgt_pad_idx=tgt_pad_idx)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    model = Seq2SeqTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        tie_embeddings=(not args.no_tie_embeddings),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0,  # managed by Noam scheduler
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=args.weight_decay,
    )
    scheduler = NoamLR(
        optimizer=optimizer,
        model_size=args.d_model,
        warmup_steps=args.warmup_steps,
        factor=args.lr_factor,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx, label_smoothing=args.label_smoothing)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_bleu = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    patience_count = 0
    loss_explosion_count = 0
    history: List[Dict[str, Any]] = []
    stop_reason = "max_epochs_reached"
    train_log_csv_path = run_dir / "train_log.csv"
    metrics_train_path = run_dir / "metrics_train.json"
    best_ckpt_path = run_dir / "best_model.pt"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    top_k_ckpts: List[Tuple[float, Path]] = []

    print(f"Train/Val/Test sizes: {len(train_set)}/{len(val_set)}/{len(test_set)}")
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, current_lr = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            src_pad_idx=src_pad_idx,
            tgt_pad_idx=tgt_pad_idx,
            grad_clip=args.grad_clip,
            use_amp=use_amp,
            scaler=scaler,
            max_batches=args.max_train_batches,
            rdrop_alpha=args.rdrop_alpha,
        )
        val_loss = evaluate_loss(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            src_pad_idx=src_pad_idx,
            tgt_pad_idx=tgt_pad_idx,
            max_batches=args.max_eval_batches,
        )
        val_bleu, _, _ = evaluate_bleu(
            model=model,
            dataset=val_set,
            idx2word_tgt=idx2word_tgt,
            device=device,
            src_pad_idx=src_pad_idx,
            tgt_pad_idx=tgt_pad_idx,
            tgt_bos_idx=tgt_bos_idx,
            tgt_eos_idx=tgt_eos_idx,
            beam_size=args.beam_size,
            max_decode_len=args.max_decode_len,
            beam_alpha=args.beam_alpha,
            max_samples=args.max_bleu_samples,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )

        epoch_time = time.time() - epoch_start
        log_item = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_bleu": val_bleu,
            "lr": current_lr,
            "epoch_seconds": epoch_time,
        }
        history.append(log_item)
        append_train_log_csv(train_log_csv_path, log_item)

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_bleu={val_bleu:.2f} "
            f"lr={current_lr:.6e} "
            f"time={epoch_time:.1f}s"
        )

        if not (math.isfinite(train_loss) and math.isfinite(val_loss) and math.isfinite(val_bleu)):
            # 防守式停止：出现 NaN/Inf 直接终止，防止写出无意义模型。
            print("Non-finite metric detected (NaN/Inf). Early stopping for safety.")
            stop_reason = "non_finite_metric"
            break

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            loss_explosion_count = 0
        elif val_loss > best_val_loss * args.loss_explosion_ratio:
            # 连续多轮验证损失爆炸则早停，避免浪费训练时间。
            loss_explosion_count += 1
            print(
                f"  -> Warning: val_loss explosion detected "
                f"({val_loss:.4f} > {best_val_loss:.4f} x {args.loss_explosion_ratio:.2f}); "
                f"count={loss_explosion_count}/{args.loss_explosion_patience}"
            )
            if loss_explosion_count >= args.loss_explosion_patience:
                print("Early stopping triggered by exploding validation loss.")
                stop_reason = "loss_explosion"
                break
        else:
            loss_explosion_count = 0

        is_new_best = val_bleu > best_val_bleu
        is_significant_improvement = val_bleu > (best_val_bleu + args.early_stop_min_delta)
        if is_new_best:
            best_val_bleu = val_bleu
            best_epoch = epoch

        ckpt_payload = {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "best_val_bleu": best_val_bleu,
            "best_val_loss": best_val_loss,
            "current_val_bleu": val_bleu,
            "current_val_loss": val_loss,
            "src_vocab_size": len(src_vocab),
            "tgt_vocab_size": len(tgt_vocab),
            "model_hparams": {
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_encoder_layers": args.num_encoder_layers,
                "num_decoder_layers": args.num_decoder_layers,
                "dim_feedforward": args.dim_feedforward,
                "dropout": args.dropout,
                "tie_embeddings": (not args.no_tie_embeddings),
            },
        }

        if is_new_best:
            torch.save(ckpt_payload, best_ckpt_path)
            print(f"  -> New best checkpoint saved to: {best_ckpt_path}")

        epoch_ckpt_path = ckpt_dir / f"epoch_{epoch:03d}_bleu_{val_bleu:.2f}.pt"
        torch.save(ckpt_payload, epoch_ckpt_path)
        top_k_ckpts.append((val_bleu, epoch_ckpt_path))
        top_k_ckpts.sort(key=lambda x: x[0], reverse=True)
        # 只保留 top-k，旧的较差 checkpoint 会被删除。
        while len(top_k_ckpts) > args.top_k_checkpoints:
            _, old_path = top_k_ckpts.pop()
            if old_path.exists():
                old_path.unlink()

        if is_significant_improvement:
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                stop_reason = "early_stopping_patience"
                break

        save_json(
            metrics_train_path,
            {
                "mode": "train",
                "best_epoch": best_epoch,
                "best_val_bleu_during_train": best_val_bleu,
                "best_val_loss_during_train": best_val_loss,
                "epochs_completed": len(history),
                "last_lr": current_lr,
                "stop_reason": "running",
                "history": history,
                "paths": {
                    "best_checkpoint": str(best_ckpt_path),
                    "csv_log": str(train_log_csv_path),
                },
            },
        )

    total_train_time = time.time() - train_start
    print(f"Training finished in {total_train_time / 60.0:.2f} minutes.")

    if not best_ckpt_path.exists():
        raise RuntimeError("Best checkpoint not found. Training may have failed.")

    avg_ckpt_path = run_dir / "avg_model.pt"
    if len(top_k_ckpts) >= 2:
        # checkpoint averaging：常见于翻译任务，可提升泛化稳定性。
        print(f"Averaging top-{len(top_k_ckpts)} checkpoints...")
        avg_state = average_checkpoints([p for _, p in top_k_ckpts], device)
        model.load_state_dict(avg_state)
        torch.save({"model_state_dict": avg_state, "args": vars(args), "model_hparams": {
            "d_model": args.d_model, "nhead": args.nhead,
            "num_encoder_layers": args.num_encoder_layers,
            "num_decoder_layers": args.num_decoder_layers,
            "dim_feedforward": args.dim_feedforward,
            "dropout": args.dropout,
            "tie_embeddings": (not args.no_tie_embeddings),
        }}, avg_ckpt_path)
    else:
        checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    final_val_bleu, _, _ = evaluate_bleu(
        model=model,
        dataset=val_set,
        idx2word_tgt=idx2word_tgt,
        device=device,
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=tgt_pad_idx,
        tgt_bos_idx=tgt_bos_idx,
        tgt_eos_idx=tgt_eos_idx,
        beam_size=args.beam_size,
        max_decode_len=args.max_decode_len,
        beam_alpha=args.beam_alpha,
        max_samples=0,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    test_bleu = None
    test_tok_path = None
    test_detok_path = None
    if args.run_test_after_train:
        test_bleu, test_hypotheses, _ = evaluate_bleu(
            model=model,
            dataset=test_set,
            idx2word_tgt=idx2word_tgt,
            device=device,
            src_pad_idx=src_pad_idx,
            tgt_pad_idx=tgt_pad_idx,
            tgt_bos_idx=tgt_bos_idx,
            tgt_eos_idx=tgt_eos_idx,
            beam_size=args.beam_size,
            max_decode_len=args.max_decode_len,
            beam_alpha=args.beam_alpha,
            max_samples=0,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        test_tok_path, test_detok_path = write_prediction_files(run_dir, test_hypotheses, split_name="test")

    final_metrics = {
        "mode": "train",
        "best_epoch": best_epoch,
        "best_val_bleu_during_train": best_val_bleu,
        "best_val_loss_during_train": best_val_loss,
        "final_val_bleu": final_val_bleu,
        "test_bleu": test_bleu,
        "train_minutes": total_train_time / 60.0,
        "stop_reason": stop_reason,
        "history": history,
        "paths": {
            "best_checkpoint": str(best_ckpt_path),
            "csv_log": str(train_log_csv_path),
            "test_predictions_tok": str(test_tok_path) if test_tok_path else None,
            "test_predictions_detok": str(test_detok_path) if test_detok_path else None,
        },
    }
    save_json(metrics_train_path, final_metrics)

    print(f"Best epoch: {best_epoch}, best val BLEU: {best_val_bleu:.2f}")
    print(f"Final val BLEU: {final_val_bleu:.2f}")
    if test_bleu is None:
        print("Test evaluation skipped (set --run_test_after_train to enable).")
    else:
        print(f"Test BLEU: {test_bleu:.2f}")
        print(f"Predictions saved to: {test_tok_path}")
    print(f"Training metrics saved to: {metrics_train_path}")


if __name__ == "__main__":
    main()
