from typing import Dict, List, Optional

import torch
import torch.nn as nn


class CRF(nn.Module):
    """Linear-chain Conditional Random Field."""

    def __init__(self, num_tags: int, constraints: Optional[Dict[str, torch.Tensor]] = None) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.constraint_penalty = -10000.0

        if constraints is None:
            allowed_start = torch.ones(num_tags, dtype=torch.bool)
            allowed_end = torch.ones(num_tags, dtype=torch.bool)
            allowed_transitions = torch.ones((num_tags, num_tags), dtype=torch.bool)
        else:
            allowed_start = constraints.get("allowed_start")
            allowed_end = constraints.get("allowed_end")
            allowed_transitions = constraints.get("allowed_transitions")
            if allowed_start is None or allowed_end is None or allowed_transitions is None:
                raise ValueError("constraints must contain allowed_start / allowed_end / allowed_transitions")
            if tuple(allowed_transitions.shape) != (num_tags, num_tags):
                raise ValueError("allowed_transitions shape mismatch")
            if tuple(allowed_start.shape) != (num_tags,) or tuple(allowed_end.shape) != (num_tags,):
                raise ValueError("allowed_start/end shape mismatch")

        self.register_buffer("allowed_start", allowed_start.bool())
        self.register_buffer("allowed_end", allowed_end.bool())
        self.register_buffer("allowed_transitions", allowed_transitions.bool())
        self.reset_parameters()

    def _constrained_start(self) -> torch.Tensor:
        return self.start_transitions.masked_fill(~self.allowed_start, self.constraint_penalty)

    def _constrained_end(self) -> torch.Tensor:
        return self.end_transitions.masked_fill(~self.allowed_end, self.constraint_penalty)

    def _constrained_transitions(self) -> torch.Tensor:
        return self.transitions.masked_fill(~self.allowed_transitions, self.constraint_penalty)

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def neg_log_likelihood(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if emissions.dim() != 3:
            raise ValueError("emissions should have shape [batch, seq_len, num_tags]")
        if tags.dim() != 2:
            raise ValueError("tags should have shape [batch, seq_len]")
        if mask.dim() != 2:
            raise ValueError("mask should have shape [batch, seq_len]")

        log_numerator = self._compute_score(emissions, tags, mask)
        log_denominator = self._compute_normalizer(emissions, mask)
        nll = log_denominator - log_numerator
        return nll.mean()

    def _compute_score(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, _ = emissions.shape
        mask = mask.bool()
        start_transitions = self._constrained_start()
        transitions = self._constrained_transitions()
        end_transitions = self._constrained_end()

        first_tags = tags[:, 0]
        batch_indices = torch.arange(batch_size, device=emissions.device)
        score = start_transitions[first_tags]
        score += emissions[batch_indices, 0, first_tags]

        for t in range(1, seq_len):
            prev_tags = tags[:, t - 1]
            curr_tags = tags[:, t]
            transition_score = transitions[prev_tags, curr_tags]
            emission_score = emissions[batch_indices, t, curr_tags]
            score += (transition_score + emission_score) * mask[:, t]

        seq_ends = mask.long().sum(dim=1) - 1
        last_tags = tags[batch_indices, seq_ends]
        score += end_transitions[last_tags]
        return score

    def _compute_normalizer(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = emissions.shape
        mask = mask.bool()
        start_transitions = self._constrained_start()
        transitions = self._constrained_transitions()
        end_transitions = self._constrained_end()

        score = start_transitions + emissions[:, 0]

        for t in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_trans = transitions.unsqueeze(0)
            broadcast_emit = emissions[:, t].unsqueeze(1)
            next_score = torch.logsumexp(
                broadcast_score + broadcast_trans + broadcast_emit, dim=1
            )
            score = torch.where(mask[:, t].unsqueeze(1), next_score, score)

        score += end_transitions
        return torch.logsumexp(score, dim=1)

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        if emissions.dim() != 3:
            raise ValueError("emissions should have shape [batch, seq_len, num_tags]")
        if mask.dim() != 2:
            raise ValueError("mask should have shape [batch, seq_len]")

        mask = mask.bool()
        batch_size, seq_len, num_tags = emissions.shape
        if num_tags != self.num_tags:
            raise ValueError("num_tags mismatch between emissions and CRF")

        start_transitions = self._constrained_start()
        transitions = self._constrained_transitions()
        end_transitions = self._constrained_end()

        score = start_transitions + emissions[:, 0]
        history = []

        for t in range(1, seq_len):
            candidate_score = score.unsqueeze(2) + transitions.unsqueeze(0)
            best_score, best_path = candidate_score.max(dim=1)
            best_score = best_score + emissions[:, t]
            score = torch.where(mask[:, t].unsqueeze(1), best_score, score)
            history.append(best_path)

        score += end_transitions
        _, best_last_tags = score.max(dim=1)
        seq_ends = mask.long().sum(dim=1) - 1

        best_sequences: List[List[int]] = []
        for i in range(batch_size):
            seq_len_i = int(seq_ends[i].item()) + 1
            best_tag = int(best_last_tags[i].item())
            best_path = [best_tag]

            for hist in reversed(history[: seq_len_i - 1]):
                best_tag = int(hist[i][best_tag].item())
                best_path.append(best_tag)

            best_path.reverse()
            best_sequences.append(best_path)

        return best_sequences
