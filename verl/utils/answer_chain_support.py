from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BatchAnswerChainSupport:
    token_loss_weights: torch.Tensor
    answer_chain_valid_mask: torch.Tensor


def _compute_response_span(attention_mask: torch.Tensor, response_mask: torch.Tensor) -> tuple[int, int]:
    sequence_width = attention_mask.size(0)
    response_width = response_mask.size(0)
    response_length = int(response_mask.sum().item())
    if int(attention_mask.sum().item()) <= 0:
        raise ValueError("attention_mask does not contain any valid tokens.")
    if response_length <= 0:
        raise ValueError("response_mask does not contain any valid response tokens.")
    if response_width > sequence_width:
        raise ValueError("response width cannot exceed sequence width.")
    if response_length > response_width:
        raise ValueError("response length cannot exceed response width.")

    response_start = sequence_width - response_width
    response_end = response_start + response_length
    return response_start, response_end


def _compute_reasoning_mask(valid_response_mask: torch.Tensor, answer_token_mask: torch.Tensor) -> torch.Tensor:
    answer_indices = torch.nonzero(answer_token_mask > 0, as_tuple=False).flatten()
    reasoning_mask = torch.zeros_like(valid_response_mask, dtype=torch.float32)
    if answer_indices.numel() <= 0:
        return reasoning_mask

    reasoning_end = int(answer_indices[0].item())
    if reasoning_end > 0:
        reasoning_mask[:reasoning_end] = valid_response_mask[:reasoning_end].to(torch.float32)
    return reasoning_mask


def compute_answer_chain_support_from_local_rows(
    predecessor_indices: Sequence[Sequence[torch.Tensor]],
    predecessor_weights: Sequence[Sequence[torch.Tensor]],
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    answer_token_mask: torch.Tensor,
    eps_norm: float = 1e-8,
    tiny_threshold: float = 1e-8,
) -> BatchAnswerChainSupport:
    if attention_mask.dim() != 2 or response_mask.dim() != 2 or answer_token_mask.dim() != 2:
        raise ValueError("attention_mask, response_mask, and answer_token_mask must be rank-2 tensors.")
    if response_mask.shape != answer_token_mask.shape:
        raise ValueError("response_mask and answer_token_mask must have identical shapes.")
    if len(predecessor_indices) != response_mask.size(0) or len(predecessor_weights) != response_mask.size(0):
        raise ValueError("Local predecessor rows must have one entry per batch element.")

    batch_size = attention_mask.size(0)
    response_width = response_mask.size(1)
    device = attention_mask.device
    dtype = torch.float32

    token_loss_weights = torch.zeros((batch_size, response_width), device=device, dtype=dtype)
    answer_chain_valid_mask = torch.zeros(batch_size, device=device, dtype=dtype)

    for batch_index in range(batch_size):
        current_response_mask = response_mask[batch_index].to(torch.bool)
        current_answer_mask = answer_token_mask[batch_index].to(torch.bool) & current_response_mask
        if current_answer_mask.sum().item() <= 0:
            continue

        response_start, response_end = _compute_response_span(attention_mask[batch_index], response_mask[batch_index])
        response_length = response_end - response_start
        if len(predecessor_indices[batch_index]) != response_length or len(predecessor_weights[batch_index]) != response_length:
            raise ValueError("Each batch element must provide one predecessor row per valid response token.")

        valid_answer_mask = current_answer_mask[:response_length].to(dtype)
        token_loss_weights[batch_index, :response_length] = valid_answer_mask

        current_reasoning_mask = _compute_reasoning_mask(current_response_mask, current_answer_mask)
        if current_reasoning_mask[:response_length].sum().item() <= 0:
            continue

        full_support = torch.zeros(attention_mask.size(1), device=device, dtype=dtype)
        full_support[response_start:response_end] = valid_answer_mask / valid_answer_mask.sum().clamp_min(1.0)

        for row_offset in range(response_length - 1, -1, -1):
            query_position = response_start + row_offset
            query_mass = full_support[query_position]
            if query_mass <= 0:
                continue

            row_indices = predecessor_indices[batch_index][row_offset].to(device=device, dtype=torch.long)
            row_weights = predecessor_weights[batch_index][row_offset].to(device=device, dtype=dtype)
            if row_indices.numel() == 0 or row_weights.numel() == 0:
                continue
            if row_indices.shape != row_weights.shape:
                raise ValueError("Each predecessor row must provide matching indices and weights.")

            valid_predecessor_mask = torch.logical_and(row_indices >= 0, row_indices < response_end)
            row_indices = row_indices[valid_predecessor_mask]
            row_weights = row_weights[valid_predecessor_mask]
            if row_indices.numel() == 0:
                continue

            row_weight_sum = row_weights.sum()
            if row_weight_sum <= tiny_threshold:
                continue

            full_support[row_indices] += query_mass * (row_weights / (row_weight_sum + eps_norm))

        raw_reasoning_score = full_support[response_start:response_end] * current_reasoning_mask[:response_length].to(dtype)
        score_sum = raw_reasoning_score.sum()
        if score_sum > tiny_threshold and current_reasoning_mask[:response_length].sum().item() > 0:
            answer_chain_valid_mask[batch_index] = 1.0
            reasoning_weight_scale = current_reasoning_mask[:response_length].sum().clamp_min(1.0).to(dtype)
            token_loss_weights[batch_index, :response_length] += (
                raw_reasoning_score / (score_sum + eps_norm)
            ) * reasoning_weight_scale

    return BatchAnswerChainSupport(
        token_loss_weights=token_loss_weights.detach(),
        answer_chain_valid_mask=answer_chain_valid_mask.detach(),
    )
