# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement Actor
"""

import math
import os
from collections import defaultdict
from typing import Any, Optional

import torch
import torch.distributed as dist
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...protocol import DataProto, batch_collate
from ...trainer.core_algos import average_loss, build_effective_token_loss_weights, compute_kl, compute_policy_loss
from ...utils.answer_chain_support import compute_answer_chain_support_from_local_rows
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig


try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self._answer_chain_hidden_state_cache: Optional[torch.Tensor] = None
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

    def _clear_answer_chain_hidden_state_cache(self) -> None:
        self._answer_chain_hidden_state_cache = None

    def _get_answer_chain_window_size(self) -> int:
        return int(getattr(self.config, "answer_chain_local_window_size", 64))

    def _resolve_answer_chain_layer_indices(self, num_layers: int) -> tuple[int, ...]:
        configured_layers = getattr(self.config, "answer_chain_selected_layers", (-1,))
        if isinstance(configured_layers, int):
            configured_layers = (configured_layers,)
        if len(configured_layers) <= 0:
            raise ValueError("answer_chain_selected_layers must not be empty.")

        resolved_layers = []
        for layer_index in configured_layers:
            normalized_index = int(layer_index)
            if normalized_index < 0:
                normalized_index += num_layers
            if normalized_index < 0 or normalized_index >= num_layers:
                raise ValueError(
                    f"answer_chain_selected_layers contains invalid layer index {layer_index} for {num_layers} layers."
                )
            if normalized_index not in resolved_layers:
                resolved_layers.append(normalized_index)
        return tuple(resolved_layers)

    def _get_decoder_layers(self):
        candidate_modules = [self.actor_module, getattr(self.actor_module, "model", None)]
        for candidate in candidate_modules:
            if candidate is None:
                continue

            language_model = getattr(candidate, "language_model", None)
            if language_model is not None and hasattr(language_model, "layers"):
                return language_model.layers

            if hasattr(candidate, "layers"):
                return candidate.layers

            nested_model = getattr(candidate, "model", None)
            if nested_model is not None and hasattr(nested_model, "layers"):
                return nested_model.layers

        raise RuntimeError("Unable to locate decoder layers for answer-chain routing.")

    def _compute_response_span(self, attention_mask: torch.Tensor, response_mask: torch.Tensor) -> tuple[int, int]:
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

    def _extract_selected_hidden_states(
        self,
        outputs: Any,
        batch_size: int,
        seqlen: int,
        indices: Optional[torch.Tensor] = None,
        pad_size: int = 0,
        detach_to_cpu: bool = True,
    ) -> torch.Tensor:
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("Hidden-state caching requested but the forward output did not include hidden_states.")

        decoder_layers = self._get_decoder_layers()
        layer_indices = self._resolve_answer_chain_layer_indices(len(decoder_layers))
        cache_dtype = torch.bfloat16
        cached_hidden_states = []

        for layer_index in layer_indices:
            hidden_state = hidden_states[layer_index]
            if self.config.padding_free:
                if indices is None:
                    raise RuntimeError("Padding-free hidden-state caching requires unpadding indices.")
                if self.config.ulysses_size > 1:
                    hidden_state = gather_outputs_and_unpad(
                        hidden_state,
                        gather_dim=1,
                        unpad_dim=1,
                        padding_size=pad_size,
                    )
                hidden_state = hidden_state.squeeze(0)
                hidden_state = pad_input(hidden_states=hidden_state, indices=indices, batch=batch_size, seqlen=seqlen)
            hidden_state = hidden_state.to(dtype=cache_dtype)
            if detach_to_cpu:
                hidden_state = hidden_state.detach().cpu()
            cached_hidden_states.append(hidden_state)

        return torch.stack(cached_hidden_states, dim=1)

    def _project_selected_query_key_states(self, selected_hidden_states: torch.Tensor):
        decoder_layers = self._get_decoder_layers()
        layer_indices = self._resolve_answer_chain_layer_indices(len(decoder_layers))
        if selected_hidden_states.size(1) != len(layer_indices):
            raise RuntimeError("Cached hidden-state count does not match configured answer-chain layer count.")

        projected_layers = []
        for cached_layer_index, layer_index in enumerate(layer_indices):
            decoder_layer = decoder_layers[layer_index]
            attention_module = decoder_layer.self_attn
            layer_hidden_states = selected_hidden_states[:, cached_layer_index]
            with FSDP.summon_full_params(decoder_layer, writeback=False, recurse=True):
                if hasattr(decoder_layer, "input_layernorm"):
                    layer_hidden_states = decoder_layer.input_layernorm(layer_hidden_states)
                query_states = attention_module.q_proj(layer_hidden_states)
                key_states = attention_module.k_proj(layer_hidden_states)

            num_heads = getattr(attention_module, "num_heads", None)
            if num_heads is None:
                raise RuntimeError("Answer-chain routing requires self_attn.num_heads.")

            head_dim = query_states.size(-1) // num_heads
            num_key_value_heads = getattr(attention_module, "num_key_value_heads", num_heads)
            query_states = query_states.view(query_states.size(0), query_states.size(1), num_heads, head_dim).permute(0, 2, 1, 3)
            key_states = key_states.view(
                key_states.size(0), key_states.size(1), num_key_value_heads, head_dim
            ).permute(0, 2, 1, 3)

            if num_key_value_heads != num_heads:
                if num_heads % num_key_value_heads != 0:
                    raise RuntimeError("Answer-chain routing requires num_heads to be divisible by num_key_value_heads.")
                key_states = key_states.repeat_interleave(num_heads // num_key_value_heads, dim=1)

            projected_layers.append((query_states, key_states, head_dim))

        return projected_layers

    def _build_local_predecessor_rows(
        self,
        projected_layers,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]]]:
        predecessor_indices: list[list[torch.Tensor]] = []
        predecessor_weights: list[list[torch.Tensor]] = []
        window_size = self._get_answer_chain_window_size()
        device = attention_mask.device

        for batch_index in range(attention_mask.size(0)):
            response_start, response_end = self._compute_response_span(
                attention_mask[batch_index], response_mask[batch_index]
            )
            L = response_end - response_start
            prompt_indices = torch.nonzero(
                attention_mask[batch_index, :response_start].to(torch.bool),
                as_tuple=False,
            ).flatten().to(device=device, dtype=torch.long)
            P = prompt_indices.numel()
            W = window_size

            if L == 0:
                predecessor_indices.append([])
                predecessor_weights.append([])
                continue

            # Causal local window offsets (0-indexed within response)
            r = torch.arange(L, device=device, dtype=torch.long)           # (L,)
            k = torch.arange(W, device=device, dtype=torch.long)           # (W,)
            w_off = r.unsqueeze(1) - W + k.unsqueeze(0)                    # (L, W)
            w_valid = (w_off >= 0) & (w_off < r.unsqueeze(1))              # (L, W)
            w_safe = w_off.clamp(0, L - 1)                                 # (L, W) safe gather index

            # Absolute predecessor indices and validity mask (L, P+W)
            if P > 0:
                padded_abs = torch.cat([
                    prompt_indices.unsqueeze(0).expand(L, -1),             # (L, P)
                    response_start + w_off,                                 # (L, W) — may be negative, OK for output only
                ], dim=1)
                padded_valid = torch.cat([
                    torch.ones(L, P, device=device, dtype=torch.bool),
                    w_valid,
                ], dim=1)                                                   # (L, P+W)
            else:
                padded_abs = response_start + w_off                        # (L, W)
                padded_valid = w_valid                                      # (L, W)

            # Accumulate attention probabilities over layers — two batched ops per layer
            combined_probs = None
            routing_matmul_dtype = torch.bfloat16
            for query_states, key_states, head_dim in projected_layers:
                H = query_states.size(1)
                scale = 1.0 / math.sqrt(head_dim)

                # Queries for all response positions: (L, H, D)
                q = query_states[batch_index, :, response_start:response_end, :].permute(1, 0, 2).to(routing_matmul_dtype)

                # Prompt attention: (L, H, P)
                if P > 0:
                    p_k = key_states[batch_index, :, prompt_indices, :].to(routing_matmul_dtype)  # (H, P, D)
                    p_logits = torch.matmul(
                        q.permute(1, 0, 2),         # (H, L, D)
                        p_k.transpose(-1, -2),       # (H, D, P)
                    ).permute(1, 0, 2) * scale       # (L, H, P)
                else:
                    p_logits = q.new_empty(L, H, 0)

                # Window attention — gather local keys once, one batched matmul: (L, H, W)
                k_resp = key_states[batch_index, :, response_start:response_end, :].to(routing_matmul_dtype)  # (H, L, D)
                k_win = k_resp[:, w_safe.reshape(-1), :].view(H, L, W, head_dim).permute(1, 0, 2, 3)  # (L, H, W, D)
                w_logits = torch.matmul(
                    q.unsqueeze(2),                 # (L, H, 1, D)
                    k_win.transpose(-1, -2),         # (L, H, D, W)
                ).squeeze(2) * scale                 # (L, H, W)

                # Merge, mask padding, softmax, average heads: (L, P+W)
                logits = torch.cat([p_logits, w_logits], dim=2).to(torch.float32)
                logits.masked_fill_(~padded_valid.unsqueeze(1), float("-inf"))
                layer_probs = torch.softmax(logits, dim=-1).mean(dim=1).nan_to_num(0.0)  # (L, P+W)
                combined_probs = layer_probs if combined_probs is None else combined_probs + layer_probs

            combined_probs = combined_probs / len(projected_layers)        # (L, P+W)

            # Unpack padded tensors into variable-length lists (pure indexing, no GPU compute)
            sample_indices: list[torch.Tensor] = []
            sample_weights: list[torch.Tensor] = []
            for i in range(L):
                valid = padded_valid[i]
                sample_indices.append(padded_abs[i][valid])
                sample_weights.append(combined_probs[i][valid].to(torch.float32))

            predecessor_indices.append(sample_indices)
            predecessor_weights.append(sample_weights)

        return predecessor_indices, predecessor_weights

    def _compute_answer_chain_support_from_cached_hidden_states(
        self,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
        answer_token_mask: torch.Tensor,
        selected_hidden_states: torch.Tensor,
    ):
        projected_layers = self._project_selected_query_key_states(selected_hidden_states)
        predecessor_indices, predecessor_weights = self._build_local_predecessor_rows(
            projected_layers=projected_layers,
            attention_mask=attention_mask,
            response_mask=response_mask,
        )
        return compute_answer_chain_support_from_local_rows(
            predecessor_indices=predecessor_indices,
            predecessor_weights=predecessor_weights,
            attention_mask=attention_mask,
            response_mask=response_mask,
            answer_token_mask=answer_token_mask,
        )

    def _forward_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        cache_selected_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            log_probs: # (bs, response_len)
            cached_hidden_states: optional tensor cached on CPU for answer-chain routing
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}

        cached_hidden_states = None

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # (total_nnz, 1)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
            pad_size = 0

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
                output_hidden_states=cache_selected_hidden_states,
                return_dict=True,
            )  # prevent model thinks we are generating
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            # gather log_prob if sp > 1
            if self.config.ulysses_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
            if cache_selected_hidden_states:
                cached_hidden_states = self._extract_selected_hidden_states(
                    outputs=output,
                    batch_size=batch_size,
                    seqlen=seqlen,
                    indices=indices,
                    pad_size=pad_size,
                    detach_to_cpu=True,
                )
        else:
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
                output_hidden_states=cache_selected_hidden_states,
                return_dict=True,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)
            if cache_selected_hidden_states:
                cached_hidden_states = self._extract_selected_hidden_states(
                    outputs=output,
                    batch_size=batch_size,
                    seqlen=seqlen,
                    detach_to_cpu=True,
                )

        return log_probs, cached_hidden_states

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()
        self._clear_answer_chain_hidden_state_cache()

        temperature = data.meta_info["temperature"]
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses"]
        non_tensor_select_keys = ["multi_modal_inputs"]

        data = data.select(select_keys, non_tensor_select_keys)
        if self.config.dynamic_batching:
            max_token_len = self.config.micro_batch_size_per_device_for_experience * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)
            batch_idx_list = None

        log_probs_lst = []
        cached_hidden_state_batches = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs, cached_hidden_states = self._forward_micro_batch(
                model_inputs,
                temperature=temperature,
                cache_selected_hidden_states=True,
            )
            log_probs_lst.append(log_probs)
            if cached_hidden_states is not None:
                cached_hidden_state_batches.append(cached_hidden_states)

        log_probs = torch.concat(log_probs_lst, dim=0)
        if cached_hidden_state_batches:
            cached_hidden_states = torch.concat(cached_hidden_state_batches, dim=0)
        else:
            cached_hidden_states = None

        if self.config.dynamic_batching:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if cached_hidden_states is not None:
                cached_hidden_states = restore_dynamic_batch(cached_hidden_states, batch_idx_list)

        self._answer_chain_hidden_state_cache = cached_hidden_states

        return log_probs

    @torch.no_grad()
    def compute_answer_chain_weights(self, data: DataProto) -> DataProto:
        self.actor_module.eval()
        if self._answer_chain_hidden_state_cache is None:
            raise RuntimeError("Answer-chain hidden-state cache is empty. compute_log_prob must run before answer-chain routing.")

        select_keys = ["input_ids", "attention_mask", "response_mask", "answer_token_mask"]
        non_tensor_select_keys = []

        data = data.select(select_keys, non_tensor_select_keys)
        if len(self._answer_chain_hidden_state_cache) != len(data):
            raise RuntimeError("Cached answer-chain hidden states do not match the current batch size.")

        micro_batch_size = max(int(self.config.micro_batch_size_per_device_for_experience), 1)
        micro_batches = [data[start : start + micro_batch_size] for start in range(0, len(data), micro_batch_size)]

        weight_batches = []
        valid_mask_batches = []

        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute answer-chain weights", position=1)

        batch_offset = 0
        try:
            for micro_batch in micro_batches:
                model_inputs = micro_batch.batch
                micro_batch_size = len(micro_batch)
                selected_hidden_states = self._answer_chain_hidden_state_cache[
                    batch_offset : batch_offset + micro_batch_size
                ].to(device=model_inputs["input_ids"].device)
                batch_offset += micro_batch_size

                support = self._compute_answer_chain_support_from_cached_hidden_states(
                    attention_mask=model_inputs["attention_mask"],
                    response_mask=model_inputs["response_mask"],
                    answer_token_mask=model_inputs["answer_token_mask"],
                    selected_hidden_states=selected_hidden_states,
                )
                weight_batches.append(support.token_loss_weights)
                valid_mask_batches.append(support.answer_chain_valid_mask)
        finally:
            self._clear_answer_chain_hidden_state_cache()

        token_loss_weights = torch.concat(weight_batches, dim=0)
        answer_chain_valid_mask = torch.concat(valid_mask_batches, dim=0)

        return DataProto.from_dict(
            tensors={
                "token_loss_weights": token_loss_weights,
                "answer_chain_valid_mask": answer_chain_valid_mask,
            }
        )

    def update_policy(self, data: DataProto) -> dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses", "response_mask"]
        select_keys.extend(["old_log_probs", "ref_log_probs", "advantages"])
        optional_select_keys = [
            "token_loss_weights",
            "answer_chain_valid_mask",
        ]
        select_keys.extend([key for key in optional_select_keys if key in data.batch])
        non_tensor_select_keys = ["multi_modal_inputs"]

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

            for mini_batch in mini_batches:
                total_response_tokens = torch.sum(mini_batch.batch["response_mask"])
                dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)

                if self.config.dynamic_batching:
                    max_input_len = mini_batch.batch["input_ids"].size(-1)
                    max_token_len = self.config.micro_batch_size_per_device_for_update * max_input_len
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=2)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    # all return: (bsz, response_length)
                    log_probs, _ = self._forward_micro_batch(model_inputs, temperature=temperature)

                    pg_loss, pg_metrics = compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        token_loss_weights=model_inputs.get("token_loss_weights"),
                        sequence_weight_mask=model_inputs.get("answer_chain_valid_mask"),
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                        tau_positive=self.config.tau_positive,
                        tau_negative=self.config.tau_negative,
                        loss_type=self.config.loss_type,
                        loss_avg_mode=self.config.loss_avg_mode,
                        token_loss_weight_clip_min=getattr(self.config, "reasoning_loss_weight_clip_min", None),
                        token_loss_weight_clip_max=getattr(self.config, "reasoning_loss_weight_clip_max", None),
                    )

                    if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        kld = compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        kl_loss = average_loss(kld, response_mask, mode=self.config.loss_avg_mode)
                        loss = pg_loss + kl_loss * self.config.kl_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_coef
                    else:
                        loss = pg_loss

                    loss = loss * torch.sum(response_mask) * self.world_size / total_response_tokens
                    loss.backward()

                    batch_metrics = {f"actor/{k}": v for k, v in pg_metrics.items()}
                    batch_metrics["actor/pg_loss"] = pg_loss.detach().item()
                    append_to_dict(metrics, batch_metrics)

                    token_loss_weights = model_inputs.get("token_loss_weights")
                    if token_loss_weights is not None:
                        token_loss_weights = build_effective_token_loss_weights(
                            token_loss_weights=token_loss_weights,
                            response_mask=response_mask,
                            sequence_weight_mask=model_inputs.get("answer_chain_valid_mask"),
                            clip_min=getattr(self.config, "reasoning_loss_weight_clip_min", None),
                            clip_max=getattr(self.config, "reasoning_loss_weight_clip_max", None),
                            dtype=torch.float32,
                        )
                        response_mask_float = response_mask.to(torch.float32)
                        masked_weights = token_loss_weights * response_mask_float
                        valid_weights = token_loss_weights[response_mask_float > 0]
                        weight_mean = valid_weights.mean()
                        weight_std = valid_weights.std() if valid_weights.numel() > 1 else valid_weights.new_zeros(())
                        reasoning_weight_metrics = {
                            "actor/reasoning_loss_weight_mean": weight_mean.detach().item(),
                            "actor/reasoning_loss_weight_std": weight_std.detach().item(),
                            "actor/reasoning_loss_weight_nonzero_frac": VF.masked_mean((token_loss_weights > 0).to(torch.float32), response_mask_float).detach().item(),
                            "actor/reasoning_loss_weight_seq_sum_mean": masked_weights.sum(dim=-1).mean().detach().item(),
                        }
                        answer_chain_valid_mask = model_inputs.get("answer_chain_valid_mask")
                        if answer_chain_valid_mask is not None:
                            reasoning_weight_metrics["actor/reasoning_loss_weight_valid_seq_frac"] = answer_chain_valid_mask.to(torch.float32).mean().detach().item()
                        append_to_dict(metrics, reasoning_weight_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
