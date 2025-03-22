from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.domain import Domain


@dataclass
class CausalLMOutputWithPastAndLabels(CausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    labels: torch.LongTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


def forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    domains: Optional[torch.LongTensor] = None,
    shared_semantic_indices: Optional[torch.LongTensor] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

    >>> prompt = "Hey, are you consciours? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
    ```"""

    output_attentions = (
        output_attentions if output_attentions is not None else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()

        if domains is not None:
            loss_fct = CrossEntropyLoss(reduction="none")

            shared_semantic_indices = shared_semantic_indices.cpu().numpy()

            loss_real = None
            real_indices = (np.array(domains) == Domain.real).nonzero()[0]
            real_indices = np.setdiff1d(real_indices, shared_semantic_indices)
            if real_indices.size:
                # Select only real samples
                real_logits = logits[real_indices]
                real_labels = labels[real_indices]
                # Shift so that tokens < n predict n
                shift_logits_real = real_logits[..., :-1, :].contiguous()
                shift_labels_real = real_labels[..., 1:].contiguous()
                # Flatten the tokens
                batch_size_real = shift_logits_real.size(0)
                shift_logits_real = shift_logits_real.view(-1, self.config.vocab_size)
                shift_labels_real = shift_labels_real.view(-1)
                # Enable model parallelism
                shift_labels_real = shift_labels_real.to(shift_logits_real.device)
                loss_real = loss_fct(shift_logits_real, shift_labels_real)
                loss_real = loss_real.view(batch_size_real, -1)

                mask_real = (shift_labels_real != -100).view(batch_size_real, -1)
                masked_loss_real = loss_real * mask_real.float()

            loss_synthetic = None
            synthetic_indices = (np.array(domains) == Domain.synthetic).nonzero()[0]
            synthetic_indices = np.setdiff1d(synthetic_indices, shared_semantic_indices)
            if synthetic_indices.size:
                # Select only synthetic samples
                synthetic_logits = logits[synthetic_indices]
                synthetic_labels = labels[synthetic_indices]
                # Shift so that tokens < n predict n
                shift_logits_synthetic = synthetic_logits[..., :-1, :].contiguous()
                shift_labels_synthetic = synthetic_labels[..., 1:].contiguous()
                # Flatten the tokens
                batch_size_synthetic = shift_logits_synthetic.size(0)
                shift_logits_synthetic = shift_logits_synthetic.view(-1, self.config.vocab_size)
                shift_labels_synthetic = shift_labels_synthetic.view(-1)
                # Enable model parallelism
                shift_labels_synthetic = shift_labels_synthetic.to(shift_logits_synthetic.device)
                loss_synthetic = loss_fct(shift_logits_synthetic, shift_labels_synthetic)
                loss_synthetic = loss_synthetic.view(batch_size_synthetic, -1)

                mask_synthetic = (shift_labels_synthetic != -100).view(batch_size_synthetic, -1)
                masked_loss_synthetic = loss_synthetic * mask_synthetic.float()

            total_valid_tokens = mask_real.sum()
            if synthetic_indices.size:
                total_valid_tokens += mask_synthetic.sum()

            loss_real = masked_loss_real.sum(dim=1) / total_valid_tokens
            if synthetic_indices.size:
                loss_synthetic = masked_loss_synthetic.sum(dim=1) / total_valid_tokens
            loss = (loss_real, loss_synthetic)

        else:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPastAndLabels(
        loss=loss,
        logits=logits,
        labels=labels,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
