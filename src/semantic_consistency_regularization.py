from typing import Dict

import numpy as np
import torch

from src.alignment import compute_alignment
from src.domain import Domain


def triplet_loss(
    alignment_orig_pos,
    alignment_orig_sh_sem,
    alignment_contr_sh_sem,
    alignment_contr_pos,
    margin=0.2,
):
    """Computes the triplet loss to promote better alignment of the positive pair w.r.t. the
    generic pair and the generic pair w.r.t. the negative pair.

    Parameters:
    - alignment_orig_pos: Alignments for pairs (V^z, t^z).
    - alignment_orig_sh_sem: Alignments for pairs (V^z, t').
    - alignment_contr_sh_sem: Alignments for pairs (V^\bar{z}, t').
    - alignment_contr_pos: Alignments for pairs (V^\bar{z}, t^z).
    - margin: Float, margin

    Returns:
    - loss: The computed triplet loss, promoting better alignment of the positive pair w.r.t. the generic pair
            and the generic pair w.r.t. the negative pair.
    """
    # Promotes the better alignment of the positive pair f(V^z, t^z) w.r.t the generic pair (V^z, t')
    loss_pos = torch.relu(margin + alignment_orig_sh_sem - alignment_orig_pos)
    # Promotes the better alignment of the generic pair f(V^\bar{z}, t') w.r.t the negative pair f(V^\bar{z}, t^z)
    loss_neg = torch.relu(margin + alignment_contr_pos - alignment_contr_sh_sem)

    return loss_pos.mean() + loss_neg.mean()


def compute_semantic_consistency_regularization(
    logits: torch.Tensor,
    labels: torch.Tensor,
    shared_semantic_indices: np.ndarray,
    inputs: Dict[str, torch.Tensor],
    dynamic_weights: torch.Tensor,
    sem_cons_margin: float,
    tokenizer,
) -> torch.Tensor:
    """Compute semantic consistency regularization.

    Args:
        logits (torch.Tensor): Model logits.
        labels (torch.Tensor): Ground truth labels.
        shared_semantic_indices (np.ndarray): Indices of shared semantic texts.
        inputs (dict): Input dictionary.
        dynamic_weights (torch.Tensor): Weights for synthetic videos.
        sem_cons_margin (float): Margin for the semantic consistency regularization.
        tokenizer: Tokenizer object.

    Returns:
        torch.Tensor: The computed semantic consistency regularization.
    """
    if shared_semantic_indices.size == 0:
        return torch.tensor(0.0, device=logits.device)

    indices = np.sort(np.concatenate((shared_semantic_indices - 1, shared_semantic_indices)))

    alignments = compute_alignment(logits[indices], labels[indices], tokenizer)

    real_indices = (np.array(inputs["domains"]) == Domain.real).nonzero()[0]
    real_sh_sem_indices = np.intersect1d(real_indices, shared_semantic_indices)

    synth_indices = (np.array(inputs["domains"]) == Domain.synthetic).nonzero()[0]
    synth_sh_sem_indices = np.intersect1d(synth_indices, shared_semantic_indices)

    token_id_yes = tokenizer.encode("Yes", add_special_tokens=False)[0]
    token_id_no = tokenizer.encode("No", add_special_tokens=False)[0]

    semantic_consistency_regularization = torch.tensor(0.0, device=logits.device)
    count = 0

    assert real_sh_sem_indices.size == synth_sh_sem_indices.size

    for idx, real_sh_sem_index in enumerate(real_sh_sem_indices):
        synth_sh_sem_index = synth_sh_sem_indices[idx]

        alignment_real_sh_sem_index = np.where(indices == real_sh_sem_index)[0][0]
        alignment_synth_sh_sem_index = np.where(indices == synth_sh_sem_index)[0][0]

        if token_id_yes in labels[real_sh_sem_index - 1]:
            # (V^r, t^r) and (V^s, t^r)
            assert token_id_no in labels[synth_sh_sem_index - 1]

            alignment_real_video_real_text = alignments[
                alignment_real_sh_sem_index - 1
            ]  # f(V^r, t^r)
            alignment_real_video_sh_sem_text = alignments[
                alignment_real_sh_sem_index
            ]  # f(V^r, t')
            alignment_synth_video_real_text = alignments[
                alignment_synth_sh_sem_index - 1
            ]  # f(V^s, t^r)
            alignment_synth_video_sh_sem_text = alignments[
                alignment_synth_sh_sem_index
            ]  # f(V^s, t')

            semantic_consistency_regularization += dynamic_weights[synth_sh_sem_index] * (
                triplet_loss(
                    alignment_real_video_real_text,
                    alignment_real_video_sh_sem_text,
                    alignment_synth_video_sh_sem_text,
                    alignment_synth_video_real_text,
                    margin=sem_cons_margin,
                )
            )
            count += 1

        else:
            # (V^r, t^s) and (V^s, t^s)
            assert token_id_yes in labels[synth_sh_sem_index - 1]

            alignment_real_video_synth_text = alignments[
                alignment_real_sh_sem_index - 1
            ]  # f(V^r, t^s)
            alignment_real_video_sh_sem_text = alignments[
                alignment_real_sh_sem_index
            ]  # f(V^r, t')
            alignment_synth_video_synth_text = alignments[
                alignment_synth_sh_sem_index - 1
            ]  # f(V^s, t^s)
            alignment_synth_video_sh_sem_text = alignments[
                alignment_synth_sh_sem_index
            ]  # f(V^s, t')

            semantic_consistency_regularization += dynamic_weights[synth_sh_sem_index] * (
                triplet_loss(
                    alignment_synth_video_synth_text,
                    alignment_synth_video_sh_sem_text,
                    alignment_real_video_sh_sem_text,
                    alignment_real_video_synth_text,
                    margin=sem_cons_margin,
                )
            )
            count += 1

    return (
        semantic_consistency_regularization / count
        if count > 0
        else semantic_consistency_regularization
    )
