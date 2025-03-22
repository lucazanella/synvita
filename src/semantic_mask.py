import pylcs
import torch

from src.utils import extract_description, find_starting_index


def generate_shared_semantic_mask(data, neg_data, enc_chunk, tokenizer):
    shared_semantic_mask = torch.ones_like(enc_chunk).long()

    caption = f'"{extract_description(data)}"?'
    neg_caption = f'"{extract_description(neg_data)}"?'
    tokens = tokenizer(caption, add_special_tokens=False)["input_ids"]
    neg_tokens = tokenizer(neg_caption, add_special_tokens=False)["input_ids"]

    tokens_str = " ".join(str(token) for token in tokens)
    neg_tokens_str = " ".join(str(token) for token in neg_tokens)

    caption_start_index = find_starting_index(enc_chunk, torch.tensor(tokens))
    assert caption_start_index != -1, (caption, enc_chunk)

    tokens_mask = [0 if char == " " else 1 for char in tokens_str]

    lcs = pylcs.lcs_sequence_idx(tokens_str, neg_tokens_str)

    # Extract continuous LCS segments
    lcs_segments = [
        lcs[i:j]
        for i, j in zip(
            [0] + [i + 1 for i, x in enumerate(tokens_mask) if x == 0],
            [i for i, x in enumerate(tokens_mask) if x == 0] + [None],
        )
    ]
    # Filter segments where any element is -1
    lcs_mask = [1 if all(x != -1 for x in lcs_segments[i]) else 0 for i in range(len(tokens))]

    # Update attention mask based on LCS mask
    for i, token in enumerate(tokens):
        if caption_start_index + i < len(shared_semantic_mask):
            if lcs_mask[i] == 0:
                shared_semantic_mask[caption_start_index + i] = 0

    return shared_semantic_mask
