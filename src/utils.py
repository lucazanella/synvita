import re

import torch


def extract_description(conversation):
    pattern = r'Does this video entail the description: "(.*)"\?'
    match = re.search(pattern, conversation)
    if match:
        return match.group(1)
    else:
        raise ValueError("Description not found")


def find_starting_index(enc_chunk, tokens):
    for i in range(len(enc_chunk) - len(tokens) + 1):
        if torch.all(enc_chunk[i : i + len(tokens)] == tokens):
            return i
    return -1
