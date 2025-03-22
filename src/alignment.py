import torch


def compute_alignment(logits, labels, tokenizer):
    """Compute alignment scores for each batch element.

    Args:
        logits (torch.Tensor): Logits tensor of shape (batch_size, seq_len, vocab_size).
        labels (torch.Tensor): Labels tensor of shape (batch_size, seq_len).
        tokenizer (Tokenizer): Tokenizer object to encode tokens.

    Returns:
        torch.Tensor: Alignment scores for each batch element.
    """
    probs = torch.nn.functional.softmax(logits, dim=-1)

    token_id_yes = tokenizer.encode("Yes", add_special_tokens=False)[0]
    token_id_no = tokenizer.encode("No", add_special_tokens=False)[0]

    label_indices = []
    for label in labels:
        if token_id_yes in label:
            label_indices.append((label == token_id_yes).nonzero(as_tuple=True)[0])
        elif token_id_no in label:
            label_indices.append((label == token_id_no).nonzero(as_tuple=True)[0])
        else:
            raise ValueError("No ('Yes' or 'No') found in the labels.")
    label_indices = torch.cat(label_indices)

    batch_indices = torch.arange(probs.size(0), device=probs.device)
    selected_probs = probs[batch_indices, label_indices - 1]

    probs_yes = selected_probs[:, token_id_yes]
    probs_no = selected_probs[:, token_id_no]

    alignments = probs_yes / (probs_yes + probs_no)

    return alignments
