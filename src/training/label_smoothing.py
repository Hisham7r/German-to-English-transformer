import torch
import torch.nn as nn


def compare_losses(vocab_size, target_idx, logits, pad_id=None):
    """
    Compares CrossEntropyLoss with and without label smoothing,
    on a single toy example, to see the numeric difference.
    """
    target = torch.tensor([target_idx])
    logits = logits.unsqueeze(0)  # add batch dimension: [1, vocab_size]

    criterion_hard = nn.CrossEntropyLoss()
    criterion_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)

    loss_hard = criterion_hard(logits, target)
    loss_smooth = criterion_smooth(logits, target)

    print(f"  hard-target loss:    {loss_hard.item():.4f}")
    print(f"  label-smoothed loss: {loss_smooth.item():.4f}")


# ── Test-Block ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    vocab_size = 5
    target_idx = 2

    # Case 1: model is already very confident about the correct word
    confident_logits = torch.tensor([1.0, 2.0, 8.0, 1.5, 0.5])
    print("Case 1 — model is already very confident:")
    compare_losses(vocab_size, target_idx, confident_logits)

    print()

    # Case 2: model is unsure, several words look plausible
    unsure_logits = torch.tensor([1.0, 2.0, 3.0, 2.5, 1.5])
    print("Case 2 — model is unsure:")
    compare_losses(vocab_size, target_idx, unsure_logits)
