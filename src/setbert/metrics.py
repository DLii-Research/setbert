import torch

def relative_abundance_accuracy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the relative abundance accuracy between two relative abundance vectors.

    Args:
        pred: The predicted relative abundance vector.
        target: The target relative abundance vector.

    Returns:
        The relative abundance accuracy.
    """
    device = pred.device
    assert torch.allclose(pred.sum(-1), torch.tensor(1.0, device=device))
    assert torch.allclose(target.sum(-1), torch.tensor(1.0, device=device))
    return torch.sum(torch.clamp(pred, min=None, max=target), -1)
