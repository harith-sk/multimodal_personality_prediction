"""
training/losses.py
Loss functions for OCEAN personality trait regression.

CombinedLoss = alpha * MSE + (1 - alpha) * MAE
  MSE  : penalises large errors heavily
  MAE  : robust to noisy crowdsourced labels
  Default alpha = 0.5 gives equal weight to both.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    """
    MSE + MAE combined loss for regression on noisy labels.

    Args:
        alpha: weight for MSE term. (1-alpha) is weight for MAE term.
               Default 0.5 — equal weighting.
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]"
        self.alpha = alpha

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds:   (B, 5) predicted OCEAN scores in [0, 1]
            targets: (B, 5) ground truth OCEAN scores in [0, 1]
        Returns:
            Scalar loss tensor.
        """
        mse = F.mse_loss(preds, targets)
        mae = F.l1_loss(preds, targets)
        return self.alpha * mse + (1.0 - self.alpha) * mae

    def __repr__(self):
        return f"CombinedLoss(alpha={self.alpha})"


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch
    criterion = CombinedLoss(alpha=0.5)
    preds   = torch.rand(32, 5)
    targets = torch.rand(32, 5)
    loss = criterion(preds, targets)
    assert loss.item() > 0
    print(f"Loss value : {loss.item():.6f}")
    print(f"Loss repr  : {criterion}")
    print("✅ losses.py OK")