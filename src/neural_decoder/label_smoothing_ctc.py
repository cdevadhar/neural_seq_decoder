import torch
from torch import nn

class LabelSmoothingCTCLoss(nn.Module):
    """
    CTC Loss with label smoothing for better generalization.
    
    Args:
        blank: Index of blank token (default: 0)
        smoothing: Smoothing parameter epsilon (default: 0.1)
        reduction: 'mean', 'sum', or 'none'
        zero_infinity: Whether to zero out infinite losses
    """
    def __init__(self, blank=0, smoothing=0.03, reduction='mean', zero_infinity=True):
        super().__init__()
        self.blank = blank
        self.smoothing = smoothing
        self.reduction = reduction
        self.zero_infinity = zero_infinity
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction='none', zero_infinity=zero_infinity)

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        Compute label-smoothed CTC loss.
        
        Args:
            log_probs: Log probabilities (T, N, C) where T=time, N=batch, C=classes
            targets: Target sequences (N, S) where S=target sequence length
            input_lengths: Lengths of input sequences (N,)
            target_lengths: Lengths of target sequences (N,)
        """
        # Standard CTC loss
        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)

        # Add label smoothing by mixing with uniform distribution
        if self.smoothing > 0:
            # Compute entropy of the predictions as regularization
            # Higher entropy = more uniform = less confident
            probs = torch.exp(log_probs)
            entropy = -(probs * log_probs).sum(dim=-1).mean()

            # Mix CTC loss with entropy regularization
            # (1 - smoothing) weight on correct predictions
            # smoothing weight on encouraging uncertainty
            loss = (1 - self.smoothing) * loss - self.smoothing * entropy

        if self.reduction == 'mean':
            # print("taking mean")
            loss = loss/target_lengths.float()
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

print("✓ LabelSmoothingCTCLoss class defined")