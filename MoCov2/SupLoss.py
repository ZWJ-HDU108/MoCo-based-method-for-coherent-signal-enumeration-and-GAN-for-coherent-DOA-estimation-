import torch
import torch.nn as nn


class MoCoSupConLoss(nn.Module):
    """Asymmetric Supervised Contrastive Loss"""

    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, all_labels):
        """
        logits: (B, B+Q)  B: batch size, B+Q: batch keys + queue keys
        labels: (B, )
        all_labels: (B+Q, )
        """

        # log-sum-exp trick
        logits_max, _ = logits.max(dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # mask          (B, ) -> (B, 1)      (B, ) -> (1, B+Q)       finally broadcast to (B, B+Q)
        mask = torch.eq(labels.unsqueeze(1), all_labels.unsqueeze(0)).float()

        # Eliminate invalid entries in the queue that have not yet been filled
        valid_mask = (all_labels != -1).unsqueeze(0).float()  # (1, B+Q)
        mask = mask * valid_mask  # element-wise

        exp_logits = torch.exp(logits) * valid_mask  # The denominator also excludes invalid entries
        #       log(exp(s_ij)) - log(sum(exp(s_in)))   (B, B+Q) -> (B, 1)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)  # 1e-12 to prevent log(0)

        mask_pos_count = mask.sum(dim=1)  # calculate number of positive
        # mask_pos_count = torch.clamp(mask_pos_count, min=1.0)  # If a certain query happens to have no positive samples (an extreme case), set it to 1 to avoid division by zero
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_pos_count  # For each query, take the mean of the log_prob of all its positive samples

        loss = -mean_log_prob_pos.mean()  # calculate mean in every batch
        return loss