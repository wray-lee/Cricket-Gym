import torch
import torch.nn as nn
import torch.nn.functional as F


class BioMoRLoss(nn.Module):
    """
    Multi-task Loss for BioMoR Model (Phase 2).
    """

    def __init__(self, lambda_dir=1.0, lambda_act=0.05):
        super().__init__()
        self.lambda_dir = lambda_dir
        self.lambda_act = lambda_act

        # [CRITICAL] Use BCELoss for probabilities to avoid saturation gradients
        self.cls_loss = nn.BCELoss()

    def forward(self, y_pred, y_gt, router_state):
        # Masking: Only calculate direction loss when an action is actually happening
        action_mask = (y_gt[:, :, 0] + y_gt[:, :, 1]) > 0.01

        # --- Task 1: Policy Classification ---
        # Clamp to prevent NaN in BCELoss
        probs_pred = torch.clamp(y_pred[:, :, :2], 1e-7, 1.0 - 1e-7)
        probs_gt = y_gt[:, :, :2]
        loss_cls = self.cls_loss(probs_pred, probs_gt)

        # --- Task 2: Motor Direction ---
        masked_pred = y_pred[action_mask]
        masked_gt = y_gt[action_mask]

        if masked_pred.shape[0] > 0:
            pred_dir = masked_pred[:, 2:]
            gt_dir = masked_gt[:, 2:]

            # Cosine Similarity: 1.0 means identical, -1.0 means opposite
            # Loss = 1 - CosSim. (Range: 0 to 2)
            cos_sim = F.cosine_similarity(pred_dir, gt_dir, dim=-1)
            loss_dir = (1.0 - cos_sim).mean()
        else:
            loss_dir = torch.tensor(0.0).to(y_pred.device)

        # --- Task 3: Metabolic Cost ---
        loss_act = torch.mean(torch.abs(router_state))

        # Total Loss
        total_loss = (
            loss_cls + (self.lambda_dir * loss_dir) + (self.lambda_act * loss_act)
        )

        return total_loss, {
            "cls": loss_cls.item(),
            "dir": loss_dir.item(),
            "act": loss_act.item(),
        }