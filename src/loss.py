import torch
import torch.nn as nn
import torch.nn.functional as F


class BioMoRLoss(nn.Module):
    """
    Multi-task Loss for BioMoR Model (Phase 2).
    Components:
    1. Classification Loss (MSE/BCE) -> for Run/Jump Probabilities
    2. Directional Loss (Cosine) -> for Motor Heading (Cos/Sin)
    3. Metabolic Loss (L1) -> for Sparse LAL Activity
    """

    def __init__(self, lambda_dir=1.0, lambda_act=0.05):
        super().__init__()
        self.lambda_dir = lambda_dir
        self.lambda_act = lambda_act

        # use BCEWithLogitsLoss instead of MSELoss
        self.pos_weight = torch.tensor([5.0])  # improve miss punishment
        self.cls_loss = None

    def forward(self, y_pred, y_gt, router_state):
        """
        Args:
            y_pred: (Batch, Seq, 4) -> [P_run, P_jump, Cos, Sin]
            y_gt:   (Batch, Seq, 4)
            router_state: (Batch, Seq, Hidden)
        """

        device = y_pred.device
        if self.cls_loss is None or self.cls_loss.pos_weight.device != device:
            self.cls_loss = nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight.to(device)
            )

        # Masking: Only calculate loss when an action is actually happening
        # Ground Truth action intensity: sum of P_run + P_jump
        # If GT is all zeros, it means "no action required", we generally don't penalize direction there
        # but we do penalize Policy (should be 0).
        # 过滤掉没有动作的时间步
        # todo: 后续根据具体运动调整
        action_mask = (y_gt[:, :, 0] + y_gt[:, :, 1]) > 0.01

        # --- Task 1: Policy Classification (P_run, P_jump) ---
        # Calculate on all time steps to force silence when needed

        loss_cls = self.cls_loss(y_pred[:, :, :2], y_gt[:, :, :2])

        # --- Task 2: Motor Direction (Cos, Sin) ---
        # Only calculate direction loss when there IS an action
        # Pred vectors

        # Avoid dimision not mapping with `y_pred[action_mask, 2:]`
        masked_pred = y_pred[action_mask]
        masked_gt = y_gt[action_mask]

        pred_dir = masked_pred[:, 2:]
        gt_dir = masked_gt[:, 2:]

        if pred_dir.shape[0] > 0:
            # Cosine Embedding Loss wants flag 1 for "similar"
            # Or manually: 1 - cosine_similarity
            # We use manual calculation for explicit control
            cos_sim = F.cosine_similarity(pred_dir, gt_dir, dim=-1)
            loss_dir = (1.0 - cos_sim).mean()
        else:
            loss_dir = torch.tensor(0.0).to(device)

        # --- Task 3: Metabolic Cost (LAL Activity) ---
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
