import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import complete_box_iou_loss

class DenseDetectionLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, lambda_box=2.0, lambda_cls=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_box = lambda_box  
        self.lambda_cls = lambda_cls   

    def forward(self, pred_logits, pred_boxes, gt_labels, gt_bboxes):
        """
        pred_logits: [B, 4760, 2] (배경=0, 참외/멜론=1)
        pred_boxes : [B, 4760, 4] ([cx, cy, bw, bh] 정규화 비율)
        gt_labels  : 리스트 [B]
        gt_bboxes  : 리스트 [B]
        """
        # 입력 텐서 자체에 NaN이 들어왔는지 1차 방어
        if torch.isnan(pred_logits).any() or torch.isnan(pred_boxes).any():
            return torch.tensor(0.0, device=pred_logits.device, requires_grad=True)

        batch_size, num_queries, _ = pred_logits.shape
        device = pred_logits.device

        max_gts = max([len(gt) for gt in gt_bboxes])
        if max_gts == 0: 
            return (pred_logits.sum() * 0.0) + (pred_boxes.sum() * 0.0)

        with torch.no_grad():
            padded_gt_boxes = torch.zeros((batch_size, max_gts, 4), device=device)
            gt_mask = torch.zeros((batch_size, max_gts), dtype=torch.bool, device=device)
            
            for b in range(batch_size):
                num_gts = gt_bboxes[b].shape[0]
                if num_gts > 0:
                    padded_gt_boxes[b, :num_gts] = gt_bboxes[b].to(device)
                    gt_mask[b, :num_gts] = True

            pred_cxcy_detached = pred_boxes[..., :2].detach()
            gt_cxcy = padded_gt_boxes[..., :2]  
            
            dist_matrix = torch.norm(gt_cxcy.unsqueeze(2) - pred_cxcy_detached.unsqueeze(1), dim=-1)
            dist_matrix = dist_matrix.masked_fill(~gt_mask.unsqueeze(-1), float('inf'))

            vals, topk_indices = torch.topk(dist_matrix, k=5, dim=2, largest=False)

            flat_vals = vals.flatten()          
            flat_indices = topk_indices.flatten()  
            
            b_indices = torch.arange(batch_size, device=device).view(batch_size, 1, 1).repeat(1, max_gts, 5).flatten()
            gt_indices = torch.arange(max_gts, device=device).view(1, max_gts, 1).repeat(batch_size, 1, 5).flatten()

            valid_match = flat_vals < float('inf')
            flat_vals = flat_vals[valid_match]
            flat_indices = flat_indices[valid_match]
            b_indices = b_indices[valid_match]
            gt_indices = gt_indices[valid_match]

            sort_idx_desc = torch.argsort(flat_vals, descending=True)
            b_desc = b_indices[sort_idx_desc]
            f_desc = flat_indices[sort_idx_desc]
            g_desc = gt_indices[sort_idx_desc]

            target_labels = torch.zeros((batch_size, num_queries), dtype=torch.long, device=device)
            target_boxes = torch.zeros((batch_size, num_queries, 4), dtype=torch.float32, device=device)
            pos_mask = torch.zeros((batch_size, num_queries), dtype=torch.bool, device=device)

            if b_desc.numel() > 0:
                target_labels[b_desc, f_desc] = 1
                target_boxes[b_desc, f_desc] = padded_gt_boxes[b_desc, g_desc]
                pos_mask[b_desc, f_desc] = True

        num_pos = max(pos_mask.sum().item(), 1.0)

        # ==========================================
        # 1. 💡 Focal Loss 수치 안정성 극대화 최적화
        # ==========================================
        # log_softmax의 안정적인 계산을 위해 안정화 장치 확보
        log_prob = F.log_softmax(pred_logits, dim=-1) 
        # 수치 언더플로우를 방지하기 위해 1e-7 ~ 1.0 사이로 클램핑된 확률 계산
        pred_prob = torch.clamp(torch.exp(log_prob), min=1e-7, max=1.0)
        
        target_labels_onehot = F.one_hot(target_labels, num_classes=2).float()
        ce_loss = -target_labels_onehot * log_prob
        
        alpha_factor = target_labels_onehot[..., 1] * self.alpha + target_labels_onehot[..., 0] * (1.0 - self.alpha)
        alpha_factor = alpha_factor.unsqueeze(-1) 
        
        focal_loss = alpha_factor * ((1.0 - pred_prob) ** self.gamma) * ce_loss
        loss_cls = focal_loss.sum() / num_pos

        # ==========================================
        # 2. 💡 위치 회귀 손실 계산 (CIoU Loss 및 크기 붕괴 방어)
        # ==========================================
        loss_box = torch.tensor(0.0, device=device)
        if pos_mask.sum() > 0:
            matched_pred_boxes = pred_boxes[pos_mask]   
            matched_target_boxes = target_boxes[pos_mask] 

            def cxcywh_to_xyxy(box):
                # 💡 폭(w)과 높이(h)가 0 이하가 되어 나눗셈 NaN이 나는 것을 원천 방어
                cx = box[:, 0]
                cy = box[:, 1]
                w = torch.clamp(box[:, 2], min=1e-4)
                h = torch.clamp(box[:, 3], min=1e-4)
                
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                return torch.stack([x1, y1, x2, y2], dim=1)

            p_xyxy = cxcywh_to_xyxy(matched_pred_boxes)
            t_xyxy = cxcywh_to_xyxy(matched_target_boxes)

            p_xyxy = torch.clamp(p_xyxy, min=0.001, max=0.999)
            t_xyxy = torch.clamp(t_xyxy, min=0.0, max=1.0)

            loss_box_sum = complete_box_iou_loss(p_xyxy, t_xyxy, reduction="sum")
            loss_box = loss_box_sum / num_pos

        # 💡 [핵심 방어] 만약 연산 중 어떤 이유로든 NaN이 발생했다면, 해당 배치의 손실을 0으로 밀어버려 가중치 오염 방지
        if torch.isnan(loss_box) or torch.isinf(loss_box):
            loss_box = torch.tensor(0.0, device=device)
        if torch.isnan(loss_cls) or torch.isinf(loss_cls):
            loss_cls = torch.tensor(0.0, device=device)

        total_loss = self.lambda_cls * loss_cls + self.lambda_box * loss_box
        return total_loss
