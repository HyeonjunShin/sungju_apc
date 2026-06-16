import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# 연동 컴포넌트 임포트
from model import MelonDetectorDense
from loss import DenseDetectionLoss
from datasets import MelonDataset, melon_collate_fn

# [수정] 행렬 연산 기반 고속화 mIoU 산출 함수로 대체
def calculate_batch_bbox_miou(pred_logits, pred_boxes, gt_labels, gt_bboxes):
    batch_size = pred_logits.shape[0]
    total_iou = 0.0
    total_matched_objects = 0

    with torch.no_grad():
        for b in range(batch_size):
            bboxes_gt = gt_bboxes[b] 
            num_gts = bboxes_gt.shape[0]
            if num_gts == 0:
                continue

            cls_scores = F.softmax(pred_logits[b], dim=-1)[:, 1] # [4760]
            pred_b = pred_boxes[b] # [4760, 4]

            # cxcywh -> xyxy 변환 연산
            p_x1, p_x2 = pred_b[:, 0] - pred_b[:, 2] / 2, pred_b[:, 0] + pred_b[:, 2] / 2
            p_y1, p_y2 = pred_b[:, 1] - pred_b[:, 3] / 2, pred_b[:, 1] + pred_b[:, 3] / 2
            
            g_x1, g_x2 = bboxes_gt[:, 0] - bboxes_gt[:, 2] / 2, bboxes_gt[:, 0] + bboxes_gt[:, 2] / 2
            g_y1, g_y2 = bboxes_gt[:, 1] - bboxes_gt[:, 3] / 2, bboxes_gt[:, 1] + bboxes_gt[:, 3] / 2

            # Matrix 브로드캐스팅
            inter_x1 = torch.max(p_x1.unsqueeze(1), g_x1.unsqueeze(0))
            inter_y1 = torch.max(p_y1.unsqueeze(1), g_y1.unsqueeze(0))
            inter_x2 = torch.min(p_x2.unsqueeze(1), g_x2.unsqueeze(0))
            inter_y2 = torch.min(p_y2.unsqueeze(1), g_y2.unsqueeze(0))

            inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
            inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
            intersection = inter_w * inter_h

            area_pred = ((p_x2 - p_x1) * (p_y2 - p_y1)).unsqueeze(1)
            area_gt = ((g_x2 - g_x1) * (g_y2 - g_y1)).unsqueeze(0)
            union = area_pred + area_gt - intersection

            ious_matrix = (intersection + 1e-6) / (union + 1e-6) # [4760, Num_gts]
            matching_metric = cls_scores.unsqueeze(1) * ious_matrix 

            best_query_indices = matching_metric.argmax(dim=0) 

            for j in range(num_gts):
                best_idx = best_query_indices[j]
                total_iou += ious_matrix[best_idx, j].item()
                total_matched_objects += 1

    return total_iou, total_matched_objects


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = (640, 360) 
    learning_rate = 1e-3
    batch_size = 64
    num_workers = 8
    print(f"--- 현재 사용 중인 장치: {device} ---")

    train_dataset = MelonDataset("/data/_melon_train", target_size=image_size)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=melon_collate_fn, drop_last=True,
        num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False
    )

    test_dataset = MelonDataset("/data/_melon_test", target_size=image_size)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=melon_collate_fn, drop_last=False,
        num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False
    )

    model = MelonDetectorDense(num_bifpns=1, num_features=64, num_classes=1).to(device)
    criterion = DenseDetectionLoss() 
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    
    num_epochs = 100
    best_test_iou = 0.0  
 
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, "best_melon_bbox_model.pth")
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        best_test_iou = checkpoint["best_iou"]

    print("--- 멜론 2D 격자형 바운딩 박스 검출망 학습 및 mIoU 평가 루프 시작 ---")
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        train_total_iou = 0.0
        train_total_objects = 0

        train_bar = tqdm(train_loader, desc=f"[Epoch {epoch:02d}/{num_epochs:02d} - Train]")

        for batch in train_bar:
            imgs, gt_labels, gt_bboxes = batch
            imgs = imgs.to(device)
            
            gt_labels = [label.to(device) for label in gt_labels]
            gt_bboxes = [bbox.to(device) for bbox in gt_bboxes]

            optimizer.zero_grad()
            pred_logit, pred_boxes = model(imgs)
            
            loss = criterion(pred_logit, pred_boxes, gt_labels, gt_bboxes)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            # [속도 대폭 개선됨]
            batch_iou, batch_objects = calculate_batch_bbox_miou(
                pred_logit, pred_boxes, gt_labels, gt_bboxes
            )
            
            train_loss_sum += loss.item()
            train_batches += 1
            train_total_iou += batch_iou
            train_total_objects += batch_objects

            current_batch_iou = batch_iou / (batch_objects + 1e-8)
            train_bar.set_postfix({
                "Loss": f"{loss.item():.4f}", 
                "BBox_mIoU": f"{current_batch_iou:.4f}"
            })

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        avg_train_loss = train_loss_sum / train_batches
        avg_train_iou = train_total_iou / (train_total_objects + 1e-8)

        # ==========================================
        #         검증 모드 (Test / Eval)
        # ==========================================
        model.eval()
        test_loss_sum = 0.0
        test_batches = 0
        epoch_total_iou = 0.0
        epoch_total_objects = 0

        with torch.no_grad():
            for batch in test_loader:
                imgs, gt_labels, gt_bboxes = batch
                imgs = imgs.to(device)
                
                gt_labels = [label.to(device) for label in gt_labels]
                gt_bboxes = [bbox.to(device) for bbox in gt_bboxes]

                pred_logit, pred_boxes = model(imgs)
                loss = criterion(pred_logit, pred_boxes, gt_labels, gt_bboxes)
                
                test_loss_sum += loss.item()
                test_batches += 1

                batch_iou, batch_objects = calculate_batch_bbox_miou(
                    pred_logit, pred_boxes, gt_labels, gt_bboxes
                )
                epoch_total_iou += batch_iou
                epoch_total_objects += batch_objects

        avg_test_loss = test_loss_sum / test_batches
        avg_test_iou = epoch_total_iou / (epoch_total_objects + 1e-8)

        print("-" * 80)
        print(f"[Epoch {epoch:02d} 완료 결과 요약]")
        print(f"  👉 Learning Rate: {current_lr:.6f}")
        print(f"  👉 Train Loss: {avg_train_loss:.4f} | Train BBox mIoU: {avg_train_iou:.4f}")
        print(f"  👉 Test Loss:  {avg_test_loss:.4f}  | Test BBox mIoU:  {avg_test_iou:.4f}")
        print("-" * 80)

        if avg_test_iou > best_test_iou:
            best_test_iou = avg_test_iou
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_iou": best_test_iou,
                "corresponding_loss": avg_test_loss
            }
            torch.save(checkpoint_data, checkpoint_path)
            print(f"    ⭐ 최고 위치 정밀도(Test BBox mIoU) 달성! 모델 저장 완료 -> 최신 mIoU: {best_test_iou:.4f}\n")

if __name__ == "__main__":
    main()