import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# =====================================================================
# MODEL COMPONENT (수치 안정화 및 상대 좌표 가이드 내장 버전)
# =====================================================================

class DepthwiseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))
    
class BiFPNBlock(nn.Module):
    def __init__(self, feature_size=64, epsilon=0.0001):
        super().__init__()
        self.epsilon = epsilon

        self.w4_td = nn.Parameter(torch.ones(2, dtype=torch.float32)) 
        self.w3_out = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.w4_out = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.w5_out = nn.Parameter(torch.ones(2, dtype=torch.float32))

        self.p4_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p3_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p4_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p5_out = DepthwiseConvBlock(feature_size, feature_size)
        
    def forward(self, inputs):
        p3_in, p4_in, p5_in = inputs

        # --- Top-Down: P5 -> P4 ---
        w4_td = F.relu(self.w4_td)
        w4_td = w4_td / (torch.sum(w4_td) + self.epsilon)
        p5_upsampled = F.interpolate(p5_in, size=p4_in.shape[2:], mode='nearest')
        p4_td = self.p4_td(w4_td[0] * p4_in + w4_td[1] * p5_upsampled)

        # --- Top-Down: P4 -> P3 ---
        w3_out = F.relu(self.w3_out)
        w3_out = w3_out / (torch.sum(w3_out) + self.epsilon)
        p4_td_upsampled = F.interpolate(p4_td, size=p3_in.shape[2:], mode='nearest')
        p3_out = self.p3_out(w3_out[0] * p3_in + w3_out[1] * p4_td_upsampled)

        # --- Bottom-Up: P3 -> P4 ---
        w4_out = F.relu(self.w4_out)
        w4_out = w4_out / (torch.sum(w4_out) + self.epsilon)
        p3_out_down = F.max_pool2d(p3_out, kernel_size=3, stride=2, padding=1)
        if p3_out_down.shape[2:] != p4_in.shape[2:]:
            p3_out_down = F.interpolate(p3_out_down, size=p4_in.shape[2:], mode='nearest')
        p4_out = self.p4_out(w4_out[0] * p4_in + w4_out[1] * p4_td + w4_out[2] * p3_out_down)

        # --- Bottom-Up: P4 -> P5 ---
        w5_out = F.relu(self.w5_out)
        w5_out = w5_out / (torch.sum(w5_out) + self.epsilon)
        p4_out_down = F.max_pool2d(p4_out, kernel_size=3, stride=2, padding=1)
        if p4_out_down.shape[2:] != p5_in.shape[2:]:
            p4_out_down = F.interpolate(p4_out_down, size=p5_in.shape[2:], mode='nearest')
        p5_out = self.p5_out(w5_out[0] * p5_in + w5_out[1] * p4_out_down)

        return [p3_out, p4_out, p5_out]

class Backbone(nn.Module):
    def __init__(self, out_features=64):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.p3 = nn.Conv2d(128, out_features, kernel_size=1) 
        self.p4 = nn.Conv2d(256, out_features, kernel_size=1) 
        self.p5 = nn.Conv2d(512, out_features, kernel_size=1) 

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        c2 = self.resnet.layer1(x)   
        c3 = self.resnet.layer2(c2)  
        c4 = self.resnet.layer3(c3)  
        c5 = self.resnet.layer4(c4)  

        return (self.p3(c3), self.p4(c4), self.p5(c5))

class ScaleSpecificHead(nn.Module):
    def __init__(self, in_channels, num_classes=1, stride=8):
        super().__init__()
        self.stride = stride
        
        # 분류 브랜치
        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes + 1, kernel_size=1)
        )
        # 2D BBox 회귀 브랜치
        self.box_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4, kernel_size=1)
        )

    def forward(self, x):
        logits = self.cls_conv(x)
        box_reg = self.box_conv(x) 
        
        b, _, h, w = logits.shape
        device = x.device
        
        # 1. 격자 세포 고유의 물리적 중심점 Prior 생성
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij"
        )
        grid_x = (grid_x + 0.5) / w
        grid_y = (grid_y + 0.5) / h
        
        # [1, H*W, 2] 차원으로 배치 동기화 브로드캐스팅 완벽 고정
        grid_centers = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 2)

        # 차원 평탄화 및 순서 정렬 [B, H*W, Channels]
        logits = logits.flatten(2).permute(0, 2, 1)
        box_reg = box_reg.flatten(2).permute(0, 2, 1)

        # 2. 💡 [nan 차단 핵심 수정] 중심점이 절대로 격자 범위를 벗어나지 못하도록 유도
        # 기존 grid_centers에서 격자 1칸 크기(1/w, 1/h) 내에서만 미세 움직임 오프셋을 가지도록 제약하되,
        # 전체 범위가 이미지 경계(0.001 ~ 0.999) 내에 있도록 하드 클램핑을 즉시 먹입니다.
        cx = grid_centers[..., 0] + (torch.tanh(box_reg[:, :, 0]) * (0.5 / w))
        cy = grid_centers[..., 1] + (torch.tanh(box_reg[:, :, 1]) * (0.5 / h))
        
        cx = torch.clamp(cx, min=0.001, max=0.999)
        cy = torch.clamp(cy, min=0.001, max=0.999)
        
        # 3. 💡 [크기 발산 및 나눗셈 0 방어] 
        # exp() 방식 대신 안전한 sigmoid 축소 방식을 쓰되, 최소 픽셀 크기(1e-4)를 확보하여
        # 폭과 높이가 0이 되어 분모 붕괴 nan이 나는 것을 구조적으로 불가능하게 만듭니다.
        bw = torch.sigmoid(box_reg[:, :, 2]) * (16.0 / w)
        bh = torch.sigmoid(box_reg[:, :, 3]) * (16.0 / h)
        
        bw = torch.clamp(bw, min=1e-4, max=0.999)
        bh = torch.clamp(bh, min=1e-4, max=0.999)

        bboxes = torch.stack([cx, cy, bw, bh], dim=-1)
        
        # 4. 💡 [최종 방어 벨트] 혹시 모를 순방향 연산 과정의 무한대(Inf)나 결측치 원천 소멸
        if torch.isnan(bboxes).any() or torch.isinf(bboxes).any():
            bboxes = torch.where(torch.isnan(bboxes) | torch.isinf(bboxes), torch.full_like(bboxes, 0.5), bboxes)
        
        return logits, bboxes



class PredictionHeadDense(nn.Module):
    def __init__(self, in_channels=64, num_classes=1):
        super().__init__()
        # P3(Stride 8), P4(Stride 16), P5(Stride 32) 가이드 스케일 부여
        self.head_p3 = ScaleSpecificHead(in_channels, num_classes, stride=8)
        self.head_p4 = ScaleSpecificHead(in_channels, num_classes, stride=16)
        self.head_p5 = ScaleSpecificHead(in_channels, num_classes, stride=32)

    def forward(self, features):
        p3, p4, p5 = features

        logits_p3, boxes_p3 = self.head_p3(p3)
        logits_p4, boxes_p4 = self.head_p4(p4)
        logits_p5, boxes_p5 = self.head_p5(p5)

        all_logits = torch.cat([logits_p3, logits_p4, logits_p5], dim=1)
        all_boxes = torch.cat([boxes_p3, boxes_p4, boxes_p5], dim=1)

        return all_logits, all_boxes

class MelonDetectorDense(nn.Module):
    def __init__(self, num_bifpns=2, num_features=64, num_classes=1):
        super().__init__()
        self.backbone = Backbone(num_features)
        self.bifpns = nn.ModuleList([BiFPNBlock(num_features) for _ in range(num_bifpns)])
        self.prediction = PredictionHeadDense(in_channels=num_features, num_classes=num_classes)

    def forward(self, x):
        features = self.backbone(x)
        for bifpn in self.bifpns:
            features = bifpn(features)

        pred_logits, pred_boxes = self.prediction(features)
        return pred_logits, pred_boxes

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    height, width = 360, 640
    dummy_imgs = torch.randn(batch_size, 3, height, width).to(device)

    detector = MelonDetectorDense(num_features=64, num_classes=1).to(device)
    pred_logits, pred_boxes = detector(dummy_imgs)
    
    print("\n--- [완료] 안정화 아키텍처 모델 스크리닝 성공 ---")
    print(f"입력 데이터 Shape : {dummy_imgs.shape}")
    print(f"출력 Logits Shape : {pred_logits.shape}")  # [16, 4760, 2]
    print(f"출력 Boxes  Shape : {pred_boxes.shape}")   # [16, 4760, 4]