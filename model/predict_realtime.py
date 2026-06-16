import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np

# 기존 프로젝트의 모델 구조 임포트
from model import MelonDetectorDense  
# 카메라 라이브러리
from camera.gemini336 import Gemini336Camera
from torchvision.ops import nms


def preprocess_image(color_image, device):
    """
    카메라 프레임(HWC, BGR)을 모델 입력 형식(1CHW, RGB, 0~1 Scale)으로 변환
    """
    image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    
    # 학습 규격인 640x360 해상도로 리사이즈하여 4760개 격자 차원 동기화
    image_resized = cv2.resize(image_rgb, (640, 360), interpolation=cv2.INTER_LINEAR)
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    return image_tensor


def visualize_inference(color_image, pred_logits, pred_boxes, conf_threshold=0.55, iou_threshold=0.45):
    """
    💡 바운딩 박스 드로잉, 중심점(Center X, Y) 추출 및 시각화를 동시에 수행하는 함수
    """
    orig_h, orig_w, _ = color_image.shape
    vis_image = color_image.copy()

    logits = pred_logits[0]   
    boxes = pred_boxes[0]   

    scores = F.softmax(logits, dim=-1)
    class_scores = scores[:, 1]  

    # 1. 신뢰도 기준 1차 필터링
    keep = class_scores > conf_threshold
    if not keep.any():
        return vis_image  

    filtered_scores = class_scores[keep]
    filtered_boxes = boxes[keep]
    filtered_indices = torch.where(keep)[0] 

    # 2. [cx, cy, bw, bh] -> NMS용 [x1, y1, x2, y2] 변환
    x1 = filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2
    y1 = filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2
    x2 = filtered_boxes[:, 0] + filtered_boxes[:, 2] / 2
    y2 = filtered_boxes[:, 1] + filtered_boxes[:, 3] / 2
    boxes_tlbr = torch.stack([x1, y1, x2, y2], dim=1)

    # 3. NMS 구동
    nms_keep_indices = nms(boxes_tlbr, filtered_scores, iou_threshold=iou_threshold)

    if nms_keep_indices.numel() == 0:
        return vis_image

    final_scores = filtered_scores[nms_keep_indices].cpu().numpy()
    final_boxes_tlbr = boxes_tlbr[nms_keep_indices].cpu().numpy()
    final_query_ids = filtered_indices[nms_keep_indices].cpu().numpy()

    # 4. 카메라 원본 해상도 원복 및 드로잉
    for score, box, q_id in zip(final_scores, final_boxes_tlbr, final_query_ids):
        
        np.random.seed(int(q_id))
        color_id = np.random.randint(0, 256, size=3).tolist()

        xmin = int(np.clip(box[0] * orig_w, 0, orig_w))
        ymin = int(np.clip(box[1] * orig_h, 0, orig_h))
        xmax = int(np.clip(box[2] * orig_w, 0, orig_w))
        ymax = int(np.clip(box[3] * orig_h, 0, orig_h))

        # A. 바운딩 박스 드로잉
        cv2.rectangle(vis_image, (xmin, ymin), (xmax, ymax), color_id, 2, cv2.LINE_AA)

        # 💡 B. [중심점 계산 및 시각화]
        # 원본 해상도 픽셀 스케일 기준의 중심점(cx, cy) 산출
        cx_pixel = int((xmin + xmax) / 2)
        cy_pixel = int((ymin + ymax) / 2)

        # 중심점에 작은 채워진 원(Circle)과 조준선(Crosshair) 그리기
        cv2.circle(vis_image, (cx_pixel, cy_pixel), 5, color_id, -1, cv2.LINE_AA)
        cv2.circle(vis_image, (cx_pixel, cy_pixel), 6, (255, 255, 255), 1, cv2.LINE_AA) # 테두리 강조
        
        # C. 텍스트 정보 가독성 보정 라벨링 (상자 상단에 ID, Score 및 중심 좌표 표기)
        text = f"Melon #{q_id} [{score:.2f}]"
        coord_text = f"X:{cx_pixel}, Y:{cy_pixel}"
        
        text_y = ymin - 25 if ymin - 25 > 20 else ymin + 20
        
        # 메인 정보 표기
        cv2.putText(vis_image, text, (xmin, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis_image, text, (xmin, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_id, 1, cv2.LINE_AA)
        
        # 픽셀 좌표 정보 표기
        cv2.putText(vis_image, coord_text, (xmin, text_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis_image, coord_text, (xmin, text_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return vis_image


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 가속화 장치 상태: {device} ---")

    model_path = "./best_melon_bbox_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ 에러: 가중치 파일이 존재하지 않습니다: {model_path}")
        return

    print("-> AI 모델 로드 중...")
    model = MelonDetectorDense(num_bifpns=1, num_features=64, num_classes=1)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()  
    print("-> 모델 준비 완료.")

    # 카메라 인스턴스 셋업
    camera = Gemini336Camera()
    camera.set_color_profile(1280, 720, 30)
    camera.set_depth_profile(848, 480, 30, hw_align=False) # 필요 시 카메라 래퍼 규격에 따라 알맞게 하드웨어 정렬 조정
    camera.set_camera_properties(30, 1, 1)
    camera.start()

    print("--- 멜론 실시간 AI 검출 및 Depth 동시 투영 시작 ('q'를 누르면 종료) ---")
    while True:
        frame = camera.get_frames()
        if frame is None:
            continue

        color_image = frame.get_color()  # 원본 비디오 프레임 (1280x720, BGR)
        depth_image = frame.get_depth()  # 원본 댑스 프레임 (단일 채널 또는 거리 미터 텐서)

        if color_image is None or depth_image is None:
            continue

        # 💡 [Depth 영상 전처리 및 컬러 맵 투영]
        # 1. 만약 depth_image의 스케일이 raw 데이터(uint16 등)라면 시각화를 위해 0~255 범위 uint8로 정규화합니다.
        if depth_image.dtype == np.uint16 or depth_image.dtype == torch.uint16:
            depth_scaled = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            depth_scaled = depth_image.astype(np.uint8)

        # 2. 깊이 왜곡 인지를 쉽게 만들기 위해 JET 혹은 다채로운 컬러 맵(ColorMap)으로 변경합니다.
        depth_colormap = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)

        # 3. 흑백/단일 채널 상태의 Depth 해상도(예: 848x480)를 컬러 맵 입힌 후 컬러 프레임(1280x720) 크기와 일치하도록 강제 업샘플링 리사이즈합니다.
        depth_colormap_resized = cv2.resize(depth_colormap, (1280, 720), interpolation=cv2.INTER_LINEAR)

        # 실시간 프레임 전처리 및 추론
        input_tensor = preprocess_image(color_image, device)
        
        with torch.no_grad():
            pred_logit, pred_boxes = model(input_tensor)

        # 바운딩 박스 + 중심점 투영 연산 수행
        result_color_view = visualize_inference(
            color_image, pred_logit, pred_boxes, conf_threshold=0.55, iou_threshold=0.45
        )

        # 💡 [화면 일괄 결합 연산]
        # 복원된 AI 컬러 이미지(좌)와 컬러맵 입힌 댑스 이미지(우)를 가로로 이어 붙입니다 (Horizontal Concatenate)
        final_combined_display = np.hstack((result_color_view, depth_colormap_resized))

        # 결합 창 생성 및 출력 (해상도가 가로로 2배인 2560x720으로 출력되므로, 너무 크면 cv2.resize로 축소 조정 가능)
        cv2.imshow("Melon Detection (LEFT: AI BBox & Center / RIGHT: Depth Jet Map)", final_combined_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    camera.stop()
    print("--- 스트리밍 프로그램 종료 ---")


if __name__ == "__main__":
    main()