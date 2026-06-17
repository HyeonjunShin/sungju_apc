import os
import time
import json
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import open3d as o3d
import multiprocessing as mp
from multiprocessing import shared_memory
from torchvision.ops import nms
import zenoh  

# 기존 프로젝트의 모델 구조 및 카메라 임포트
from model import MelonDetectorDense  
from camera.camera_worker import camera_worker

# =====================================================================
# 1. GLOBAL CALIBRATION & COORDINATE MATRIX
# =====================================================================
DEPTH_PATCH_SIZE = 40
DEPTH_HALF_SIZE = DEPTH_PATCH_SIZE // 2

fx = 693.3101806640625
fy = 693.4061279296875
cx = 639.6598510742188
cy = 365.0723876953125
INTRINSIC = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
DISTORTION = np.array([0.00743896747007966, -0.05456198751926422, 0.03670734167098999, 0.0, 0.0, 0.0, 0.00016195396892726421, -0.001005938509479165])

# Hand-Eye Calibration 매트릭스
flange2camera = np.array([
    [ 0.0, -0.93969262,  0.34202014,  0.05991575],
    [ 1.0,  0.0,         0.0,         0.00358377],
    [-0.0,  0.34202014,  0.93969262,  0.03416166],
    [ 0.0,  0.0,         0.0,         1.0       ]
])


def normal_to_tf(centroid, normal):
    z_axis = normal / np.linalg.norm(normal)
    ref_vec = np.array([1, 0, 0])
    if abs(np.dot(ref_vec, z_axis)) > 0.95:
        ref_vec = np.array([0, 1, 0])
        
    x_axis = np.cross(ref_vec, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    tf_matrix = np.eye(4)
    tf_matrix[0:3, 0] = x_axis
    tf_matrix[0:3, 1] = y_axis
    tf_matrix[0:3, 2] = z_axis
    tf_matrix[0:3, 3] = centroid
    return tf_matrix


# =====================================================================
# 2. 3D VISUALIZATION ENGINE (Off-screen 렌더러)
# =====================================================================
class OffscreenViewer3D:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.vis = o3d.visualization.rendering.OffscreenRenderer(width, height)
        
        self.scene = self.vis.scene
        self.scene.set_background([0, 0, 0, 1])

        self.geom_counter = 0
        self.current_node_names = []

        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
        self.material = o3d.visualization.rendering.MaterialRecord()
        self.material.shader = "defaultUnlit" 
        self.scene.add_geometry("base_axes", axes, self.material)

        self.setup_camera()

    def setup_camera(self):
        center = [0.0, 0.0, 0.6] 
        eye = [0.0, -0.3, 0.1]   
        up = [0.0, 0.0, -1.0]     
        self.vis.setup_camera(60.0, center, eye, up)

    def update_scene(self, points_3d, tf_list):
        for name in self.current_node_names:
            self.scene.remove_geometry(name)
        self.current_node_names.clear()
        self.geom_counter = 0

        if len(points_3d) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_3d)
            pcd.paint_uniform_color([1.0, 0.8, 0.0])
            
            pcd_name = f"pcd_{self.geom_counter}"
            self.scene.add_geometry(pcd_name, pcd, self.material)
            self.current_node_names.append(pcd_name)
            self.geom_counter += 1

        for tf in tf_list:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.07)
            frame.transform(tf)
            
            frame_name = f"frame_{self.geom_counter}"
            self.scene.add_geometry(frame_name, frame, self.material)
            self.current_node_names.append(frame_name)
            self.geom_counter += 1

    def capture_frame(self):
        image_o3d = self.vis.render_to_image()
        img_np = np.asarray(image_o3d)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        return img_bgr

    def close(self):
        # OffscreenRenderer 자원 해제용 메서드 안전장치
        pass


# =====================================================================
# 3. CORE PROCESSING HELPER
# =====================================================================
def preprocess_image(color_image, device):
    image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (640, 360), interpolation=cv2.INTER_LINEAR)
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    return image_tensor


# =====================================================================
# 4. GLOBAL STATE VARIABLE FOR ZENOH (비동기 로봇 Flange 수신 데이터)
# =====================================================================
global_tf_flange = None

def on_robot_request(sample):
    global global_tf_flange
    payload = sample.payload.to_string()
    try:
        global_tf_flange = np.fromstring(payload, sep=" ").reshape((4, 4))
    except Exception as e:
        print(f"⚠️ Robot TF 파싱 실패: {e}")


# =====================================================================
# 5. MAIN EXECUTIVE LOOP (실시간 스트리밍 루프)
# =====================================================================
def main():
    global global_tf_flange
    mp.set_start_method('spawn', force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 통합 제어 및 Zenoh 실시간 스트리밍 허브 가동: {device} ---")

    # Zenoh 통신 초기화
    try:
        zenoh_config = zenoh.Config()
        z_session = zenoh.open(zenoh_config)
        zenoh_pub = z_session.declare_publisher("detector/response")
        zenoh_sub = z_session.declare_subscriber("detector/request", on_robot_request)
        print("✅ Zenoh 실시간 스트리밍 파이프라인 연결 성공 (detector/response)")
    except Exception as e:
        print(f"❌ Zenoh 바인딩 실패: {e}")
        return

    model_path = ".checkpoints/best_melon_bbox_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ 가중치 유실 종료: {model_path}")
        return

    model = MelonDetectorDense(num_bifpns=1, num_features=64, num_classes=1)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    # IPC 공유 메모리 구성 (1280x720)
    color_shape = (720, 1280, 3)
    depth_shape = (720, 1280) 
    
    color_bytes = np.prod(color_shape) * np.dtype(np.uint8).itemsize
    depth_bytes = np.prod(depth_shape) * np.dtype(np.uint16).itemsize

    shm_color = shared_memory.SharedMemory(create=True, size=color_bytes)
    shm_depth = shared_memory.SharedMemory(create=True, size=depth_bytes)

    local_color_buf = np.ndarray(color_shape, dtype=np.uint8, buffer=shm_color.buf)
    local_depth_buf = np.ndarray(depth_shape, dtype=np.uint16, buffer=shm_depth.buf)
    local_color_buf[:] = 0
    local_depth_buf[:] = 0

    shutdown_event = mp.Event()
    p_cam = mp.Process(target=camera_worker, args=(shm_color.name, shm_depth.name, color_shape, depth_shape, shutdown_event))
    p_cam.start()

    viewer_3d = OffscreenViewer3D(width=1280, height=720)

    color_image = np.zeros(color_shape, dtype=np.uint8)
    depth_image = np.zeros(depth_shape, dtype=np.uint16)
    prev_loop_time = time.time()

    print("--- 2D AI 디텍션 + 3D 법선 벡터 실시간 고속 스트리밍 시작 ---")

    try:
        while True:
            if np.sum(local_color_buf) == 0:
                time.sleep(0.01)
                continue

            color_image[:] = local_color_buf
            depth_image[:] = local_depth_buf
            
            orig_h, orig_w, _ = color_image.shape
            vis_image = color_image.copy()

            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_inference_time = time.time()

            input_tensor = preprocess_image(color_image, device)
            with torch.no_grad():
                pred_logit, pred_boxes = model(input_tensor)

            logits = pred_logit[0]
            boxes = pred_boxes[0]
            scores = F.softmax(logits, dim=-1)
            class_scores = scores[:, 1]

            keep = class_scores > 0.55
            
            all_detected_points_3d = []
            all_tf_matrices = []
            candidate_objects = []

            if keep.any():
                filtered_scores = class_scores[keep]
                filtered_boxes = boxes[keep]

                x1 = filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2
                y1 = filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2
                x2 = filtered_boxes[:, 0] + filtered_boxes[:, 2] / 2
                y2 = filtered_boxes[:, 1] + filtered_boxes[:, 3] / 2
                boxes_tlbr = torch.stack([x1, y1, x2, y2], dim=1)

                nms_keep_indices = nms(boxes_tlbr, filtered_scores, iou_threshold=0.45)
                
                if nms_keep_indices.numel() > 0:
                    final_scores = filtered_scores[nms_keep_indices].cpu().numpy()
                    final_boxes_tlbr = boxes_tlbr[nms_keep_indices].cpu().numpy()
                    final_query_ids = torch.where(keep)[0][nms_keep_indices].cpu().numpy()

                    # 1차 구조 연산 수집 루프
                    for score, box, q_id in zip(final_scores, final_boxes_tlbr, final_query_ids):
                        xmin = int(np.clip(box[0] * orig_w, 0, orig_w))
                        ymin = int(np.clip(box[1] * orig_h, 0, orig_h))
                        xmax = int(np.clip(box[2] * orig_w, 0, orig_w))
                        ymax = int(np.clip(box[3] * orig_h, 0, orig_h))

                        cx_pixel = int((xmin + xmax) / 2)
                        cy_pixel = int((ymin + ymax) / 2)

                        patch_xmin = np.clip(cx_pixel - DEPTH_HALF_SIZE, 0, orig_w - DEPTH_PATCH_SIZE)
                        patch_ymin = np.clip(cy_pixel - DEPTH_HALF_SIZE, 0, orig_h - DEPTH_PATCH_SIZE)
                        
                        depth_patch = depth_image[patch_ymin : patch_ymin + DEPTH_PATCH_SIZE, patch_xmin : patch_xmin + DEPTH_PATCH_SIZE] * 0.001
                        depth_patch_flat = depth_patch.ravel()

                        mx, my = np.meshgrid(np.arange(patch_xmin, patch_xmin + DEPTH_PATCH_SIZE), np.arange(patch_ymin, patch_ymin + DEPTH_PATCH_SIZE))
                        mesh = np.vstack((mx.ravel(), my.ravel())).T.astype(np.float32)

                        if depth_patch_flat.size > 0:
                            x_normal, y_normal = cv2.undistortPoints(mesh, INTRINSIC, DISTORTION).squeeze().T
                            x_c = x_normal * depth_patch_flat
                            y_c = y_normal * depth_patch_flat
                            z_c = depth_patch_flat

                            valid_mask = z_c > 0
                            points_3d = np.vstack((x_c[valid_mask], y_c[valid_mask], z_c[valid_mask])).T

                            if len(points_3d) >= 20:
                                centroid = np.mean(points_3d, axis=0)
                                centered_pts = points_3d - centroid
                                cov = np.cov(centered_pts.T)
                                
                                evals, evecs = np.linalg.eig(cov)
                                normal = evecs[:, np.argmin(evals)]
                                if normal[2] < 0:
                                    normal = -normal

                                tf_matrix = normal_to_tf(centroid, normal)
                                
                                # 기하 융합 랭킹 스코어링 산출
                                camera_optical_axis = np.array([0.0, 0.0, 1.0])
                                angle_similarity = np.dot(normal, camera_optical_axis) / (np.linalg.norm(normal) * np.linalg.norm(camera_optical_axis) + 1e-6)
                                angle_similarity = max(0.0, angle_similarity)
                                
                                distance = centroid[2]
                                min_work_dist, max_work_dist = 0.2, 1.2
                                dist_score = (max_work_dist - distance) / (max_work_dist - min_work_dist + 1e-6)
                                dist_score = np.clip(dist_score, 0.0, 1.0)

                                # 🔥 [추가] 이미지 중앙 가중치 랭킹 계산 알고리즘
                                img_center_x = orig_w / 2.0
                                img_center_y = orig_h / 2.0
                                max_dist_from_center = np.sqrt(img_center_x**2 + img_center_y**2) # 최대 가능 거리를 기준으로 정규화
                                
                                pixel_dist_from_center = np.sqrt((cx_pixel - img_center_x)**2 + (cy_pixel - img_center_y)**2)
                                center_score = 1.0 - (pixel_dist_from_center / max_dist_from_center)
                                center_score = np.clip(center_score, 0.0, 1.0)

                                # 가중치 비중 수정: AI(30%) + 법선각도(20%) + 작업거리(30%) + 이미지중앙(20%) = 1.0
                                grabbing_score = (0.3 * score) + (0.2 * angle_similarity) + (0.3 * dist_score) + (0.2 * center_score)

                                candidate_objects.append({
                                    "grabbing_score": grabbing_score,
                                    "ai_score": score,
                                    "box_2d": (xmin, ymin, xmax, ymax, cx_pixel, cy_pixel),
                                    "points_3d": points_3d,
                                    "tf_matrix": tf_matrix,
                                    "q_id": q_id,
                                    "dist_m": distance
                                })

                    # 파지 랭킹 내림차순 정렬
                    candidate_objects.sort(key=lambda item: item["grabbing_score"], reverse=True)

                    # 2차 루프 시각화 처리
                    for rank, obj in enumerate(candidate_objects):
                        g_score = obj["grabbing_score"]
                        q_id = obj["q_id"]
                        xmin, ymin, xmax, ymax, cx, cy = obj["box_2d"]
                        dist_m = obj["dist_m"]
                        
                        color_id = np.random.randint(0, 256, size=3).tolist()
                        all_detected_points_3d.append(obj["points_3d"])
                        all_tf_matrices.append(obj["tf_matrix"])

                        cv2.rectangle(vis_image, (xmin, ymin), (xmax, ymax), color_id, 2, cv2.LINE_AA)
                        
                        if rank == 0:
                            cv2.drawMarker(vis_image, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=22, thickness=2)
                        else:
                            cv2.circle(vis_image, (cx, cy), 4, color_id, -1, cv2.LINE_AA)

                        text = f"Rank {rank+1} [ID:{q_id}]"
                        score_text = f"Grab Score: {g_score:.2f} (Dist: {dist_m:.2f}m)"
                        text_y = ymin - 25 if ymin - 25 > 20 else ymin + 20
                        cv2.putText(vis_image, text, (xmin, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color_id, 1, cv2.LINE_AA)
                        cv2.putText(vis_image, score_text, (xmin, text_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

            # [핵심 최적화: 조건문 없이 실시간 고속 스트리밍 데이터 전송]
            if global_tf_flange is not None and len(candidate_objects) > 0:
                send_data_structure = []
                tf_camera_const = global_tf_flange @ flange2camera
                
                for rank, obj in enumerate(candidate_objects):
                    tf_camera = obj["tf_matrix"]
                    tf_base = tf_camera_const @ tf_camera
                    
                    tf_matrix_base = tf_base.flatten().tolist()
                    centroid_base = tf_base[0:3, 3].tolist()
                    
                    send_data_structure.append({
                        "rank": rank,
                        "score": round(float(obj["grabbing_score"]), 4),
                        "centroid_base": [round(c, 4) for c in centroid_base],
                        "tf_matrix_base": [round(x, 6) for x in tf_matrix_base]
                    })
                
                # 비차단식 비동기 퍼블리시
                json_payload = json.dumps(send_data_structure)
                zenoh_pub.put(json_payload)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_inference_time = time.time()
            inference_ms = (end_inference_time - start_inference_time) * 1000

            if len(all_detected_points_3d) > 0:
                merged_points_3d = np.vstack(all_detected_points_3d)
                viewer_3d.update_scene(merged_points_3d, all_tf_matrices)
            else:
                viewer_3d.update_scene(np.zeros((0, 3)), [])
            
            view_3d_captured = viewer_3d.capture_frame()

            # 성능 지표
            current_loop_time = time.time()
            fps = 1.0 / (current_loop_time - prev_loop_time + 1e-6)
            prev_loop_time = current_loop_time

            cv2.putText(vis_image, f"System FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(vis_image, f"Inference: {inference_ms:.1f} ms", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

            # 단일 창 가로 결합 송출
            final_combined_display = np.hstack((vis_image, view_3d_captured))
            
            if global_tf_flange is not None:
                cv2.putText(final_combined_display, "● STREAMING ON (Zenoh Auto-Pub)", (25, orig_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(final_combined_display, "○ STREAMING OFF (Waiting for Robot TF)", (25, orig_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("Unified Melon AI System (Left: 2D Detection / Right: 3D Vector View)", final_combined_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[Main] 인터럽트 종료.")
    finally:
        shutdown_event.set()
        p_cam.join()
        try:
            shm_color.close()
            shm_depth.close()
            shm_color.unlink()
            shm_depth.unlink()
        except:
            pass
        z_session.close() 
        viewer_3d.close()
        cv2.destroyAllWindows()
        print("[Main] 리소스 완전히 셧다운 완료.")


if __name__ == "__main__":
    main()