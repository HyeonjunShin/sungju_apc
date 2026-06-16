import zenoh
import time
import numpy as np
import cv2
import os
os.environ["XDG_SESSION_TYPE"] = "x11"
import open3d as o3d
import json
from camera.gemini336 import Gemini336Camera

WIDTH = 1920
HEIGHT = 1080
CHANNELS = 3
DEPTH_PATCH_SIZE = 50
DEPTH_HALF_SIZE = DEPTH_PATCH_SIZE // 2

PATCH_MASK = np.arange(-DEPTH_HALF_SIZE, DEPTH_HALF_SIZE + 1)


fx=693.3101806640625
fy=693.4061279296875
cx=639.6598510742188
cy=365.0723876953125
w=1280
h=720

INTRINSIC = np.array(
    [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
)

DISTORTION = np.array(
    [0.00743896747007966, -0.05456198751926422, 0.03670734167098999, 0.0, 0.0, 0.0, 0.00016195396892726421, -0.001005938509479165]
)

flange2camera = np.array([
    [ 0.0, -0.93969262,  0.34202014,  0.05991575],
    [ 1.0,  0.0,         0.0,         0.00358377],
    [-0.0,  0.34202014,  0.93969262,  0.03416166],
    [ 0.0,  0.0,         0.0,         1.0       ]
])



def normal_to_tf(centroid, normal):
    """
    Centroid와 Normal을 결합하여 안정적인 4x4 TF 행렬 생성
    """
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


class Viewer3D:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='Viewer 3D', width=800, height=600)
        
        opt = self.vis.get_render_option()
        opt.point_size = 1.0
        opt.background_color = np.asarray([0, 0, 0])

        self.view_ctl = self.vis.get_view_control()
        self.view_ctl.set_constant_z_near(0.01)
        self.view_ctl.set_constant_z_far(5.0)
        self.view_ctl.set_zoom(2)

        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)

        self.target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        self.vis.add_geometry(self.target_frame)

        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.vis.add_geometry(axes)
    
    def update_pcd(self, points_3d):
        self.pcd.points = o3d.utility.Vector3dVector(points_3d)
        self.pcd.paint_uniform_color([1.0, 0.8, 0.0])
        self.vis.update_geometry(self.pcd)

        centroid = np.mean(points_3d, axis=0)
        self.view_ctl.set_lookat(centroid)
        self.view_ctl.set_zoom(0.5)

    def update_frame(self, tf):
        new_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        new_frame.transform(tf)
        
        self.target_frame.vertices = new_frame.vertices
        self.vis.update_geometry(self.target_frame)

    def close(self):
        self.vis.destroy_window()

    def update(self):
        self.vis.poll_events()
        self.vis.update_renderer()


class CommModule:
    def __init__(self):
        try:
            conf = zenoh.Config()
            self.z_session = zenoh.open(conf)
            self.pub = self.z_session.declare_publisher("detector/response")
            self.sub = self.z_session.declare_subscriber("detector/request", self.on_zenoh)
            print("✅ Zenoh 연결 성공 (Topic: detector/response)")
        except Exception as e:
            print(f"❌ Zenoh 연결 실패: {e}")
            return
        
        cv2.namedWindow("view", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("view", self.on_mouse)

        self.camera = Gemini336Camera(1280, 720, 30, False)
        self.camera.set_camera_properties(50, 0, 1)
        self.camera.start()

        self.tf_flange = None
        self.x = None
        self.y = None
        self.depth_patch = None

        self.viewer_3d = Viewer3D()

    def on_zenoh(self, sample):
        payload = sample.payload.to_string()
        self.tf_flange = np.fromstring(payload, sep=" ").reshape((4,4))

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 이미지 가장자리 클릭 시 에러 방지를 위한 바운더리 체크
            if (DEPTH_HALF_SIZE <= x < WIDTH - DEPTH_HALF_SIZE) and (DEPTH_HALF_SIZE <= y < HEIGHT - DEPTH_HALF_SIZE):
                self.x = x
                self.y = y
            else:
                print("⚠️ 이미지의 가장자리 영역은 선택할 수 없습니다.")

    def run(self):
        try:
            while True:
                frame = self.camera.get_frames()
                if frame is None:
                    continue

                timestamp = frame.get_timestamp()
                color = frame.get_color()
                depth = frame.get_depth()
                print(depth.shape)

                # view_color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                view_color = color
                
                if self.x is not None and self.y is not None:
                    # self.depth_patch = depth[self.y - DEPTH_HALF_SIZE : self.y + DEPTH_HALF_SIZE + 1, self.x - DEPTH_HALF_SIZE : self.x + DEPTH_HALF_SIZE + 1, 0] * 0.001
                    self.depth_patch = depth[self.y - DEPTH_HALF_SIZE : self.y + DEPTH_HALF_SIZE + 1, self.x - DEPTH_HALF_SIZE : self.x + DEPTH_HALF_SIZE + 1] * 0.001

                    self.depth_patch = self.depth_patch.ravel()
                    
                    mx, my = np.meshgrid(PATCH_MASK + self.x, PATCH_MASK + self.y)
                    mesh = np.vstack((mx.ravel(), my.ravel())).T.astype(np.float32)

                    if self.depth_patch is not None and self.depth_patch.size > 0:
                        x_normal, y_normal = cv2.undistortPoints(mesh, INTRINSIC, DISTORTION).squeeze().T

                        x_c = x_normal * self.depth_patch
                        y_c = y_normal * self.depth_patch
                        z_c = self.depth_patch

                        # 유효한 Depth 값만 필터링 (0보다 큰 값)
                        valid_mask = z_c > 0
                        points_3d = np.vstack((x_c[valid_mask], y_c[valid_mask], z_c[valid_mask])).T

                        # ====== 🚨 [핵심 수정] 유효 포인트 개수 검증을 통한 LinAlgError 예방 ======
                        if len(points_3d) < 20: 
                            print(f"⚠️ 클릭한 영역에 유효한 Depth가 부족합니다. (유효 포인트 수: {len(points_3d)} / 최소 요구: 20)")
                            self.x, self.y = None, None  # 플래그 초기화
                            continue
                        # =========================================================================

                        centroid = np.mean(points_3d, axis=0)
                        centered_pts = points_3d - centroid
                        cov = np.cov(centered_pts.T)
                        
                        evals, evecs = np.linalg.eig(cov)
                        normal = evecs[:, np.argmin(evals)]
                        if normal[2] < 0:
                            normal = -normal

                        tf = normal_to_tf(centroid, normal)

                        self.viewer_3d.update_pcd(points_3d)
                        self.viewer_3d.update_frame(tf)

                        cv2.circle(view_color, (self.x, self.y), 5, (0, 255, 0), -1)

                cv2.imshow("view", view_color)
                key = cv2.waitKey(1)

                self.viewer_3d.update()
                if key == ord("q"):
                    break

                if key == 32: # Space bar
                    if self.tf_flange is not None and 'tf' in locals():
                        tf_camera_const = self.tf_flange @ flange2camera
                        tf_base = tf_camera_const @ tf
                        
                        tf_matrix_base = tf_base.flatten().tolist()
                        centroid_base = tf_base[0:3, 3].tolist()
                        
                        send_data_structure = [
                            {
                                "rank": 0,
                                "score": 1.0,
                                "centroid_base": [round(c, 4) for c in centroid_base],
                                "tf_matrix_base": [round(x, 6) for x in tf_matrix_base]
                            }
                        ]
                        print(send_data_structure)
                        json_payload = json.dumps(send_data_structure)
                        print(f"🚀 [Sending JSON] Base Frame data serialized.")
                        self.pub.put(json_payload)
                    else:
                        print("⚠️ 로봇 TF 데이터가 없거나 계산된 타겟 TF가 없습니다.")

        except KeyboardInterrupt:
            print("\n 모니터링을 종료합니다.")
        finally:
            self.z_session.close()
            self.viewer_3d.close()


def main():
    comm_module = CommModule()
    comm_module.run()

if __name__ == "__main__":
    main()