import numpy as np
import open3d as o3d
from open3d.visualization import gui

# 1. STL 파일 로드 및 법선 계산
mesh = o3d.io.read_triangle_mesh('/home/uon/code/hand-eye_calibration/model/grippers/handEyeG_V1.STL')
mesh.compute_vertex_normals()

target_point = [0.06146, 0.001, 0.03085]
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector([target_point])
# pcd.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]])
tf_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.05, 
    origin=target_point
)
R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0.0, np.radians(20), 0.0]))
tf_frame.rotate(R)

# 2. GUI Application 초기화
gui.Application.instance.initialize()

# 3. 해상도를 명시하여 시각화 창 객체 생성
vis = o3d.visualization.O3DVisualizer("Open3D - handEyeG", 1920, 1080)
vis.show_settings = True  # 우측 UI 패널 표시 여부

# 4. 메쉬 등록 (딕셔너리 포맷 사용)
vis.add_geometry({"name": "handEyeG_Mesh", "geometry": mesh})
vis.add_geometry({"name": "My_Target_Point", "geometry": tf_frame})


# 5. 창을 띄우고 앱 실행
vis.reset_camera_to_default()
gui.Application.instance.add_window(vis)
gui.Application.instance.run()

t = tf_frame.get_center()

# 3. 4x4 변환 행렬(T) 조립
T = np.eye(4)
T[0:3, 0:3] = R       # 좌상단 3x3에 회전 행렬 대입
T[0:3, 3] = t         # 우측 3x1에 이동 벡터 대입

print("--- 원점에서 TF 프레임으로의 4x4 변환 행렬 ---")
print(T)

T_inverse = np.linalg.inv(T)
print("--- TF 프레임에서 원점으로의 4x4 역변환 행렬 ---")
print(T_inverse)