from camera.gemini336 import Gemini336Camera
import numpy as np
from multiprocessing import shared_memory

def camera_worker(color_mem_name, depth_mem_name, color_shape, depth_shape, shutdown_event):
    camera = Gemini336Camera()
    
    camera.set_color_profile(1280, 720, 30)
    camera.set_depth_profile(848, 480, 30, hw_align=False) 
    camera.set_camera_properties(70, 0)
    camera.start()
    
    existing_color_shm = shared_memory.SharedMemory(name=color_mem_name)
    existing_depth_shm = shared_memory.SharedMemory(name=depth_mem_name)
    
    # 공유 메모리를 NumPy 배열 버퍼 구조로 연결 (포인터 맵핑)
    shared_color_buf = np.ndarray(color_shape, dtype=np.uint8, buffer=existing_color_shm.buf)
    shared_depth_buf = np.ndarray(depth_shape, dtype=np.uint16, buffer=existing_depth_shm.buf)

    
    try:
        while not shutdown_event.is_set():
            frame = camera.get_frames()
            if frame is None:
                continue

            color_image = frame.get_color()  # [720, 1280, 3] (BGR)
            depth_image = frame.get_depth()  # [720, 1280] (uint16)

            if color_image is None or depth_image is None:
                continue

            # 💡 메인 프로세스 처리 속도와 관계없이 하드웨어 최대 속도로 공유 메모리 덮어쓰기
            shared_color_buf[:] = color_image
            shared_depth_buf[:] = depth_image
            
    except Exception as e:
        print(f"[Camera Process] 내부 예외 발생: {e}")
    finally:
        camera.stop()
        existing_color_shm.close()
        existing_depth_shm.close()
        print("[Camera Process] 프로세스 자원이 안전하게 반환되었습니다.")


"""
    color_shape = (720, 1280, 3)
    depth_shape = (720, 1280) 

    color_bytes = np.prod(color_shape) * np.dtype(np.uint8).itemsize
    depth_bytes = np.prod(depth_shape) * np.dtype(np.uint16).itemsize

    # OS 가상 메모리 단에 공유 메모리 할당
    shm_color = shared_memory.SharedMemory(create=True, size=color_bytes)
    shm_depth = shared_memory.SharedMemory(create=True, size=depth_bytes)

    # 로컬 버퍼를 공유 메모리 주소(buf)와 직접 링크
    local_color_buf = np.ndarray(color_shape, dtype=np.uint8, buffer=shm_color.buf)
    local_depth_buf = np.ndarray(depth_shape, dtype=np.uint16, buffer=shm_depth.buf)

    # 초기 쓰레기 값 방지를 위한 무효화 마스킹
    local_color_buf[:] = 0
    local_depth_buf[:] = 0

    # 안전 제어용 종료 이벤트 인터럽트 정의
    shutdown_event = mp.Event()

    # 카메라 전용 자식 프로세스(Process) 셋업 및 실행
    p_cam = mp.Process(
        target=camera_worker, 
        args=(shm_color.name, shm_depth.name, color_shape, depth_shape, shutdown_event)
    )
    p_cam.start()
"""

"""
    shutdown_event.set()
    p_cam.join()

    shm_color.close()
    shm_depth.close()
    shm_color.unlink()
    shm_depth.unlink()
"""