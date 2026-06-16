import time
import json
import numpy as np
import zenoh

# =====================================================================
# 1. 랜덤 Flange 변환 매트릭스(TF) 생성기
# =====================================================================
def generate_random_flange_tf():
    """
    로봇이 움직이는 것처럼 연출하기 위해 
    약간의 변동성이 있는 무작위 4x4 변환 매트릭스를 생성합니다.
    """
    # 임의의 회전 각도 (라디안)
    roll = np.random.uniform(-0.1, 0.1)
    pitch = np.random.uniform(-0.1, 0.1)
    yaw = np.random.uniform(-3.14, 3.14)
    
    # 회전 매트릭스 계산
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    
    # 임의의 위치 (X, Y, Z 미터 단위)
    t = np.array([
        np.random.uniform(0.3, 0.6),   # X
        np.random.uniform(-0.2, 0.2),  # Y
        np.random.uniform(0.4, 0.8)    # Z
    ])
    
    # 4x4 매트릭스 조립
    tf = np.eye(4)
    tf[0:3, 0:3] = R
    tf[0:3, 3] = t
    return tf


# =====================================================================
# 2. 수신 데이터 콜백 함수 (메인 루프에서 보낸 결과 출력)
# =====================================================================
def on_detector_response(sample):
    payload = sample.payload.to_string()
    try:
        data = json.loads(payload)
        print("\n==================================================")
        print(f"📥 [수신] Melon AI 응답 데이터 (검출 수: {len(data)}개)")
        print("==================================================")
        for item in data:
            print(f"▶ Rank {item['rank'] + 1} (Score: {item['score']})")
            print(f"   - Centroid Base (XYZ): {item['centroid_base']}")
            # 4x4 형태를 가독성 좋게 출력하기 위해 재배열
            tf_mat = np.array(item['tf_matrix_base']).reshape(4, 4)
            print("   - TF Matrix Base:")
            for row in tf_mat:
                print(f"     [ {row[0]:.4f}, {row[1]:.4f}, {row[2]:.4f}, {row[3]:.4f} ]")
    except Exception as e:
        print(f"⚠️ 데이터 파싱 에러: {e}\nRaw Payload: {payload}")


# =====================================================================
# 3. 메인 실행 루프
# =====================================================================
def main():
    print("--- 🤖 Robot Flange Mock 테스트 프로그램 가동 ---")
    
    # Zenoh 세션 초기화
    try:
        config = zenoh.Config()
        session = zenoh.open(config)
        
        # 1) 메인 루프에게서 처리 결과를 받을 Subscriber 등록
        sub = session.declare_subscriber("detector/response", on_detector_response)
        
        # 2) 메인 루프에게 로봇 위치를 던져줄 Publisher 등록
        pub = session.declare_publisher("detector/request")
        print("✅ Zenoh 테스트 파이프라인 수발신 바인딩 완료.")
    except Exception as e:
        print(f"❌ Zenoh 연결 실패: {e}")
        return

    print("\n[INFO] 1초 간격으로 무작위 Flange TF를 전송합니다. (종료: Ctrl+C)")
    
    try:
        while True:
            # 1. 무작위 TF 생성
            tf_flange = generate_random_flange_tf()
            
            # 2. 메인 루프가 파싱할 수 있게 공백(" ") 구분자 형태의 문자열 포맷팅 전환
            # 예: "1.0 0.0 0.0 0.5 0.0 1.0 ..."
            tf_string = " ".join(map(str, tf_flange.flatten()))
            
            # 3. 데이터 송신
            print(f"\n📤 [송신] Flange XYZ: {[round(x, 3) for x in tf_flange[0:3, 3]]}")
            pub.put(tf_string)
            
            time.sleep(1.0) # 1초 대기
            
    except KeyboardInterrupt:
        print("\n[Test] 사용자에 의해 프로그램이 종료되었습니다.")
    finally:
        session.close()
        print("[Test] Zenoh 세션이 안전하게 닫혔습니다.")


if __name__ == "__main__":
    main()