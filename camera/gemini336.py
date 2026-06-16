import cv2
from pyorbbecsdk import OBAlignMode, Pipeline, Config
from pyorbbecsdk import OBSensorType, OBFormat, OBPropertyID, AlignFilter, OBStreamType, OBException
from .frame import Frame

def check_available_d2c_profile(pipeline, color_profile):
    depth_profiles = pipeline.get_d2c_depth_profile_list(color_profile, OBAlignMode.HW_MODE)
    for i in range(depth_profiles.get_count()):
        profile = depth_profiles.get_stream_profile_by_index(i)
        print(f"[{i:02d}] {color_profile} {color_profile.get_format()} {profile} {profile.get_format()}")

def check_available_d2c_profiles(pipeline, color_profiles):
    for i in range(color_profiles.get_count()):
        color_profile = color_profiles.get_stream_profile_by_index(i)
        if color_profile.get_fps() != 30: continue
        if color_profile.get_format() != OBFormat.MJPG: continue
        # if color_profile.get_width() != 1280: continue
        # if color_profile.get_height() != 720: continue
        check_available_d2c_profile(pipeline, color_profile)

def find_color_profile(pipeline, width, height, fps):
    try:
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = color_profiles.get_video_stream_profile(width, height, OBFormat.MJPG, fps)
    except Exception as e:
        print(f"Color stream profile not found: {e}")
        return None
    return color_profile

def find_profile(width, height, fps):
    pipline = Pipeline()
    color_profiles = pipline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = color_profiles.get_video_stream_profile(1920, 1080, OBFormat.MJPG, 30)

class Gemini336Camera:
    def __init__(self):
        try:
            self.pipeline = Pipeline()
        except RuntimeError as e:
            print(f"파이프라인을 초기화할 수 없습니다: {e}")
            exit(1)
        self.config = Config()

    def set_color_profile(self, width, height, fps):
        self.color_width = width
        self.color_height = height
        self.color_fps = fps

        color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = color_profiles.get_video_stream_profile(self.color_width, self.color_height, OBFormat.MJPG, self.color_fps)
        self.config.enable_stream(color_profile)
        # self.config.enable_video_stream(OBStreamType.COLOR_STREAM, 640, 480, 30, OBFormat.MJPG)

    def set_depth_profile(self, width, height, fps, hw_align=True):
        self.depth_width = width
        self.depth_height = height
        self.depth_fps = fps

        depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profiles.get_video_stream_profile(self.depth_width, self.depth_height, OBFormat.Y16, self.depth_fps)
        self.config.enable_stream(depth_profile)
        # self.config.enable_video_stream(OBStreamType.DEPTH_STREAM, 640, 480, 30, OBFormat.Y16)

        self.align_mode = OBAlignMode.HW_MODE if hw_align else OBAlignMode.SW_MODE
        self.config.set_align_mode(self.align_mode)

    def get_available_devices(self):
        for i in range(self.pipeline.get_device().get_sensor_list().get_count()):
            sensor = self.pipeline.get_device().get_sensor_list().get_type_by_index(i)
            print(sensor)

    def get_available_stream_profiles(self):
        print("=== Available Color Stream Profiles ===")
        color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        for i in range(color_profiles.get_count()):
            profile = color_profiles.get_stream_profile_by_index(i)
            print(f"[{i:02d}] {profile} {profile.get_format()}")

        print("\n=== Available Depth Stream Profiles ===")
        depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        for i in range(depth_profiles.get_count()):
            profile = depth_profiles.get_stream_profile_by_index(i)
            print(f"[{i:02d}] {profile} {profile.get_format()}")

    def set_camera_properties(self, 
                              exposure_time: int = 100, 
                              gain: int = 2, 
                              laser_power: int = 3,
                              auto_white_balance: bool = False,
                              white_balance_temp: int = 4600, # 2800 ~ 6500K 범위
                              brightness: int = 0,            # -64 ~ 64 범위
                              contrast: int = 50):            # 0 ~ 100 범위
        
        device = self.pipeline.get_device()
        
        # --- 1. Color (RGB) 노출 및 게인 제어 ---
        device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, False)
        device.set_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, exposure_time)
        device.set_int_property(OBPropertyID.OB_PROP_COLOR_GAIN_INT, gain)

        # --- 2. Color (RGB) 화이트 밸런스 및 색상 제어 ---
        try:
            # 화이트 밸런스 자동/수동 제어
            device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, auto_white_balance)
            if not auto_white_balance:
                device.set_int_property(OBPropertyID.OB_PROP_COLOR_WHITE_BALANCE_INT, white_balance_temp)
            
            # 기타 화질 관련 제어 (필요 시 주석 해제)
            # device.set_int_property(OBPropertyID.OB_PROP_COLOR_BRIGHTNESS_INT, brightness)
            # device.set_int_property(OBPropertyID.OB_PROP_COLOR_CONTRAST_INT, contrast)
        except OBException as e:
            print(f"Color 추가 속성 설정 실패 (장치 지원 여부 확인 필요): {e}")

        # --- 3. 플리커 현상 방지 (전원 주파수) ---
        device.set_int_property(OBPropertyID.OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT, 2) # 0: Disabled, 1: 50Hz, 2: 60Hz, 3: Auto

        # --- 4. Depth (깊이) 필터 및 IR 레이저 제어 ---
        device.set_bool_property(OBPropertyID.OB_PROP_DEPTH_NOISE_REMOVAL_FILTER_BOOL, True)
        device.set_int_property(OBPropertyID.OB_PROP_LASER_CONTROL_INT, 1) # 강제 ON
        device.set_int_property(OBPropertyID.OB_PROP_LASER_POWER_LEVEL_CONTROL_INT, laser_power)

    def get_camera_properties(self):
        device = self.pipeline.get_device()
        
        # 값 읽어오기
        exposure_time = device.get_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT)
        gain = device.get_int_property(OBPropertyID.OB_PROP_COLOR_GAIN_INT)
        laser_power = device.get_int_property(OBPropertyID.OB_PROP_LASER_POWER_LEVEL_CONTROL_INT)
        power_line_frequency = device.get_int_property(OBPropertyID.OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT)
        
        # Color 추가 속성 읽기 (안전한 예외 처리)
        auto_wb = "Unknown"
        wb_temp = "Unknown"
        try:
            auto_wb = device.get_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL)
            wb_temp = device.get_int_property(OBPropertyID.OB_PROP_COLOR_WHITE_BALANCE_INT)
        except OBException:
            pass

        # 출력 출력
        print("\n=== Current Camera Properties ===")
        print(f"Exposure Time        : {exposure_time} µs")
        print(f"Gain                 : {gain}")
        print(f"Auto White Balance   : {auto_wb}")
        print(f"White Balance Temp   : {wb_temp} K")
        print(f"Laser Power Level    : {laser_power}")
        print(f"Power Line Frequency : {power_line_frequency}")
        print("=================================\n")

    def start(self):
        if self.align_mode == OBAlignMode.HW_MODE:
            print("Starting Gemini 336 camera with HW_ALIGN mode...")
        else:
            print("Starting Gemini 336 camera with SW_ALIGN mode...")
            self.align_filter = AlignFilter(OBStreamType.COLOR_STREAM)

        self.pipeline.enable_frame_sync()
        self.pipeline.start(self.config)

    def get_frames(self):
        frames = self.pipeline.wait_for_frames(100)
        if frames is None or frames.get_color_frame() is None or frames.get_depth_frame() is None:
            return None
        
        if self.align_mode == OBAlignMode.SW_MODE:
            frames = self.align_filter.process(frames)
        
        timestamp = frames.get_color_frame().get_timestamp()
        color_data = frames.get_color_frame().get_data()
        depth_data = frames.get_depth_frame().get_data()

        return Frame(timestamp, color_data, depth_data, self.color_width, self.color_height)
    
    def get_intrinsic(self):
        cam_prop = self.pipeline.get_camera_param().rgb_intrinsic
        fx = cam_prop.fx
        fy = cam_prop.fy
        cx = cam_prop.cx
        cy = cam_prop.cy
        w = cam_prop.width
        h = cam_prop.height
        return fx, fy, cx, cy, w, h
    
    def get_distortion(self):
        cam_prop = self.pipeline.get_camera_param().rgb_distortion
        k1 = cam_prop.k1
        k2 = cam_prop.k2
        k3 = cam_prop.k3
        k4 = cam_prop.k4
        k5 = cam_prop.k5
        k6 = cam_prop.k6
        p1 = cam_prop.p1
        p2 = cam_prop.p2
        return k1, k2, k3, k4, k5, k6, p1, p2

    def stop(self):
        self.pipeline.stop()
        print("Stop the Gemini 336 camera.")

def main():
    camera = Gemini336Camera()
    camera.set_color_profile(1280, 720, 30)
    camera.set_depth_profile(848, 480, 30, hw_align=False)

    # camera.get_available_devices()
    # camera.get_available_stream_profiles()
    camera.set_camera_properties(80, 0, 1)
    camera.get_camera_properties()
    camera.start()

    print(camera.get_intrinsic())
    print(camera.get_distortion())

    while True:
        frame = camera.get_frames()
        if frame is None:
            continue

        timestamp = frame.get_timestamp()
        color_image = frame.get_color()
        depth_image = frame.get_depth()
        print(f"Timestamp: {timestamp}, Color Frame: {color_image.shape}, Depth Frame: {depth_image.shape}")

        cv2.imshow("Gemini 336 RGB", color_image)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        view = cv2.addWeighted(depth_colormap, 0.5, color_image, 0.5, 0, depth_colormap)
        # cv2.imshow("Gemini 336 Overlay", view)
        # cv2.imshow("Gemini 336 Depth", depth_colormap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    camera.stop()

if __name__ == "__main__":
    main()
