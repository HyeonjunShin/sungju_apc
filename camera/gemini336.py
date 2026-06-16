import cv2
from pyorbbecsdk import OBAlignMode, Pipeline, Config
from pyorbbecsdk import OBSensorType, OBFormat, OBPropertyID, OBAlignMode
from frame import Frame

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

class Gemini336Camera:
    def __init__(self, width=640, height=480, fps=30, hw_align=True):
        try:
            self.pipeline = Pipeline()
        except RuntimeError as e:
            print(f"파이프라인을 초기화할 수 없습니다: {e}")
            exit(1)
        self.config = Config()

        self.width = width
        self.height = height
        self.FPS = fps

        color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = color_profiles.get_video_stream_profile(self.width, self.height, OBFormat.MJPG, self.FPS)
        self.config.enable_stream(color_profile)
        # self.config.enable_video_stream(OBStreamType.COLOR_STREAM, 640, 480, 30, OBFormat.MJPG)


        depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profiles.get_video_stream_profile(self.width, self.height, OBFormat.Y16, self.FPS)
        self.config.enable_stream(depth_profile)
        # self.config.enable_video_stream(OBStreamType.DEPTH_STREAM, 640, 480, 30, OBFormat.Y16)

        self.pipeline.enable_frame_sync()
        self.config.set_align_mode(OBAlignMode.HW_MODE if hw_align else OBAlignMode.SW_MODE)

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

    def set_camera_properties(self, exposure_time: int = 100, gain: int = 2, laser_power: int = 3):
        device = self.pipeline.get_device()
        device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, False)
        device.set_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, exposure_time)
        device.set_int_property(OBPropertyID.OB_PROP_COLOR_GAIN_INT, gain)

        device.set_bool_property(OBPropertyID.OB_PROP_DEPTH_ALIGN_HARDWARE_BOOL, True)
        # device.set_bool_property(OBPropertyID.OB_PROP_HARDWARE_DISTORTION_SWITCH_BOOL, True)

        # device.set_int_property(OBPropertyID.OB_PROP_MIN_DEPTH_INT, min_depth_mm)
        # device.set_int_property(OBPropertyID.OB_PROP_MAX_DEPTH_INT, max_depth_mm)

        # device.set_bool_property(OBPropertyID.OB_PROP_DEPTH_HOLEFILTER_BOOL, True)
        device.set_bool_property(OBPropertyID.OB_PROP_DEPTH_NOISE_REMOVAL_FILTER_BOOL, True)

        device.set_int_property(OBPropertyID.OB_PROP_LASER_CONTROL_INT, 1) # 강제 ON
        device.set_int_property(OBPropertyID.OB_PROP_LASER_POWER_LEVEL_CONTROL_INT, laser_power)
        # device.set_int_property(OBPropertyID.OB_PROP_DEVICE_AE_STRATEGY_INT, 1) # 1: Motion (잔상 방지)

        device.set_int_property(OBPropertyID.OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT, 2) # 0: Disabled, 1: 50Hz, 2: 60Hz, 3: Auto

    def get_camera_properties(self):
        device = self.pipeline.get_device()
        exposure_time = device.get_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT)
        gain = device.get_int_property(OBPropertyID.OB_PROP_COLOR_GAIN_INT)
        laser_power = device.get_int_property(OBPropertyID.OB_PROP_LASER_POWER_LEVEL_CONTROL_INT)
        power_line_frequency = device.get_int_property(OBPropertyID.OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT)
        print(f"Exposure Time: {exposure_time} µs, Gain: {gain}, Laser Power: {laser_power}, Power Line Frequency: {power_line_frequency}")

    def start(self):
        self.pipeline.start(self.config)

    def get_frames(self):
        frames = self.pipeline.wait_for_frames(100)
        if frames is None:
            return None

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return None
        
        timestamp = color_frame.get_timestamp()
        color_data = color_frame.get_data()
        depth_data = depth_frame.get_data()

        return Frame(timestamp, color_data, depth_data, self.width, self.height)
    
    def stop(self):
        self.pipeline.stop()
        print("Stop the Gemini 336 camera.")

def main():
    camera = Gemini336Camera()
    # camera.get_available_devices()
    # camera.get_available_stream_profiles()
    camera.set_camera_properties(1000, 1, 1)
    camera.get_camera_properties()
    camera.start()

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
