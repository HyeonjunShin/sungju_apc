import numpy as np
import blosc
import msgpack
import cv2

class Frame:
    def __init__(self, timestamp, color, depth, width, height):
        self.timestamp = timestamp
        self.color_bytes = color
        self.depth_bytes = depth
        self.width = width
        self.height = height

        self.color_image = None
        self.depth_image = None

    def to_bytes(self):
        color_bytes = self.color_bytes.tobytes() if isinstance(self.color_bytes, np.ndarray) else self.color_bytes
        depth_bytes = blosc.compress(self.depth_bytes.tobytes(), typesize=2, cname='lz4')

        payload = {
            'timestamp': self.timestamp,
            'color_bytes': color_bytes,      # MJPG bytes
            'depth_bytes': depth_bytes,      # depth bytes
            'width': self.width,
            'height': self.height
        }
        return msgpack.packb(payload)
    
    @classmethod
    def from_bytes(cls, data):
        payload = msgpack.unpackb(data, raw=False)
        
        return cls(
            timestamp=payload['timestamp'],
            color_bytes=payload['color_bytes'],     # 아직 압축된 MJPG 바이너리 상태
            depth_bytes=payload['depth_bytes'],     # 아직 압축된 blosc 바이너리 상태
            width=payload['width'],
            height=payload['height'],   
        )

    def get_color(self):
        if self.color_image is None and isinstance(self.color_bytes, np.ndarray):
            self.color_image = cv2.imdecode(self.color_bytes, cv2.IMREAD_COLOR)
        elif self.color_image is None and isinstance(self.color_bytes, (bytes, bytearray)):
            color_array = np.frombuffer(self.color_bytes, dtype=np.uint8)
            self.color_image = cv2.imdecode(color_array, cv2.IMREAD_COLOR)
        return self.color_image

    def get_depth(self):
        if self.depth_image is None and isinstance(self.depth_bytes, np.ndarray):
            self.depth_image = np.frombuffer(self.depth_bytes, dtype=np.uint16).reshape((self.height, self.width))
        elif self.depth_image is None and isinstance(self.depth_bytes, (bytes, bytearray)):
            depth_raw = blosc.decompress(self.depth_bytes)
            self.depth_image = np.frombuffer(depth_raw, dtype=np.uint16).reshape((self.height, self.width))
        return self.depth_image
            
    def get_timestamp(self):
        return self.timestamp
