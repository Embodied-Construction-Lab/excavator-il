import numpy as np

from excavator_il.collector.camera import UvcCamera
from excavator_il.collector.config import CameraConfig


class _Encoded:
    def tobytes(self):
        return b"jpeg-data"


class _Capture:
    def __init__(self):
        self.settings = []
        self.released = False

    def isOpened(self):
        return True

    def set(self, key, value):
        self.settings.append((key, value))
        return True

    def read(self):
        return True, np.zeros((24, 32, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class _Cv2:
    CAP_PROP_FRAME_WIDTH = 1
    CAP_PROP_FRAME_HEIGHT = 2
    CAP_PROP_FPS = 3
    IMWRITE_JPEG_QUALITY = 4
    COLOR_BGR2RGB = 5

    def __init__(self):
        self.capture = _Capture()

    def VideoCapture(self, device):
        assert device == "/dev/video0"
        return self.capture

    @staticmethod
    def imencode(extension, frame, params):
        assert extension == ".jpg"
        assert frame.shape == (24, 32, 3)
        assert params == [4, 95]
        return True, _Encoded()

    @staticmethod
    def cvtColor(frame, conversion):
        assert conversion == 5
        return frame[:, :, ::-1]


def test_uvc_camera_stamps_completed_capture_and_encodes_jpeg():
    cv2 = _Cv2()
    camera = UvcCamera(
        CameraConfig(
            device="/dev/video0",
            width=32,
            height=24,
            nominal_fps=30,
            jpeg_quality=95,
        ),
        monotonic_ns=lambda: 123_456,
        cv2_module=cv2,
    )

    frame = camera.read_encoded()
    camera.close()

    assert frame.capture_monotonic_ns == 123_456
    assert frame.encoded_image == b"jpeg-data"
    assert frame.extension == "jpg"
    assert cv2.capture.released is True


def test_uvc_camera_returns_timestamped_rgb_for_live_policy():
    cv2 = _Cv2()
    cv2.capture.read = lambda: (
        True,
        np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8),
    )
    camera = UvcCamera(
        CameraConfig(
            device="/dev/video0",
            width=2,
            height=1,
            nominal_fps=30,
            jpeg_quality=95,
        ),
        monotonic_ns=lambda: 987_654,
        cv2_module=cv2,
    )

    frame = camera.read_rgb()

    assert frame.capture_monotonic_ns == 987_654
    assert frame.rgb.dtype == np.uint8
    assert frame.rgb.tolist() == [[[3, 2, 1], [6, 5, 4]]]
