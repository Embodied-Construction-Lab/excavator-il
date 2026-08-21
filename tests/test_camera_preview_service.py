import logging

from excavator_il.camera_preview_service import CameraPreviewService
from excavator_il.collector.preview import LatestJpegFrame


def test_camera_preview_service_publishes_without_opening_serial():
    frames = LatestJpegFrame()
    holder = {}

    class _Frame:
        capture_monotonic_ns = 123
        encoded_image = b"preview"

    class _Camera:
        def read_encoded(self):
            holder["service"].request_stop()
            return _Frame()

    class _Server:
        def serve_forever(self):
            return None

        def close(self):
            return None

    service = CameraPreviewService(
        camera=_Camera(),
        frames=frames,
        server=_Server(),
        ready_message="camera preview ready: test",
    )
    holder["service"] = service

    service.run()

    published = frames.wait_after(0, timeout_s=0.01)
    assert published is not None
    assert published.encoded_image == b"preview"


def test_camera_preview_service_reports_ready_after_first_frame(caplog):
    frames = LatestJpegFrame()
    holder = {}

    class _Frame:
        capture_monotonic_ns = 123
        encoded_image = b"preview"

    class _Camera:
        def read_encoded(self):
            holder["service"].request_stop()
            return _Frame()

    class _Server:
        def serve_forever(self):
            return None

        def close(self):
            return None

    service = CameraPreviewService(
        camera=_Camera(),
        frames=frames,
        server=_Server(),
        ready_message="camera preview ready: test",
    )
    holder["service"] = service

    with caplog.at_level(logging.INFO, logger="excavator_il.camera_preview"):
        service.run()

    assert "camera preview ready: test" in caplog.text
