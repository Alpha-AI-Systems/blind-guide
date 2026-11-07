"""Test output parsing without downloading YOLO weights or requiring a GPU."""

import unittest
from types import SimpleNamespace

import numpy as np

from fast_path.object_detection.detector import YOLODetector


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeModel:
    def __init__(self, boxes):
        self.boxes = boxes
        self.kwargs = None

    def predict(self, **kwargs):
        self.kwargs = kwargs
        return [SimpleNamespace(boxes=self.boxes, names={0: "person", 1: "bicycle"})]


class YOLODetectorTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_converts_detections_to_plain_python_data(self):
        boxes = SimpleNamespace(
            xyxy=FakeTensor([[10.4, 20.1, 110.2, 220.3], [200, 100, 310, 250]]),
            conf=FakeTensor([0.948, 0.83]),
            cls=FakeTensor([0, 1]),
        )
        fake_model = FakeModel(boxes)
        detector = YOLODetector(model=fake_model)

        result = detector.predict(self.frame)

        self.assertEqual((result.width, result.height), (640, 480))
        self.assertEqual(len(result.detections), 2)
        self.assertEqual(result.detections[0].class_name, "person")
        self.assertEqual(result.detections[0].bbox_xyxy, (10, 20, 110, 220))
        self.assertEqual(result.detections[0].center_xy, (60, 120))
        self.assertEqual(result.detections[0].confidence, 0.948)
        self.assertEqual(fake_model.kwargs["imgsz"], 640)
        self.assertNotIn("device", fake_model.kwargs)
        self.assertEqual(result.to_dict()["detections"][1]["class_name"], "bicycle")

    def test_empty_detection(self):
        boxes = SimpleNamespace(
            xyxy=FakeTensor([]), conf=FakeTensor([]), cls=FakeTensor([])
        )
        result = YOLODetector(model=FakeModel(boxes)).predict(self.frame)
        self.assertEqual(result.detections, ())

    def test_missing_boxes(self):
        result = YOLODetector(model=FakeModel(None)).predict(self.frame)
        self.assertEqual(result.detections, ())

    def test_boxes_are_clamped_and_invalid_boxes_removed(self):
        boxes = SimpleNamespace(
            xyxy=FakeTensor([[-10, -5, 700, 490], [50, 30, 50, 40]]),
            conf=FakeTensor([0.9, 0.7]),
            cls=FakeTensor([0, 1]),
        )
        result = YOLODetector(model=FakeModel(boxes)).predict(self.frame)
        self.assertEqual(len(result.detections), 1)
        self.assertEqual(result.detections[0].bbox_xyxy, (0, 0, 640, 480))

    def test_rejects_invalid_settings_and_frame(self):
        fake = FakeModel(None)
        with self.assertRaises(ValueError):
            YOLODetector(confidence=0, model=fake)
        with self.assertRaises(ValueError):
            YOLODetector(image_size=0, model=fake)
        with self.assertRaises(ValueError):
            YOLODetector(model=fake).predict(None)
        with self.assertRaises(ValueError):
            YOLODetector(model=fake).predict(np.zeros((10, 10), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
