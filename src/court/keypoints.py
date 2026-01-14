import numpy as np
import supervision as sv
from sports import ViewTransformer


def detect_keypoints(model, frame, confidence: float) -> sv.KeyPoints:
    result = model.infer(frame, confidence=confidence)[0]
    return sv.KeyPoints.from_inference(result)


def build_transformer(key_points: sv.KeyPoints, court_vertices: np.ndarray, anchor_confidence: float):
    mask = key_points.confidence[0] > anchor_confidence
    if np.count_nonzero(mask) < 4:
        return None
    frame_landmarks = key_points[:, mask].xy[0]
    court_landmarks = court_vertices[mask]
    return ViewTransformer(source=frame_landmarks, target=court_landmarks)
