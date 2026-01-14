import numpy as np
import supervision as sv
from sports import TeamClassifier


def fit_team_classifier(frame, detections: sv.Detections, device: str = "cuda"):
    boxes = sv.scale_boxes(xyxy=detections.xyxy, factor=0.4)
    crops = [sv.crop_image(frame, box) for box in boxes]
    classifier = TeamClassifier(device=device)
    classifier.fit(crops)
    teams = np.array(classifier.predict(crops))
    return classifier, teams
