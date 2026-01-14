import supervision as sv
from inference import get_model


def load_model(model_id: str):
    return get_model(model_id=model_id)


def infer_frame(model, frame, confidence: float, iou: float, class_agnostic_nms: bool = False) -> sv.Detections:
    result = model.infer(frame, confidence=confidence, iou_threshold=iou, class_agnostic_nms=class_agnostic_nms)[0]
    return sv.Detections.from_inference(result)
