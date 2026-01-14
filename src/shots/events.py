import json
from pathlib import Path
import supervision as sv
from sports.basketball import ShotEventTracker


def write_shot_events(
    output_path: Path,
    video_path: Path,
    frame_generator,
    model,
    confidence: float,
    iou: float,
    jump_shot_id: int,
    layup_dunk_id: int,
    ball_in_basket_id: int,
) -> None:
    video_info = sv.VideoInfo.from_video_path(video_path)
    tracker = ShotEventTracker(
        reset_time_frames=int(video_info.fps * 1.7),
        minimum_frames_between_starts=int(video_info.fps * 0.5),
        cooldown_frames_after_made=int(video_info.fps * 0.5),
    )
    with output_path.open("w", encoding="utf-8") as handle:
        for frame_index, frame in enumerate(frame_generator):
            result = model.infer(frame, confidence=confidence, iou_threshold=iou)[0]
            detections = sv.Detections.from_inference(result)

            has_jump_shot = len(detections[detections.class_id == jump_shot_id]) > 0
            has_layup_dunk = len(detections[detections.class_id == layup_dunk_id]) > 0
            has_ball_in_basket = len(detections[detections.class_id == ball_in_basket_id]) > 0

            events = tracker.update(
                frame_index=frame_index,
                has_jump_shot=has_jump_shot,
                has_layup_dunk=has_layup_dunk,
                has_ball_in_basket=has_ball_in_basket,
            )
            if events:
                handle.write(json.dumps({"frame": frame_index, "events": events}) + "\n")
