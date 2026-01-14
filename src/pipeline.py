from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import supervision as sv
from tqdm import tqdm

from sports import clean_paths, ConsecutiveValueTracker, TeamClassifier, MeasurementUnit, ViewTransformer
from sports.basketball import CourtConfiguration, League, draw_court, draw_points_on_court, ShotEventTracker

from src.config import load_config, project_root, resolve_path
from src.utils.env import load_env
from src.utils.paths import ensure_dir
from src.detection.rf_detr import load_model, infer_frame
from src.tracking.sam2_tracker import build_predictor, SAM2Tracker


def load_rosters(rosters_path: Path) -> Dict[str, Dict[str, str]]:
    import yaml

    with rosters_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data.get("teams", {})


def output_path(outputs_dir: Path, source_video: Path, tag: str) -> Path:
    return outputs_dir / tag / f"{source_video.stem}-{tag}{source_video.suffix}"


def filter_players(detections: sv.Detections, player_class_ids) -> sv.Detections:
    return detections[np.isin(detections.class_id, player_class_ids)]


def run_detection(cfg: dict, paths: dict) -> None:
    model = load_model(cfg["models"]["player_detection_id"])
    conf = cfg["thresholds"]["player_confidence"]
    iou = cfg["thresholds"]["player_iou"]
    colors = sv.ColorPalette.from_hex(cfg["teams"]["palette"])

    box_annotator = sv.BoxAnnotator(color=colors, thickness=2)
    label_annotator = sv.LabelAnnotator(color=colors, text_color=sv.Color.BLACK)

    source_video = paths["source_video"]
    target = output_path(paths["outputs_dir"], source_video, "detection")
    ensure_dir(target.parent)

    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        detections = infer_frame(model, frame, confidence=conf, iou=iou)
        annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections)
        return annotated

    sv.process_video(source_path=source_video, target_path=target, callback=callback, show_progress=True)


def run_tracking(cfg: dict, paths: dict) -> None:
    model = load_model(cfg["models"]["player_detection_id"])
    conf = cfg["thresholds"]["player_confidence"]
    iou = cfg["thresholds"]["player_iou"]

    colors = sv.ColorPalette.from_hex(cfg["teams"]["palette"])
    mask_annotator = sv.MaskAnnotator(color=colors, color_lookup=sv.ColorLookup.TRACK, opacity=0.5)
    box_annotator = sv.BoxAnnotator(color=colors, color_lookup=sv.ColorLookup.TRACK, thickness=2)

    predictor = build_predictor(paths["sam2_config"], paths["sam2_checkpoint"])
    tracker = SAM2Tracker(predictor)

    source_video = paths["source_video"]
    frame_gen = sv.get_video_frames_generator(source_video)
    first_frame = next(frame_gen)

    detections = infer_frame(model, first_frame, confidence=conf, iou=iou)
    detections = filter_players(detections, cfg["classes"]["player_class_ids"])
    if len(detections) == 0:
        raise RuntimeError("No player detections found in the first frame.")

    detections.tracker_id = np.arange(1, len(detections.class_id) + 1)
    tracker.prompt_first_frame(first_frame, detections)

    target = output_path(paths["outputs_dir"], source_video, "mask")
    ensure_dir(target.parent)

    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        tracked = tracker.propagate(frame)
        annotated = mask_annotator.annotate(scene=frame.copy(), detections=tracked)
        annotated = box_annotator.annotate(scene=annotated, detections=tracked)
        return annotated

    sv.process_video(source_path=source_video, target_path=target, callback=callback, show_progress=True)


def run_teams(cfg: dict, paths: dict) -> None:
    model = load_model(cfg["models"]["player_detection_id"])
    conf = cfg["thresholds"]["player_confidence"]
    iou = cfg["thresholds"]["player_iou"]

    team_names = {int(k): v for k, v in cfg["teams"]["names"].items()}
    team_colors = cfg["teams"]["colors"]

    colors = sv.ColorPalette.from_hex([
        team_colors[team_names[0]],
        team_colors[team_names[1]],
    ])

    mask_annotator = sv.MaskAnnotator(color=colors, opacity=0.5, color_lookup=sv.ColorLookup.INDEX)
    box_annotator = sv.BoxAnnotator(color=colors, thickness=2, color_lookup=sv.ColorLookup.INDEX)

    predictor = build_predictor(paths["sam2_config"], paths["sam2_checkpoint"])
    tracker = SAM2Tracker(predictor)

    source_video = paths["source_video"]
    frame_gen = sv.get_video_frames_generator(source_video)
    first_frame = next(frame_gen)

    detections = infer_frame(model, first_frame, confidence=conf, iou=iou)
    detections = filter_players(detections, cfg["classes"]["player_class_ids"])
    if len(detections) == 0:
        raise RuntimeError("No player detections found in the first frame.")

    detections.tracker_id = np.arange(1, len(detections.class_id) + 1)

    boxes = sv.scale_boxes(xyxy=detections.xyxy, factor=0.4)
    crops = [sv.crop_image(first_frame, box) for box in boxes]
    team_classifier = TeamClassifier(device=cfg.get("device", "cuda"))
    team_classifier.fit(crops)
    teams = np.array(team_classifier.predict(crops))

    tracker.prompt_first_frame(first_frame, detections)

    target = output_path(paths["outputs_dir"], source_video, "teams")
    ensure_dir(target.parent)

    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        tracked = tracker.propagate(frame)
        annotated = mask_annotator.annotate(
            scene=frame.copy(),
            detections=tracked,
            custom_color_lookup=teams[tracked.tracker_id - 1],
        )
        annotated = box_annotator.annotate(
            scene=annotated,
            detections=tracked,
            custom_color_lookup=teams[tracked.tracker_id - 1],
        )
        return annotated

    sv.process_video(source_path=source_video, target_path=target, callback=callback, show_progress=True)


def run_ocr(cfg: dict, paths: dict) -> None:
    model = load_model(cfg["models"]["player_detection_id"])
    number_model = load_model(cfg["models"]["number_recognition_id"])

    conf = cfg["thresholds"]["player_confidence"]
    iou = cfg["thresholds"]["player_iou"]
    ocr_interval = cfg["thresholds"]["ocr_interval"]
    mask_threshold = cfg["thresholds"]["mask_ios_threshold"]

    number_class_id = cfg["classes"]["number_class_id"]
    player_class_ids = cfg["classes"]["player_class_ids"]

    colors = sv.ColorPalette.from_hex(cfg["teams"]["palette"])
    mask_annotator = sv.MaskAnnotator(color=colors, color_lookup=sv.ColorLookup.TRACK, opacity=0.7)
    box_annotator = sv.BoxAnnotator(color=colors, color_lookup=sv.ColorLookup.TRACK, thickness=2)
    label_annotator = sv.LabelAnnotator(color=colors, color_lookup=sv.ColorLookup.TRACK, text_color=sv.Color.BLACK, text_scale=0.8)

    predictor = build_predictor(paths["sam2_config"], paths["sam2_checkpoint"])
    tracker = SAM2Tracker(predictor)

    team_names = {int(k): v for k, v in cfg["teams"]["names"].items()}
    rosters = load_rosters(paths["rosters"])
    number_validator = ConsecutiveValueTracker(n_consecutive=cfg["validation"]["number_consecutive"])
    team_validator = ConsecutiveValueTracker(n_consecutive=cfg["validation"]["team_consecutive"])

    source_video = paths["source_video"]
    frame_gen = sv.get_video_frames_generator(source_video)
    first_frame = next(frame_gen)

    detections = infer_frame(model, first_frame, confidence=conf, iou=iou)
    detections = filter_players(detections, player_class_ids)
    if len(detections) == 0:
        raise RuntimeError("No player detections found in the first frame.")

    detections.tracker_id = np.arange(1, len(detections.class_id) + 1)

    boxes = sv.scale_boxes(xyxy=detections.xyxy, factor=0.4)
    crops = [sv.crop_image(first_frame, box) for box in boxes]
    team_classifier = TeamClassifier(device=cfg.get("device", "cuda"))
    team_classifier.fit(crops)
    teams = np.array(team_classifier.predict(crops))
    team_validator.update(tracker_ids=detections.tracker_id, values=teams)

    tracker.prompt_first_frame(first_frame, detections)

    target = output_path(paths["outputs_dir"], source_video, "validated-numbers")
    ensure_dir(target.parent)

    def coords_above_threshold(matrix: np.ndarray, threshold: float, sort_desc: bool = True):
        values = np.asarray(matrix)
        rows, cols = np.where(values > threshold)
        pairs = list(zip(rows.tolist(), cols.tolist()))
        if sort_desc:
            pairs.sort(key=lambda rc: values[rc[0], rc[1]], reverse=True)
        return pairs

    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        player_detections = tracker.propagate(frame)

        if index % ocr_interval == 0:
            frame_h, frame_w, *_ = frame.shape
            result = model.infer(frame, confidence=conf, iou_threshold=iou)[0]
            number_detections = sv.Detections.from_inference(result)
            number_detections = number_detections[number_detections.class_id == number_class_id]
            number_detections.mask = sv.xyxy_to_mask(
                boxes=number_detections.xyxy,
                resolution_wh=(frame_w, frame_h),
            )

            number_crops = [
                sv.crop_image(frame, xyxy)
                for xyxy
                in sv.clip_boxes(
                    sv.pad_boxes(xyxy=number_detections.xyxy, px=10, py=10),
                    (frame_w, frame_h),
                )
            ]
            numbers = [
                number_model.predict(number_crop, "Read the number.")[0]
                for number_crop in number_crops
            ]

            iou_matrix = sv.mask_iou_batch(
                masks_true=player_detections.mask,
                masks_detection=number_detections.mask,
                overlap_metric=sv.OverlapMetric.IOS,
            )
            pairs = coords_above_threshold(iou_matrix, mask_threshold)
            if pairs:
                player_idx, number_idx = zip(*pairs)
                player_idx = [i + 1 for i in player_idx]
                number_idx = list(number_idx)
                matched_numbers = [numbers[int(i)] for i in number_idx]
                number_validator.update(tracker_ids=player_idx, values=matched_numbers)

        annotated = mask_annotator.annotate(scene=frame.copy(), detections=player_detections)
        annotated = box_annotator.annotate(scene=annotated, detections=player_detections)
        numbers = number_validator.get_validated(tracker_ids=player_detections.tracker_id)
        teams = team_validator.get_validated(tracker_ids=player_detections.tracker_id)
        labels = []
        for number, team in zip(numbers, teams):
            if number is None or team is None:
                labels.append("")
                continue
            team_name = team_names.get(int(team))
            player_name = rosters.get(team_name, {}).get(str(number), "")
            label = f\"#{number} {player_name}\".strip()
            labels.append(label)

        annotated = label_annotator.annotate(scene=annotated, detections=player_detections, labels=labels)
        return annotated

    sv.process_video(source_path=source_video, target_path=target, callback=callback, show_progress=True)


def run_court_mapping(cfg: dict, paths: dict) -> None:
    player_model = load_model(cfg["models"]["player_detection_id"])
    keypoint_model = load_model(cfg["models"]["keypoint_detection_id"])

    conf = cfg["thresholds"]["player_confidence"]
    iou = cfg["thresholds"]["player_iou"]
    keypoint_conf = cfg["thresholds"]["keypoint_confidence"]
    keypoint_anchor = cfg["thresholds"]["keypoint_anchor_confidence"]

    team_names = {int(k): v for k, v in cfg["teams"]["names"].items()}
    team_colors = cfg["teams"]["colors"]

    predictor = build_predictor(paths["sam2_config"], paths["sam2_checkpoint"])
    tracker = SAM2Tracker(predictor)

    source_video = paths["source_video"]
    frame_gen = sv.get_video_frames_generator(source_video)
    first_frame = next(frame_gen)

    detections = infer_frame(player_model, first_frame, confidence=conf, iou=iou)
    detections = filter_players(detections, cfg["classes"]["player_class_ids"])
    if len(detections) == 0:
        raise RuntimeError("No player detections found in the first frame.")

    detections.tracker_id = np.arange(1, len(detections.class_id) + 1)
    boxes = sv.scale_boxes(xyxy=detections.xyxy, factor=0.4)
    crops = [sv.crop_image(first_frame, box) for box in boxes]
    team_classifier = TeamClassifier(device=cfg.get("device", "cuda"))
    team_classifier.fit(crops)
    teams = np.array(team_classifier.predict(crops))

    tracker.prompt_first_frame(first_frame, detections)

    config = CourtConfiguration(league=League.NBA, measurement_unit=MeasurementUnit.FEET)
    video_xy = []

    for frame in tqdm(frame_gen, desc="Mapping frames"):
        player_detections = tracker.propagate(frame)
        result = keypoint_model.infer(frame, confidence=keypoint_conf)[0]
        key_points = sv.KeyPoints.from_inference(result)
        landmarks_mask = key_points.confidence[0] > keypoint_anchor
        if np.count_nonzero(landmarks_mask) < 4:
            continue

        court_landmarks = np.array(config.vertices)[landmarks_mask]
        frame_landmarks = key_points[:, landmarks_mask].xy[0]
        transformer = ViewTransformer(source=frame_landmarks, target=court_landmarks)
        frame_xy = player_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        court_xy = transformer.transform_points(points=frame_xy)
        video_xy.append(court_xy)

    if not video_xy:
        raise RuntimeError("No valid court mappings were produced.")

    video_xy = np.asarray(video_xy)
    cleaned_xy, _ = clean_paths(
        video_xy,
        jump_sigma=cfg["clean_paths"]["jump_sigma"],
        min_jump_dist=cfg["clean_paths"]["min_jump_dist"],
        max_jump_run=cfg["clean_paths"]["max_jump_run"],
        pad_around_runs=cfg["clean_paths"]["pad_around_runs"],
        smooth_window=cfg["clean_paths"]["smooth_window"],
        smooth_poly=cfg["clean_paths"]["smooth_poly"],
    )

    target = output_path(paths["outputs_dir"], source_video, "map")
    ensure_dir(target.parent)

    court = draw_court(config=config)
    court_h, court_w, _ = court.shape
    video_info = sv.VideoInfo.from_video_path(source_video)
    video_info.width = court_w
    video_info.height = court_h

    with sv.VideoSink(target, video_info) as sink:
        for frame_xy in tqdm(cleaned_xy, desc="Rendering court"):
            court = draw_court(config=config)
            court = draw_points_on_court(
                config=config,
                xy=frame_xy[teams == 0],
                fill_color=sv.Color.from_hex(team_colors[team_names[0]]),
                court=court,
            )
            court = draw_points_on_court(
                config=config,
                xy=frame_xy[teams == 1],
                fill_color=sv.Color.from_hex(team_colors[team_names[1]]),
                court=court,
            )
            sink.write_frame(court)


def run_shot_events(cfg: dict, paths: dict) -> None:
    player_model = load_model(cfg["models"]["player_detection_id"])
    conf = cfg["thresholds"]["player_confidence"]
    iou = cfg["thresholds"]["player_iou"]

    jump_shot_id = cfg["shot_detection"]["jump_shot_class_id"]
    layup_dunk_id = cfg["shot_detection"]["layup_dunk_class_id"]
    ball_in_basket_id = cfg["shot_detection"]["ball_in_basket_class_id"]

    source_video = paths["source_video"]
    frame_gen = sv.get_video_frames_generator(source_video)
    video_info = sv.VideoInfo.from_video_path(source_video)

    tracker = ShotEventTracker(
        reset_time_frames=int(video_info.fps * 1.7),
        minimum_frames_between_starts=int(video_info.fps * 0.5),
        cooldown_frames_after_made=int(video_info.fps * 0.5),
    )

    target_dir = paths["outputs_dir"] / "shots"
    ensure_dir(target_dir)
    events_path = target_dir / "shot_events.jsonl"

    with events_path.open("w", encoding="utf-8") as handle:
        for frame_index, frame in enumerate(frame_gen):
            detections = infer_frame(player_model, frame, confidence=conf, iou=iou)

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


def build_paths(cfg: dict, video_override: str | None) -> dict:
    root = project_root()
    source_video = resolve_path(cfg["paths"]["source_video"], root)
    if video_override:
        source_video = resolve_path(video_override, root)

    outputs_dir = resolve_path(cfg["paths"]["output_dir"], root)
    rosters = resolve_path(cfg["paths"]["rosters"], root)
    sam2_checkpoint = resolve_path(cfg["paths"]["sam2_checkpoint"], root)
    sam2_config = resolve_path(cfg["paths"]["sam2_config"], root)

    return {
        "root": root,
        "source_video": source_video,
        "outputs_dir": outputs_dir,
        "rosters": rosters,
        "sam2_checkpoint": sam2_checkpoint,
        "sam2_config": sam2_config,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--step", default="all", choices=["all", "detection", "tracking", "teams", "ocr", "court", "shots"])
    parser.add_argument("--video", default=None, help="Override source video path")
    args = parser.parse_args()

    load_env()
    cfg = load_config(args.config)
    paths = build_paths(cfg, args.video)
    ensure_dir(paths["outputs_dir"])

    if args.step in ("all", "detection"):
        run_detection(cfg, paths)
    if args.step in ("all", "tracking"):
        run_tracking(cfg, paths)
    if args.step in ("all", "teams"):
        run_teams(cfg, paths)
    if args.step in ("all", "ocr"):
        run_ocr(cfg, paths)
    if args.step in ("all", "court"):
        run_court_mapping(cfg, paths)
    if args.step in ("all", "shots"):
        run_shot_events(cfg, paths)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
