from typing import List, Tuple
import numpy as np
import supervision as sv


def coords_above_threshold(matrix: np.ndarray, threshold: float, sort_desc: bool = True) -> List[Tuple[int, int]]:
    values = np.asarray(matrix)
    rows, cols = np.where(values > threshold)
    pairs = list(zip(rows.tolist(), cols.tolist()))
    if sort_desc:
        pairs.sort(key=lambda rc: values[rc[0], rc[1]], reverse=True)
    return pairs


def match_numbers_to_players(player_detections: sv.Detections, number_detections: sv.Detections, numbers, threshold: float):
    iou = sv.mask_iou_batch(
        masks_true=player_detections.mask,
        masks_detection=number_detections.mask,
        overlap_metric=sv.OverlapMetric.IOS,
    )
    pairs = coords_above_threshold(iou, threshold)
    if not pairs:
        return []
    player_idx, number_idx = zip(*pairs)
    player_idx = [i + 1 for i in player_idx]
    number_idx = list(number_idx)
    matched_numbers = [numbers[int(i)] for i in number_idx]
    return list(zip(player_idx, matched_numbers))
