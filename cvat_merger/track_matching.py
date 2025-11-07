from typing import Dict, List
import numpy as np
from scipy.optimize import linear_sum_assignment
from .models import BoundingBox, Track


def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Calculate Intersection over Union for two bounding boxes"""
    # Calculate intersection area
    x_left = max(box1.xtl, box2.xtl)
    y_top = max(box1.ytl, box2.ytl)
    x_right = min(box1.xbr, box2.xbr)
    y_bottom = min(box1.ybr, box2.ybr)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union area
    box1_area = (box1.xbr - box1.xtl) * (box1.ybr - box1.ytl)
    box2_area = (box2.xbr - box2.xtl) * (box2.ybr - box2.ytl)
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def calculate_track_similarity(track1: Track, track2: Track,
                               overlap_start: int, overlap_end: int,
                               debug: bool = False) -> float:
    """
    Calculate similarity between two tracks in the overlap region.
    Returns a score from 0.0 (no match) to 1.0 (perfect match).
    """
    label_mismatch = track1.label != track2.label
    
    if label_mismatch:
        if debug:
            print(f"        Different labels: '{track1.label}' vs '{track2.label}' - still checking overlap for debugging")

    # Get boxes in overlap region
    overlap_frames = range(overlap_start, overlap_end + 1)

    total_similarity = 0.0
    frame_count = 0
    ious = []

    for frame in overlap_frames:
        box1 = track1.boxes.get(frame)
        box2 = track2.boxes.get(frame)

        if box1 and box2:
            # Both tracks have boxes in this frame
            iou = calculate_iou(box1, box2)
            total_similarity += iou
            frame_count += 1
            ious.append(iou)
                
        elif box1 or box2:
            # Only one track has a box - penalize
            frame_count += 1
            ious.append(0.0)

    if frame_count == 0:
        return 0.0

    avg_similarity = total_similarity / frame_count

    if debug and avg_similarity > 0.3:
        label_info = f" (LABEL MISMATCH: '{track1.label}' vs '{track2.label}')" if label_mismatch else ""
        print(f"        Labels: '{track1.label}' vs '{track2.label}'{label_info}")
        print(f"        IoUs by frame: {[f'{x:.3f}' for x in ious[:5]]}")
        print(f"        Average IoU: {avg_similarity:.3f}")

    return avg_similarity


def match_tracks_with_overlap(prev_tracks: List[Track],
                              curr_tracks: List[Track],
                              overlap_start: int,
                              overlap_end: int,
                              threshold: float = 0.5,
                              debug: bool = False) -> Dict[int, int]:
    """
    Match tracks between segments using Hungarian algorithm.
    Returns mapping: curr_track_idx -> prev_track_idx
    """
    if not prev_tracks or not curr_tracks:
        return {}

    # Build cost matrix
    cost_matrix = np.ones((len(curr_tracks), len(prev_tracks)))

    for i, curr_track in enumerate(curr_tracks):
        for j, prev_track in enumerate(prev_tracks):
            if debug and i < 3 and j < 3:  # Only debug first few
                print(f"      Comparing curr[{i}] ('{curr_track.label}') with prev[{j}] ('{prev_track.label}'):")
            similarity = calculate_track_similarity(
                prev_track, curr_track, overlap_start, overlap_end,
                debug=(debug and i < 3 and j < 3)
            )
            # Convert similarity to cost (1 - similarity)
            cost_matrix[i, j] = 1.0 - similarity
            if debug and i < 3 and j < 3:
                print(f"        Similarity: {similarity:.3f}, Cost: {cost_matrix[i, j]:.3f}")

    # Use Hungarian algorithm to find optimal matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Only keep matches above threshold
    matches = {}
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] <= (1.0 - threshold):
            matches[i] = j

    return matches

