from dataclasses import dataclass
from typing import Dict, List


@dataclass
class BoundingBox:
    """Represents a bounding box"""
    xtl: float
    ytl: float
    xbr: float
    ybr: float
    frame: int
    occluded: bool
    outside: bool
    keyframe: bool
    z_order: int
    attributes: Dict[str, str]
    rotation: float = 0.0


@dataclass
class Track:
    """Represents a track (object across multiple frames)"""
    id: int
    label: str
    source: str
    group_id: int
    boxes: Dict[int, 'BoundingBox']  # frame_number -> BoundingBox
    shape_type: str = "box"  # box, polygon, polyline, points, etc.


class SegmentData:
    """Holds data from a single segment"""
    def __init__(self, segment_name: str, start_frame: int, stop_frame: int):
        self.segment_name = segment_name
        self.start_frame = start_frame
        self.stop_frame = stop_frame
        self.tracks: List[Track] = []
        self.shapes: List = []  # For non-track annotations
        self.tags: List = []
        self.images: List[str] = []
        self.meta: Dict = {}

