from pathlib import Path
import xml.etree.ElementTree as ET
from .models import BoundingBox, Track, SegmentData


def parse_box_from_xml(box_elem: ET.Element, frame: int) -> BoundingBox:
    """Parse a box element from XML"""
    return BoundingBox(
        xtl=float(box_elem.get('xtl', 0)),
        ytl=float(box_elem.get('ytl', 0)),
        xbr=float(box_elem.get('xbr', 0)),
        ybr=float(box_elem.get('ybr', 0)),
        frame=frame,
        occluded=box_elem.get('occluded', '0') == '1',
        outside=box_elem.get('outside', '0') == '1',
        keyframe=box_elem.get('keyframe', '1') == '1',
        z_order=int(box_elem.get('z_order', 0)),
        rotation=float(box_elem.get('rotation', 0)),
        attributes={attr.get('name'): attr.text or ''
                   for attr in box_elem.findall('attribute')}
    )


def parse_annotation_xml(xml_path: Path) -> SegmentData:
    """Parse a CVAT annotation XML file"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    meta_elem = root.find('meta')
    job_elem = meta_elem.find('job') if meta_elem is not None else None

    segment_id = "unknown"
    start_frame = 0
    stop_frame = 0

    if job_elem is not None:
        id_elem = job_elem.find('id')
        if id_elem is not None and id_elem.text:
            segment_id = id_elem.text

        segments_elem = job_elem.find('segments')
        if segments_elem is not None:
            segment_elem = segments_elem.find('segment')
            if segment_elem is not None:
                start_elem = segment_elem.find('start')
                stop_elem = segment_elem.find('stop')
                if start_elem is not None and start_elem.text:
                    start_frame = int(start_elem.text)
                if stop_elem is not None and stop_elem.text:
                    stop_frame = int(stop_elem.text)

    segment_data = SegmentData(segment_id, start_frame, stop_frame)

    for track_elem in root.findall('track'):
        track = Track(
            id=int(track_elem.get('id', 0)),
            label=track_elem.get('label', ''),
            source=track_elem.get('source', 'manual'),
            group_id=int(track_elem.get('group_id', 0)),
            boxes={}
        )

        for box_elem in track_elem.findall('box'):
            frame = int(box_elem.get('frame'))
            box = parse_box_from_xml(box_elem, frame)
            track.boxes[frame] = box

        if track.boxes:  # Only add tracks with boxes
            segment_data.tracks.append(track)

    meta_start_frame = segment_data.start_frame
    meta_stop_frame = segment_data.stop_frame

    if segment_data.tracks:
        all_frames = []
        for track in segment_data.tracks:
            all_frames.extend(track.boxes.keys())
        
        if all_frames:
            true_start = min(all_frames)
            true_stop = max(all_frames)
            
            if (meta_start_frame != 0 or meta_stop_frame != 0) and \
               (meta_start_frame != true_start or meta_stop_frame != true_stop):
                print(f"    ⚠️  WARNING: XML meta ({meta_start_frame}-{meta_stop_frame}) mismatches")
                print(f"                 track data ({true_start}-{true_stop}). Using track data.")

            segment_data.start_frame = true_start
            segment_data.stop_frame = true_stop
        
    segment_data.meta = {
        'task_name': segment_id,
        'start_frame': segment_data.start_frame,
        'stop_frame': segment_data.stop_frame,
    }

    print(f"    📄 Parsed: {len(segment_data.tracks)} tracks, frames {segment_data.start_frame}-{segment_data.stop_frame}")
    if segment_data.tracks:
        label_counts = {}
        for track in segment_data.tracks:
            label_counts[track.label] = label_counts.get(track.label, 0) + 1
        print(f"       Labels: {dict(label_counts)}")

    return segment_data

