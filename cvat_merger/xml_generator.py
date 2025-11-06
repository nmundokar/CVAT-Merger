from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List
from .models import SegmentData, Track


def generate_merged_xml(segments: List[SegmentData],
                        tracks: List[Track],
                        output_path: Path):
    """Generate merged CVAT annotation XML"""
    root = ET.Element('annotations')

    # Add version
    version = ET.SubElement(root, 'version')
    version.text = '1.1'

    # Add meta
    meta = ET.SubElement(root, 'meta')
    task = ET.SubElement(meta, 'task')

    task_id = ET.SubElement(task, 'id')
    task_id.text = '1'

    task_name = ET.SubElement(task, 'name')
    task_name.text = 'merged_task'

    task_size = ET.SubElement(task, 'size')
    total_frames = segments[-1].stop_frame + 1 if segments else 0
    task_size.text = str(total_frames)

    mode = ET.SubElement(task, 'mode')
    mode.text = 'interpolation'

    overlap_elem = ET.SubElement(task, 'overlap')
    overlap_elem.text = '5'

    # Add segments info
    segments_elem = ET.SubElement(task, 'segments')
    for seg in segments:
        segment_elem = ET.SubElement(segments_elem, 'segment')
        seg_id = ET.SubElement(segment_elem, 'id')
        seg_id.text = str(segments.index(seg))
        seg_start = ET.SubElement(segment_elem, 'start')
        seg_start.text = str(seg.start_frame)
        seg_stop = ET.SubElement(segment_elem, 'stop')
        seg_stop.text = str(seg.stop_frame)

    # Collect all unique labels
    labels = set(track.label for track in tracks)
    labels_elem = ET.SubElement(task, 'labels')
    for label in sorted(labels):
        label_elem = ET.SubElement(labels_elem, 'label')
        label_name = ET.SubElement(label_elem, 'name')
        label_name.text = label

    # Add tracks
    for track in sorted(tracks, key=lambda t: t.id):
        track_elem = ET.SubElement(root, 'track')
        track_elem.set('id', str(track.id))
        track_elem.set('label', track.label)
        track_elem.set('source', track.source)
        if track.group_id:
            track_elem.set('group_id', str(track.group_id))

        # Add boxes sorted by frame
        for frame in sorted(track.boxes.keys()):
            box = track.boxes[frame]
            box_elem = ET.SubElement(track_elem, 'box')
            box_elem.set('frame', str(frame))
            box_elem.set('xtl', f'{box.xtl:.2f}')
            box_elem.set('ytl', f'{box.ytl:.2f}')
            box_elem.set('xbr', f'{box.xbr:.2f}')
            box_elem.set('ybr', f'{box.ybr:.2f}')
            box_elem.set('outside', '1' if box.outside else '0')
            box_elem.set('occluded', '1' if box.occluded else '0')
            box_elem.set('keyframe', '1' if box.keyframe else '0')
            box_elem.set('z_order', str(box.z_order))
            if box.rotation:
                box_elem.set('rotation', f'{box.rotation:.2f}')

            # Add attributes
            for attr_name, attr_value in box.attributes.items():
                attr_elem = ET.SubElement(box_elem, 'attribute')
                attr_elem.set('name', attr_name)
                attr_elem.text = attr_value

    # Pretty print XML
    xml_str = ET.tostring(root, encoding='unicode')
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent='  ')

    # Remove empty lines
    lines = [line for line in pretty_xml.split('\n') if line.strip()]
    pretty_xml = '\n'.join(lines)

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)

