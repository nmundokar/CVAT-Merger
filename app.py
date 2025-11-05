import os
import sys
import zipfile
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET
from xml.dom import minidom
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
import numpy as np


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
    boxes: Dict[int, BoundingBox]  # frame_number -> BoundingBox
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
    # Must have same label
    if track1.label != track2.label:
        if debug:
            print(f"        Different labels: '{track1.label}' vs '{track2.label}'")
        return 0.0

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
            if box1.outside != box2.outside:
                # One is outside, one is visible - not a match
                frame_count += 1
                ious.append(0.0)
            elif not box1.outside and not box2.outside:
                # Both visible - calculate IoU
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

    if debug and avg_similarity > 0.3:  # Only show promising matches
        print(f"        Labels: '{track1.label}' vs '{track2.label}'")
        print(f"        IoUs by frame: {[f'{x:.3f}' for x in ious[:5]]}")
        print(f"        Average IoU: {avg_similarity:.3f}")

    return avg_similarity


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

    # Extract segment info from meta
    meta_elem = root.find('meta')
    job_elem = meta_elem.find('job') if meta_elem is not None else None

    segment_id = "unknown"
    start_frame = 0
    stop_frame = 0

    if job_elem is not None:
        id_elem = job_elem.find('id')
        if id_elem is not None and id_elem.text:
            segment_id = id_elem.text

        # Try to get segment info
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

    # CRITICAL: If start/stop not in XML, infer from track data
    if start_frame == 0 and stop_frame == 0:
        # Will be updated after parsing tracks
        needs_inference = True
    else:
        needs_inference = False

    segment_data = SegmentData(segment_id, start_frame, stop_frame)
    segment_data._needs_frame_inference = needs_inference

    # Parse tracks
    for track_elem in root.findall('track'):
        track = Track(
            id=int(track_elem.get('id', 0)),
            label=track_elem.get('label', ''),
            source=track_elem.get('source', 'manual'),
            group_id=int(track_elem.get('group_id', 0)),
            boxes={}
        )

        # Parse all boxes in the track
        for box_elem in track_elem.findall('box'):
            frame = int(box_elem.get('frame'))
            box = parse_box_from_xml(box_elem, frame)
            track.boxes[frame] = box

        if track.boxes:  # Only add tracks with boxes
            segment_data.tracks.append(track)

    # If frame range wasn't in XML, infer from actual track data
    if segment_data._needs_frame_inference and segment_data.tracks:
        all_frames = []
        for track in segment_data.tracks:
            all_frames.extend(track.boxes.keys())

        if all_frames:
            segment_data.start_frame = min(all_frames)
            segment_data.stop_frame = max(all_frames)
            print(f"    ⚠️  WARNING: No segment metadata in XML, inferred frames {segment_data.start_frame}-{segment_data.stop_frame} from track data")

    # Store meta for later use
    segment_data.meta = {
        'task_name': segment_id,
        'start_frame': segment_data.start_frame,
        'stop_frame': segment_data.stop_frame,
    }

    # DEBUG: Print what we parsed
    print(f"    📄 Parsed: {len(segment_data.tracks)} tracks, frames {segment_data.start_frame}-{segment_data.stop_frame}")
    if segment_data.tracks:
        label_counts = {}
        for track in segment_data.tracks:
            label_counts[track.label] = label_counts.get(track.label, 0) + 1
        print(f"       Labels: {dict(label_counts)}")

    return segment_data


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


def extract_segment_zip(zip_path: Path, extract_dir: Path) -> Tuple[Path, List[str]]:
    """Extract a segment ZIP file and return paths to XML and images"""
    segment_dir = extract_dir / zip_path.stem
    segment_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(segment_dir)

    # Find annotation.xml
    xml_files = list(segment_dir.rglob('annotations.xml'))
    if not xml_files:
        xml_files = list(segment_dir.rglob('*.xml'))

    if not xml_files:
        raise FileNotFoundError(f"No XML file found in {zip_path}")

    xml_path = xml_files[0]

    # Find images
    images_dir = segment_dir / 'images'
    if not images_dir.exists():
        # Try to find images folder
        img_dirs = [d for d in segment_dir.rglob('*') if d.is_dir() and 'image' in d.name.lower()]
        if img_dirs:
            images_dir = img_dirs[0]

    image_files = []
    if images_dir.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend([str(f.name) for f in images_dir.glob(ext)])

    return xml_path, sorted(image_files)


def merge_segments(segments_folder: Path, output_dir: Path, overlap: int = 5):
    """Main function to merge all segments"""
    print(f"🔍 Scanning folder: {segments_folder}")

    # Find all ZIP files
    zip_files = sorted(segments_folder.glob('*.zip'))
    if not zip_files:
        print(f"❌ No ZIP files found in {segments_folder}")
        return

    print(f"📦 Found {len(zip_files)} segment files")

    # Create temporary extraction directory
    temp_dir = output_dir / '_temp_extraction'
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Create output directories
    output_images_dir = output_dir / 'images'
    output_images_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Extract and parse all segments
        print("\n📂 Extracting segments...")
        segments_data = []
        all_image_files = set()

        for i, zip_file in enumerate(zip_files):
            print(f"  [{i+1}/{len(zip_files)}] {zip_file.name}")
            xml_path, images = extract_segment_zip(zip_file, temp_dir)
            segment_data = parse_annotation_xml(xml_path)
            segment_data.images = images

            # OPTION: If your XMLs have relative frames (all start at 0),
            # calculate the offset based on segment position
            # Uncomment these lines if needed:
            # if segment_data.start_frame == 0 and i > 0:
            #     offset = i * (200 - overlap)  # 200 = segment_size, 5 = overlap
            #     segment_data.start_frame = offset
            #     segment_data.stop_frame = offset + 199
            #     # Update all track frame numbers
            #     for track in segment_data.tracks:
            #         track.boxes = {(frame + offset): box for frame, box in track.boxes.items()}
            #     print(f"    ⚠️  Applied frame offset: +{offset} (now frames {segment_data.start_frame}-{segment_data.stop_frame})")

            segments_data.append(segment_data)

            # Copy images (avoiding duplicates)
            images_src_dir = xml_path.parent / 'images'
            if not images_src_dir.exists():
                # Try to find images folder
                img_dirs = [d for d in xml_path.parent.rglob('*')
                           if d.is_dir() and 'image' in d.name.lower()]
                if img_dirs:
                    images_src_dir = img_dirs[0]

            if images_src_dir.exists():
                for img_file in images:
                    if img_file not in all_image_files:
                        src = images_src_dir / img_file
                        dst = output_images_dir / img_file
                        if src.exists():
                            shutil.copy2(src, dst)
                            all_image_files.add(img_file)

        print(f"✅ Extracted {len(segments_data)} segments")
        print(f"📸 Copied {len(all_image_files)} unique images")

        # Step 1.5: Create images.zip
        print("\n📦 Creating images.zip...")
        images_zip_path = output_dir / 'images.zip'

        try:
            with zipfile.ZipFile(images_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                all_images = sorted([f for f in output_images_dir.iterdir() if f.is_file()])
                total_images = len(all_images)

                for idx, image_file in enumerate(all_images):
                    # Store images in the root of the ZIP (not in 'images/' subfolder)
                    zipf.write(image_file, image_file.name)

                    # Show progress every 1000 images or at milestones
                    if (idx + 1) % 1000 == 0 or (idx + 1) == total_images:
                        print(f"  [{idx + 1}/{total_images}] images compressed")

            # Get ZIP file size
            zip_size_mb = images_zip_path.stat().st_size / (1024 * 1024)
            print(f"✅ Created images.zip ({zip_size_mb:.2f} MB)")

            # Remove the images folder to save space
            print(f"🧹 Removing temporary images folder...")
            shutil.rmtree(output_images_dir)
            print(f"✅ Cleaned up images folder")

        except Exception as e:
            print(f"⚠️  Warning: Could not create images.zip: {e}")
            print(f"   Images are still available in: {output_images_dir}")

        # CRITICAL: Sort segments by start_frame to ensure correct order
        print("\n🔢 Sorting segments by start frame...")
        segments_data_unsorted = segments_data.copy()
        segments_data.sort(key=lambda s: s.start_frame)

        for i, seg in enumerate(segments_data):
            print(f"  Segment {i+1}: frames {seg.start_frame}-{seg.stop_frame} ({seg.segment_name})")

        # Check for issues
        for i in range(1, len(segments_data)):
            prev_seg = segments_data[i-1]
            curr_seg = segments_data[i]
            expected_overlap = overlap
            actual_overlap = prev_seg.stop_frame - curr_seg.start_frame + 1

            if actual_overlap != expected_overlap:
                print(f"  ⚠️  WARNING: Segment {i} and {i+1} have {actual_overlap} frame overlap (expected {expected_overlap})")

        # Step 2: Merge tracks across segments
        print("\n🔗 Merging tracks across segments...")
        merged_tracks = []
        next_track_id = 0
        track_id_mapping = {}  # segment_idx -> {old_id -> new_id}

        for seg_idx, segment in enumerate(segments_data):
            print(f"  Processing segment {seg_idx + 1}/{len(segments_data)}")

            if seg_idx == 0:
                # First segment: assign new IDs to all tracks
                track_id_mapping[seg_idx] = {}
                for track in segment.tracks:
                    new_track = Track(
                        id=next_track_id,
                        label=track.label,
                        source=track.source,
                        group_id=track.group_id,
                        boxes=track.boxes.copy()
                    )
                    merged_tracks.append(new_track)
                    track_id_mapping[seg_idx][track.id] = next_track_id
                    next_track_id += 1
            else:
                # Match with previous segment using overlap
                prev_segment = segments_data[seg_idx - 1]
                overlap_start = segment.start_frame
                overlap_end = min(overlap_start + overlap - 1, segment.stop_frame)

                # Get tracks from previous segment that exist in overlap
                prev_tracks_in_overlap = [
                    t for t in merged_tracks
                    if any(frame >= overlap_start and frame <= overlap_end
                          for frame in t.boxes.keys())
                ]

                # DEBUG: Print overlap info
                print(f"    DEBUG: Overlap region = frames {overlap_start}-{overlap_end}")
                print(f"    DEBUG: Found {len(prev_tracks_in_overlap)} tracks from previous segment in overlap")
                print(f"    DEBUG: Current segment has {len(segment.tracks)} tracks")

                # DEBUG: Show track details for current segment
                print(f"    Current segment tracks in overlap:")
                for i, track in enumerate(segment.tracks):
                    overlap_frames = [f for f in track.boxes.keys() if overlap_start <= f <= overlap_end]
                    if overlap_frames or i < 3:  # Show all with overlap + first 3
                        sample_frame = overlap_frames[0] if overlap_frames else list(track.boxes.keys())[0] if track.boxes else None
                        box_info = ""
                        if sample_frame and sample_frame in track.boxes:
                            b = track.boxes[sample_frame]
                            box_info = f" box@{sample_frame}: ({b.xtl:.1f},{b.ytl:.1f})-({b.xbr:.1f},{b.ybr:.1f})"
                        print(f"      [{i}] ID={track.id}, label='{track.label}', {len(overlap_frames)} in overlap, total_frames={len(track.boxes)}{box_info}")

                print(f"    Previous segment tracks in overlap:")
                for i, track in enumerate(prev_tracks_in_overlap):
                    overlap_frames = [f for f in track.boxes.keys() if overlap_start <= f <= overlap_end]
                    if overlap_frames or i < 3:  # Show all with overlap + first 3
                        sample_frame = overlap_frames[0] if overlap_frames else list(track.boxes.keys())[0] if track.boxes else None
                        box_info = ""
                        if sample_frame and sample_frame in track.boxes:
                            b = track.boxes[sample_frame]
                            box_info = f" box@{sample_frame}: ({b.xtl:.1f},{b.ytl:.1f})-({b.xbr:.1f},{b.ybr:.1f})"
                        print(f"      [{i}] ID={track.id}, label='{track.label}', {len(overlap_frames)} in overlap, total_frames={len(track.boxes)}{box_info}")

                # Match tracks
                matches = match_tracks_with_overlap(
                    prev_tracks_in_overlap,
                    segment.tracks,
                    overlap_start,
                    overlap_end,
                    debug=True  # Enable debug output
                )

                print(f"    Matched {len(matches)} tracks with previous segment")

                # DEBUG: Show what matched
                for curr_idx, prev_idx in matches.items():
                    curr_track = segment.tracks[curr_idx]
                    prev_track = prev_tracks_in_overlap[prev_idx]
                    print(f"      MATCH: Curr track {curr_idx} (label '{curr_track.label}') -> Prev track {prev_idx} (label '{prev_track.label}')")

                track_id_mapping[seg_idx] = {}

                for curr_idx, track in enumerate(segment.tracks):
                    if curr_idx in matches:
                        # Matched track - merge with existing
                        prev_idx = matches[curr_idx]
                        matched_track = prev_tracks_in_overlap[prev_idx]

                        # Add boxes from current segment (skip overlap frames)
                        for frame, box in track.boxes.items():
                            if frame > overlap_end:
                                matched_track.boxes[frame] = box

                        track_id_mapping[seg_idx][track.id] = matched_track.id
                    else:
                        # New track - assign new ID
                        new_track = Track(
                            id=next_track_id,
                            label=track.label,
                            source=track.source,
                            group_id=track.group_id,
                            boxes={f: b for f, b in track.boxes.items()
                                  if f > overlap_end}  # Skip overlap
                        )
                        if new_track.boxes:  # Only add if has boxes after overlap
                            merged_tracks.append(new_track)
                            track_id_mapping[seg_idx][track.id] = next_track_id
                            next_track_id += 1

        print(f"✅ Merged into {len(merged_tracks)} unique tracks")

        # Step 3: Generate merged XML
        print("\n📝 Generating merged annotation XML...")
        output_xml_path = output_dir / 'annotations.xml'
        generate_merged_xml(segments_data, merged_tracks, output_xml_path)
        print(f"✅ Saved to: {output_xml_path}")

        # Step 4: Generate statistics
        print("\n📊 Merge Statistics:")
        print(f"  Total segments: {len(segments_data)}")
        print(f"  Total frames: {segments_data[0].start_frame} to {segments_data[-1].stop_frame}")
        print(f"  Total tracks: {len(merged_tracks)}")
        print(f"  Total images: {len(all_image_files)}")
        print(f"  Overlap: {overlap} frames")

        # Track statistics
        total_boxes = sum(len(track.boxes) for track in merged_tracks)
        print(f"  Total annotations: {total_boxes}")

        # Label distribution
        label_counts = defaultdict(int)
        for track in merged_tracks:
            label_counts[track.label] += 1

        print("\n  Label distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"    {label}: {count} tracks")

    finally:
        # Cleanup temporary directory
        print("\n🧹 Cleaning up temporary files...")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


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


def main():
    if len(sys.argv) < 2:
        print("Usage: python temp.py <segments_folder_path>")
        print("\nExample:")
        print("  python temp.py /path/to/segments")
        print("\nThe script will:")
        print("  1. Extract all ZIP files in the folder")
        print("  2. Parse annotations from each segment")
        print("  3. Match tracks using 5-frame overlap")
        print("  4. Generate merged annotations.xml and copy all images")
        sys.exit(1)

    segments_folder = Path(sys.argv[1])

    if not segments_folder.exists():
        print(f"❌ Error: Folder not found: {segments_folder}")
        sys.exit(1)

    # Create output directory
    output_dir = Path.cwd() / 'merged_output'
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("CVAT Segment Merger")
    print("=" * 60)

    try:
        merge_segments(segments_folder, output_dir, overlap=5)
        print("\n" + "=" * 60)
        print("✅ SUCCESS! Merge completed.")
        print("=" * 60)
        print(f"\n📁 Output location: {output_dir.absolute()}")
        print(f"   - annotations.xml (merged annotations)")

        # Check if images.zip exists
        images_zip = output_dir / 'images.zip'
        if images_zip.exists():
            zip_size_mb = images_zip.stat().st_size / (1024 * 1024)
            # Count files in ZIP
            try:
                with zipfile.ZipFile(images_zip, 'r') as zipf:
                    image_count = len(zipf.namelist())
                print(f"   - images.zip ({image_count} images, {zip_size_mb:.2f} MB)")
            except:
                print(f"   - images.zip ({zip_size_mb:.2f} MB)")

        # Check if images folder still exists (in case ZIP creation failed)
        images_folder = output_dir / 'images'
        if images_folder.exists():
            image_count = len([f for f in images_folder.iterdir() if f.is_file()])
            print(f"   - images/ (folder with {image_count} images)")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()