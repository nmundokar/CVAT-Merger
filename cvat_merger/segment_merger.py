from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import shutil
import time
import sys
from .models import SegmentData, Track
from .xml_parser import parse_annotation_xml
from .file_utils import extract_segment_zip, find_images_directory, copy_images_to_output, create_images_zip
from .track_matching import match_tracks_with_overlap
from .xml_generator import generate_merged_xml


class SegmentMerger:
    """Handles merging of CVAT annotation segments"""
    
    def __init__(self, overlap: int = 5):
        self.overlap = overlap
    
    def extract_and_parse_segments(self, zip_files: List[Path], temp_dir: Path) -> List[SegmentData]:
        """Extract ZIP files and parse all segments"""
        print("\n📂 Extracting segments...")
        segments_data = []
        
        for i, zip_file in enumerate(zip_files):
            print(f"  [{i+1}/{len(zip_files)}] {zip_file.name}")
            xml_path, images = extract_segment_zip(zip_file, temp_dir)
            segment_data = parse_annotation_xml(xml_path)
            segment_data.images = images
            # Store the extracted directory name (ZIP stem) for path reconstruction
            segment_data.extracted_dir_name = zip_file.stem
            segments_data.append(segment_data)
        
        print(f"✅ Extracted {len(segments_data)} segments")
        return segments_data
    
    def copy_all_images(self, segments_data: List[SegmentData], 
                       output_images_dir: Path, temp_dir: Path) -> set:
        """Copy all images from segments to output directory"""
        all_image_files = set()
        
        print(f"\n📸 Copying images from {len(segments_data)} segments...")
        
        for seg_idx, segment_data in enumerate(segments_data):
            print(f"  Segment {seg_idx + 1}: {segment_data.segment_name}")
            print(f"    Found {len(segment_data.images)} images in segment list")
            
            # Find the XML path using the extracted directory name (ZIP stem)
            if hasattr(segment_data, 'extracted_dir_name') and segment_data.extracted_dir_name:
                extracted_dir = temp_dir / segment_data.extracted_dir_name
            else:
                # Fallback: try using segment_name (old behavior)
                extracted_dir = temp_dir / segment_data.segment_name
            
            xml_path = extracted_dir / 'annotations.xml'
            if not xml_path.exists():
                xml_files = list(extracted_dir.rglob('*.xml'))
                if xml_files:
                    xml_path = xml_files[0]
            
            if not xml_path.exists():
                print(f"    ⚠️  Warning: Could not find XML path for segment {segment_data.segment_name}")
                print(f"    Tried: {xml_path}")
                print(f"    Extracted dir: {extracted_dir}")
                continue
            
            images_src_dir = find_images_directory(xml_path)
            print(f"    Looking for images in: {images_src_dir}")
            print(f"    Images directory exists: {images_src_dir.exists()}")
            
            if images_src_dir.exists():
                actual_files = list(images_src_dir.glob('*.*'))
                print(f"    Found {len(actual_files)} files in images directory")
            
            copy_images_to_output(images_src_dir, segment_data.images, 
                                output_images_dir, all_image_files)
        
        print(f"\n📸 Total: Copied {len(all_image_files)} unique images to {output_images_dir}")
        return all_image_files
    
    def sort_segments(self, segments_data: List[SegmentData]) -> List[SegmentData]:
        """Sort segments by start frame and validate"""
        print("\n🔢 Sorting segments by start frame...")
        segments_data.sort(key=lambda s: s.start_frame)
        
        for i, seg in enumerate(segments_data):
            print(f"  Segment {i+1}: frames {seg.start_frame}-{seg.stop_frame} ({seg.segment_name})")
        
        # Check for issues
        for i in range(1, len(segments_data)):
            prev_seg = segments_data[i-1]
            curr_seg = segments_data[i]
            expected_overlap = self.overlap
            actual_overlap = prev_seg.stop_frame - curr_seg.start_frame + 1
            
            if actual_overlap != expected_overlap:
                print(f"  ⚠️  WARNING: Segment {i} and {i+1} have {actual_overlap} frame overlap (expected {expected_overlap})")
        
        return segments_data
    
    def merge_tracks(self, segments_data: List[SegmentData]) -> List[Track]:
        """Merge tracks across segments"""
        print("\n🔗 Merging tracks across segments...")
        merged_tracks = []
        next_track_id = 0
        track_id_mapping = {}
        
        for seg_idx, segment in enumerate(segments_data):
            print(f"  Processing segment {seg_idx + 1}/{len(segments_data)}")
            
            if seg_idx == 0:
                # First segment: assign new IDs to all tracks
                merged_tracks, next_track_id, track_id_mapping[seg_idx] = \
                    self._process_first_segment(segment, next_track_id)
            else:
                # Match with previous segment
                merged_tracks, next_track_id, track_id_mapping[seg_idx] = \
                    self._process_subsequent_segment(
                        segment, segments_data[seg_idx - 1], 
                        merged_tracks, next_track_id, seg_idx
                    )
        
        print(f"✅ Merged into {len(merged_tracks)} unique tracks")
        return merged_tracks
    
    def _process_first_segment(self, segment: SegmentData, 
                               next_track_id: int) -> tuple:
        """Process the first segment (no matching needed)"""
        merged_tracks = []
        track_id_mapping = {}
        
        for track in segment.tracks:
            new_track = Track(
                id=next_track_id,
                label=track.label,
                source=track.source,
                group_id=track.group_id,
                boxes=track.boxes.copy()
            )
            merged_tracks.append(new_track)
            track_id_mapping[track.id] = next_track_id
            next_track_id += 1
        
        return merged_tracks, next_track_id, track_id_mapping
    
    def _process_subsequent_segment(self, segment: SegmentData,
                                    prev_segment: SegmentData,
                                    merged_tracks: List[Track],
                                    next_track_id: int,
                                    seg_idx: int) -> tuple:
        """Process a subsequent segment by matching with previous"""
        overlap_start = segment.start_frame
        overlap_end = min(overlap_start + self.overlap - 1, segment.stop_frame)
        
        # Get tracks from previous segment that exist in overlap
        prev_tracks_in_overlap = [
            t for t in merged_tracks
            if any(frame >= overlap_start and frame <= overlap_end
                  for frame in t.boxes.keys())
        ]
        
        self._debug_overlap_info(segment, prev_tracks_in_overlap, 
                                overlap_start, overlap_end)
        
        # Match tracks
        matches = match_tracks_with_overlap(
            prev_tracks_in_overlap,
            segment.tracks,
            overlap_start,
            overlap_end,
            debug=True
        )
        
        print(f"    Matched {len(matches)} tracks with previous segment")
        self._debug_matches(matches, segment.tracks, prev_tracks_in_overlap)
        
        track_id_mapping = {}
        
        for curr_idx, track in enumerate(segment.tracks):
            if curr_idx in matches:
                # Matched track - merge with existing
                prev_idx = matches[curr_idx]
                matched_track = prev_tracks_in_overlap[prev_idx]
                
                # Add boxes from current segment (skip overlap frames)
                for frame, box in track.boxes.items():
                    if frame > overlap_end:
                        matched_track.boxes[frame] = box
                
                track_id_mapping[track.id] = matched_track.id
            else:
                # New track - assign new ID
                new_track = Track(
                    id=next_track_id,
                    label=track.label,
                    source=track.source,
                    group_id=track.group_id,
                    boxes=track.boxes.copy()
                )
                merged_tracks.append(new_track)
                track_id_mapping[track.id] = next_track_id
                next_track_id += 1
        
        return merged_tracks, next_track_id, track_id_mapping
    
    def _debug_overlap_info(self, segment: SegmentData, 
                           prev_tracks_in_overlap: List[Track],
                           overlap_start: int, overlap_end: int):
        """Print debug information about overlap"""
        print(f"    DEBUG: Overlap region = frames {overlap_start}-{overlap_end}")
        print(f"    DEBUG: Found {len(prev_tracks_in_overlap)} tracks from previous segment in overlap")
        print(f"    DEBUG: Current segment has {len(segment.tracks)} tracks")
        
        print(f"    Current segment tracks in overlap:")
        for i, track in enumerate(segment.tracks):
            overlap_frames = [f for f in track.boxes.keys() if overlap_start <= f <= overlap_end]
            if overlap_frames or i < 3:
                sample_frame = overlap_frames[0] if overlap_frames else list(track.boxes.keys())[0] if track.boxes else None
                box_info = ""
                if sample_frame and sample_frame in track.boxes:
                    b = track.boxes[sample_frame]
                    box_info = f" box@{sample_frame}: ({b.xtl:.1f},{b.ytl:.1f})-({b.xbr:.1f},{b.ybr:.1f})"
                print(f"      [{i}] ID={track.id}, label='{track.label}', {len(overlap_frames)} in overlap, total_frames={len(track.boxes)}{box_info}")
        
        print(f"    Previous segment tracks in overlap:")
        for i, track in enumerate(prev_tracks_in_overlap):
            overlap_frames = [f for f in track.boxes.keys() if overlap_start <= f <= overlap_end]
            if overlap_frames or i < 3:
                sample_frame = overlap_frames[0] if overlap_frames else list(track.boxes.keys())[0] if track.boxes else None
                box_info = ""
                if sample_frame and sample_frame in track.boxes:
                    b = track.boxes[sample_frame]
                    box_info = f" box@{sample_frame}: ({b.xtl:.1f},{b.ytl:.1f})-({b.xbr:.1f},{b.ybr:.1f})"
                print(f"      [{i}] ID={track.id}, label='{track.label}', {len(overlap_frames)} in overlap, total_frames={len(track.boxes)}{box_info}")
    
    def _debug_matches(self, matches: Dict[int, int], 
                      curr_tracks: List[Track], 
                      prev_tracks: List[Track]):
        """Print debug information about matches"""
        for curr_idx, prev_idx in matches.items():
            curr_track = curr_tracks[curr_idx]
            prev_track = prev_tracks[prev_idx]
            print(f"      MATCH: Curr track {curr_idx} (label '{curr_track.label}') -> Prev track {prev_idx} (label '{prev_track.label}')")
    
    def print_statistics(self, segments_data: List[SegmentData], 
                        merged_tracks: List[Track], 
                        all_image_files: set):
        """Print merge statistics"""
        print("\n📊 Merge Statistics:")
        print(f"  Total segments: {len(segments_data)}")
        print(f"  Total frames: {segments_data[0].start_frame} to {segments_data[-1].stop_frame}")
        print(f"  Total tracks: {len(merged_tracks)}")
        print(f"  Total images: {len(all_image_files)}")
        print(f"  Overlap: {self.overlap} frames")
        
        total_boxes = sum(len(track.boxes) for track in merged_tracks)
        print(f"  Total annotations: {total_boxes}")
        
        label_counts = defaultdict(int)
        for track in merged_tracks:
            label_counts[track.label] += 1
        
        print("\n  Label distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"    {label}: {count} tracks")
    
    def merge_segments(self, segments_folder: Path, output_dir: Path, create_images_zip: bool = True):
        """Main function to merge all segments"""
        print(f"🔍 Scanning folder: {segments_folder}")
        
        zip_files = sorted(segments_folder.glob('*.zip'))
        if not zip_files:
            print(f"❌ No ZIP files found in {segments_folder}")
            return
        
        print(f"📦 Found {len(zip_files)} segment files")
        
        temp_dir = output_dir / '_temp_extraction'
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        output_images_dir = output_dir / 'images'
        output_images_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            segments_data = self.extract_and_parse_segments(zip_files, temp_dir)
            all_image_files = self.copy_all_images(segments_data, output_images_dir, temp_dir)
            
            if create_images_zip:
                print("\n📦 Creating images.zip...")
                # Verify images directory exists and has files before creating ZIP
                if output_images_dir.exists():
                    file_count = len([f for f in output_images_dir.iterdir() if f.is_file()])
                    print(f"  Verifying: {file_count} files in {output_images_dir.name}/ directory")
                else:
                    print(f"  ⚠️  Warning: Images directory does not exist: {output_images_dir}")
                
                from .file_utils import create_images_zip as create_zip_func
                create_zip_func(output_images_dir, output_dir)
            else:
                print("\n⏭️  Skipping images.zip creation (--images=False)")
            
            segments_data = self.sort_segments(segments_data)
            
            merged_tracks = self.merge_tracks(segments_data)
            
            print("\n📝 Generating merged annotation XML...")
            output_xml_path = output_dir / 'annotations.xml'
            generate_merged_xml(segments_data, merged_tracks, output_xml_path)
            print(f"✅ Saved to: {output_xml_path}")
            
            self.print_statistics(segments_data, merged_tracks, all_image_files)
        
        finally:
            print("\n🧹 Cleaning up temporary files...")
            if temp_dir.exists():
                # On Windows, files might be locked - retry with delays
                max_retries = 5
                retry_delay = 1.0  # Start with 1 second delay
                
                for attempt in range(max_retries):
                    try:
                        shutil.rmtree(temp_dir)
                        print(f"✅ Cleaned up temporary extraction directory")
                        break
                    except (PermissionError, OSError) as e:
                        if attempt < max_retries - 1:
                            # Wait and retry with increasing delay
                            print(f"  Retrying cleanup (attempt {attempt + 1}/{max_retries})...")
                            time.sleep(retry_delay)
                            retry_delay *= 1.5  # Exponential backoff
                        else:
                            # Last attempt failed
                            print(f"⚠️  Warning: Could not delete temporary directory: {e}")
                            print(f"   Temporary files remain at: {temp_dir}")
                            if sys.platform == 'win32':
                                print(f"   This is often caused by file handles still being open.")
                                print(f"   You may need to manually delete this folder later.")
                    except Exception as e:
                        print(f"⚠️  Warning: Unexpected error during cleanup: {e}")
                        print(f"   Temporary files remain at: {temp_dir}")
                        break

