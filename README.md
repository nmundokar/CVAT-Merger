# CVAT Segment Merger

A Python script to merge multiple CVAT annotation segments into a single unified annotation file. This tool intelligently matches tracks across segments using overlap regions and the Hungarian algorithm for optimal track association.

## Features

- 🔗 **Automatic Track Matching**: Uses Intersection over Union (IoU) and the Hungarian algorithm to match tracks across segments
- 📦 **ZIP Processing**: Automatically extracts and processes multiple CVAT segment ZIP files
- 🖼️ **Image Handling**: Collects and compresses all images into a single ZIP file
- 🔍 **Robust Frame Detection**: Automatically calculates true frame ranges from track data, even when metadata is unreliable
- 📊 **Detailed Statistics**: Provides comprehensive merge statistics including track counts, label distributions, and annotation counts
- ⚠️ **Validation**: Warns about frame overlap mismatches and metadata inconsistencies

## Requirements

- Python 3.7 or higher
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
```bash
git clone https://github.com/nmundokar/CVAT-Merger.git
cd CVAT-Merger
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```bash
python app.py <segments_folder_path>
```

### Example

```bash
python app.py C:\path\to\segments
```

### Input Format

The script expects a folder containing CVAT segment ZIP files. Each ZIP file should contain:
- `annotations.xml` or any `.xml` file with CVAT annotations
- `images/` folder (or any folder containing images) with frame images

**Important**: Segments must have a 5-frame overlap between consecutive segments for proper track matching.

### Output

The script creates a `merged_output/` directory in the current working directory containing:

- `annotations.xml` - Merged CVAT annotation file with all tracks properly matched and assigned new IDs
- `images.zip` - All unique images from all segments compressed into a single ZIP file

## How It Works

1. **Extraction**: Extracts all ZIP files from the input folder
2. **Parsing**: Parses annotation XML files and extracts track data
3. **Frame Validation**: Calculates actual frame ranges from track data (more reliable than metadata)
4. **Sorting**: Sorts segments by start frame to ensure correct processing order
5. **Track Matching**: 
   - For the first segment, assigns new track IDs to all tracks
   - For subsequent segments, matches tracks with previous segments using:
     - IoU (Intersection over Union) calculation in overlap regions
     - Hungarian algorithm for optimal one-to-one matching
     - Label matching (only tracks with same label can be matched)
   - Merges matched tracks and assigns new IDs to unmatched tracks
6. **Output Generation**: Creates merged XML with all tracks and compresses images

## Track Matching Algorithm

The script uses a sophisticated matching algorithm:

1. **Overlap Detection**: Identifies frames where two consecutive segments overlap (default: 5 frames)
2. **IoU Calculation**: Calculates Intersection over Union for bounding boxes in overlapping frames
3. **Similarity Score**: Averages IoU scores across all overlapping frames (same label required)
4. **Hungarian Algorithm**: Uses optimal assignment to match tracks one-to-one
5. **Threshold Filtering**: Only matches with similarity ≥ 0.5 are considered valid matches

### Matching Criteria

- Tracks must have the same label
- Average IoU in overlap region must be ≥ 0.5
- Tracks with different `outside` states are not matched
- New tracks (no match found) are assigned new IDs

## Output Statistics

After merging, the script displays:

- Total number of segments processed
- Total frame range (start to end)
- Total number of unique tracks
- Total number of images
- Total number of annotations (bounding boxes)
- Label distribution (count of tracks per label)

## Troubleshooting

### Warning: Frame Overlap Mismatch

If you see a warning about frame overlap:
```
⚠️  WARNING: Segment X and Y have Z frame overlap (expected 5)
```

This indicates that segments don't have the expected 5-frame overlap. Ensure your segments are properly configured in CVAT.

### Warning: XML Meta Mismatch

If you see:
```
⚠️  WARNING: XML meta (start-stop) mismatches track data (start-stop)
```

The script automatically uses the correct frame range from track data. This is typically due to outdated task metadata in CVAT.

### No ZIP Files Found

Ensure your input folder contains `.zip` files with CVAT annotation segments.

## Technical Details

### Dependencies

- **numpy**: Matrix operations for cost calculation
- **scipy**: Hungarian algorithm implementation (`scipy.optimize.linear_sum_assignment`)

### Code Structure

The script is organized into focused functions:

- **Parsing**: `parse_annotation_xml()`, `parse_box_from_xml()`
- **Matching**: `calculate_iou()`, `calculate_track_similarity()`, `match_tracks_with_overlap()`
- **Merging**: `merge_segments()`, `merge_tracks_across_segments()`
- **Output**: `generate_merged_xml()`

## License

[Add your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Created by nmundokar

## Repository

https://github.com/nmundokar/CVAT-Merger.git

