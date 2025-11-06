import sys
from pathlib import Path
import zipfile
from cvat_merger import SegmentMerger


def parse_args():
    """Parse command line arguments"""
    if len(sys.argv) < 2:
        print("Usage: python app.py <segments_folder_path> [--images=True/False]")
        print("\nExample:")
        print("  python app.py /path/to/segments")
        print("  python app.py /path/to/segments --images=False")
        print("\nThe script will:")
        print("  1. Extract all ZIP files in the folder")
        print("  2. Parse annotations from each segment")
        print("  3. Match tracks using 5-frame overlap")
        print("  4. Generate merged annotations.xml and copy all images")
        print("\nOptions:")
        print("  --images=True/False  Create images.zip (default: True)")
        sys.exit(1)

    segments_folder = Path(sys.argv[1])
    create_images_zip = True  # Default value

    # Parse --images flag
    for arg in sys.argv[2:]:
        if arg.startswith('--images='):
            value = arg.split('=', 1)[1].lower()
            create_images_zip = value in ('true', '1', 'yes')
        elif arg == '--images':
            # Handle --images flag without value (defaults to True)
            create_images_zip = True

    return segments_folder, create_images_zip


def main():
    segments_folder, create_images_zip = parse_args()

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
        merger = SegmentMerger(overlap=5)
        merger.merge_segments(segments_folder, output_dir, create_images_zip=create_images_zip)
        
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