from pathlib import Path
import zipfile
import shutil
from typing import Tuple, List, Optional


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


def find_images_directory(xml_path: Path) -> Path:
    """Find the images directory relative to XML path"""
    images_dir = xml_path.parent / 'images'
    if not images_dir.exists():
        img_dirs = [d for d in xml_path.parent.rglob('*') 
                   if d.is_dir() and 'image' in d.name.lower()]
        if img_dirs:
            images_dir = img_dirs[0]
    return images_dir


def copy_images_to_output(images_src_dir: Path, images: List[str], 
                         output_images_dir: Path, all_image_files: set):
    """Copy images to output directory, avoiding duplicates"""
    if not images_src_dir.exists():
        return
    
    for img_file in images:
        if img_file not in all_image_files:
            src = images_src_dir / img_file
            dst = output_images_dir / img_file
            if src.exists():
                shutil.copy2(src, dst)
                all_image_files.add(img_file)


def create_images_zip(output_images_dir: Path, output_dir: Path) -> Optional[Path]:
    """Create images.zip from images directory"""
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
        
        return images_zip_path
    except Exception as e:
        print(f"⚠️  Warning: Could not create images.zip: {e}")
        print(f"   Images are still available in: {output_images_dir}")
        return None

