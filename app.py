import os
import shutil
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

TEST = 1
DATA = {}
STATS = {}

def traverse_folder(folder_path: Path | str):
    for task_folder in os.listdir(folder_path):
        xml_file_path = folder_path / task_folder / 'annotations.xml'
        if xml_file_path.is_file():
            parse_xml_file(xml_file_path)
        else:
            print(f'XML Not Found: {folder_path}' )

def parse_main_xml_file(file_path: Path | str):
    global DATA
    file_path = Path(file_path)
    print(f'Parsing Main XML: {file_path.name}')

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    tree = ET.parse(file_path)
    root = tree.getroot()

    for track in root.findall("track"):
        track_id = int(track.get("id"))

        boxes = "\n".join(ET.tostring(box, encoding="unicode") for box in track.findall("box"))

        DATA[track_id] = boxes

    print(f"Parsed {len(DATA)} tracks from {file_path.name}")

def parse_xml_file(file_path: Path | str):
    global DATA, STATS
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    tree = ET.parse(file_path)
    root = tree.getroot()

    unmatched_count = 0
    total_count = 0
    update_count = 0
    box_count = 0

    for track in root.findall("track"):
        total_count += 1
        track_id = int(track.get("id"))
        boxes = track.findall("box")
        box_count += len(boxes)
        boxes_raw = "\n".join(ET.tostring(box, encoding="unicode") for box in boxes)

        is_Found = False
        for stored_id, stored_boxes in DATA.items():
            if boxes_raw.strip() in stored_boxes.strip():
                track.set("id", str(stored_id))
                update_count += 1
                is_Found = True
                break  

        if is_Found == False:
            unmatched_count += 1
        
    tree.write(file_path, encoding="utf-8", xml_declaration=True)
    STATS['unmatched_count'] = STATS.get('unmatched_count', 0) + unmatched_count
    STATS['update_count'] = STATS.get('update_count', 0) + update_count
    STATS['total_count'] = STATS.get('total_count', 0) + total_count

    print(f"Filepath: {file_path.parent.name}\nUnmatched Tracks: {unmatched_count} | Updated Tracks: {update_count} | Total Tracks: {total_count} | Total Boxes: {box_count}\n")

def extract_zip(zip_path: Path, destination_folder: Path):
    print(f"-> Extracting '{zip_path.name}' to '{destination_folder}'")
    try:
        os.makedirs(destination_folder, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(destination_folder)
        
        print(f"   ... Success.")
            
    except zipfile.BadZipFile:
        print(f"   ... Error: Failed to extract. File may be corrupt: {zip_path.name}")
    except Exception as e:
        print(f"   ... An unexpected error occurred: {e}")

def extract_zips(source_folder: Path | str, destination_folder: Path | str):
    source_path = Path(source_folder)
    dest_path = Path(destination_folder)

    if not source_path.is_dir():
        print(f"Error: Source folder not found: {source_path}")
        return

    print(f"Starting extraction from '{source_path}' to '{dest_path}'\n")

    zip_files = list(source_path.glob('*.zip'))

    if not zip_files:
        print("No .zip files found in the source folder.")
        return

    for zip_path in zip_files:
        output_folder = dest_path / zip_path.stem
        
        extract_zip(zip_path, output_folder)
        print("-" * 20) 

def zip_folder(folder_to_zip: Path, zip_file_path: Path):
    print(f"-> Zipping '{folder_to_zip.name}' to '{zip_file_path}'")
    try:
        base_name = zip_file_path.with_suffix('')
        shutil.make_archive(
            base_name=base_name,
            format='zip',
            root_dir=folder_to_zip
        )
        print(f"   ... Success.")
        
    except Exception as e:
        print(f"   ... An error occurred while zipping {folder_to_zip.name}: {e}")

def zip_folders(source_folder: Path | str, destination_folder: Path | str):
    source_path = Path(source_folder)
    dest_path = Path(destination_folder)

    if not source_path.is_dir():
        print(f"Error: Source folder not found: {source_path}")
        return

    os.makedirs(dest_path, exist_ok=True)
    
    print(f"Starting to zip folders from '{source_path}' to '{dest_path}'\n")

    folders_found = 0
    for item in source_path.iterdir():
        if item.is_dir():
            folders_found += 1
            
            output_zip_path = dest_path / (item.name + '.zip')
            
            zip_folder(item, output_zip_path)
            print("-" * 20) 

    if folders_found == 0:
        print("No sub-folders found in the source directory.")
    else:
        print(f"\nZipping complete. {folders_found} folders processed.")

def copy_images_to_flat_folder(source_folder: Path | str, destination_folder: Path | str):

    IMAGE_EXTENSIONS = {
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg'
    }

    src_path = Path(source_folder)
    dest_path = Path(destination_folder)

    if not src_path.is_dir():
        print(f"Error: Source folder not found: {src_path}")
        return

    os.makedirs(dest_path, exist_ok=True)
    
    print(f"Scanning for images in: {src_path}")
    
    copied_count = 0
    skipped_count = 0

    for file_path in src_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            
            dest_file_path = dest_path / file_path.name
            
            if not dest_file_path.exists():
                shutil.copy2(file_path, dest_file_path)
            else:
                counter = 1
                new_dest_path = dest_path / f"{file_path.stem}_{counter}{file_path.suffix}"
                
                while new_dest_path.exists():
                    counter += 1
                    new_dest_path = dest_path / f"{file_path.stem}_{counter}{file_path.suffix}"
                
                shutil.copy2(file_path, new_dest_path)
                print(f"  - Renamed collision: {file_path.name} -> {new_dest_path.name}")
            
            copied_count += 1
        
        elif file_path.is_file():
            skipped_count += 1

    print("\n" + "="*30)
    print("      Copy Complete")
    print(f"  Total Images Copied: {copied_count}")
    print(f"  Non-Image Files Skipped: {skipped_count}")
    print(f"  Destination: {dest_path.resolve()}")
    print("="*30)

def main():
    # # zips extract
    zip_folder = Path(r'data\zips')
    extract_folder = Path(r'test\extracted_zips')
    extract_zips(zip_folder, extract_folder)

    # # parse main xml
    # main_xml = Path(r'test\CVAT Main task without Images\annotations.xml')
    # parse_main_xml_file(main_xml)

    # # replace task ids
    # traverse_folder(extract_folder)

    # processed_zips = Path(r'test\processed_zips')
    # zip_folders(extract_folder, processed_zips)
    # print(STATS)

    # merged_images = Path(r'test\merged_images')
    # copy_images_to_flat_folder(extract_folder, merged_images)



if __name__ == '__main__':
    main()
