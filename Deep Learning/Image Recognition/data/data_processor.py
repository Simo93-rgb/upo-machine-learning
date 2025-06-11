import os
import shutil
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

def get_workers_count():
    """Determines the number of worker threads based on CPU count."""
    return max(4, os.cpu_count() - 2)

def parse_label_file(file_path):
    """
    Parses a single label file and extracts class IDs and bounding box coordinates.
    Returns a list of (class_id, [coco_coords]) tuples.
    """
    annotations = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    # COCO format: [x_center, y_center, width, height] (normalized)
                    coco_coords = [float(p) for p in parts[1:5]]
                    annotations.append((class_id, coco_coords))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return annotations

def collect_labels_from_patch(patch_path, base_images_path):
    """
    Collects label information for all .txt files in a given patch directory
    and maps them to their corresponding image paths.
    Returns a list of (image_path, annotations) tuples.
    """
    patch_annotations = []
    for root, _, files in os.walk(patch_path):
        for file_name in files:
            if file_name.endswith('.txt'):
                label_file_path = os.path.join(root, file_name)
                annotations = parse_label_file(label_file_path)
                if annotations:
                    # Construct the corresponding image path
                    relative_path = os.path.relpath(label_file_path, patch_path)
                    image_name = file_name.replace('.txt', '.jpg') # Assuming .jpg images
                    
                    # Get the patch directory name (e.g., patch15_a)
                    patch_name = os.path.basename(patch_path)
                    
                    # Build the full image path including the patch directory
                    if relative_path == image_name:  # File is directly in patch root
                        full_image_path = os.path.join(base_images_path, patch_name, image_name)
                    else:  # File is in subdirectory
                        image_path_suffix = os.path.join(patch_name, os.path.dirname(relative_path), image_name)
                        full_image_path = os.path.join(base_images_path, image_path_suffix)
                    
                    if os.path.exists(full_image_path):
                        patch_annotations.append((full_image_path, annotations))
                    else:
                        print(f"Warning: Image not found for label file {label_file_path} at {full_image_path}")
    return patch_annotations

def discover_and_read_labels(labels_base_path, images_base_path):
    """
    Discovers all patchX_a/b directories and reads their label files in parallel.
    Returns a dictionary where keys are class IDs and values are lists of
    (image_path, bounding_box_coords) for that class.
    """
    print(f"Scanning labels in: {labels_base_path}")
    all_class_data = defaultdict(list)
    patch_dirs = []

    # Find all patchX_a and patchX_b directories
    for root, dirs, _ in os.walk(labels_base_path):
        for d in dirs:
            if d.startswith('patch') and ('_a' in d or '_b' in d):
                patch_dirs.append(os.path.join(root, d))
    
    if not patch_dirs:
        print(f"No patch directories found in {labels_base_path}. Please check the path and structure.")
        return all_class_data

    print(f"Found {len(patch_dirs)} patch directories. Processing with {get_workers_count()} workers.")

    with ThreadPoolExecutor(max_workers=get_workers_count()) as executor:
        future_to_patch = {
            executor.submit(collect_labels_from_patch, patch_dir, images_base_path): patch_dir
            for patch_dir in patch_dirs
        }
        for future in as_completed(future_to_patch):
            patch_dir = future_to_patch[future]
            try:
                patch_annotations = future.result()
                for image_path, annotations_for_image in patch_annotations:
                    for class_id, bbox_coords in annotations_for_image:
                        all_class_data[class_id].append((image_path, bbox_coords))
            except Exception as exc:
                print(f'{patch_dir} generated an exception: {exc}')
    
    print("Finished collecting all labels.")
    return all_class_data

def perform_subsampling(all_class_data, target_samples_per_class=None, total_target_images=None):
    """
    Performs subsampling on the collected class data.
    - If target_samples_per_class is set, tries to level each class to this amount.
    - If total_target_images is set, it will aim for this many images,
      maintaining relative class distribution while ensuring a minimum per class.
    
    Returns a set of image paths to be included in the subsample.
    """
    print("Starting subsampling...")
    sampled_images = set()
    class_counts_before = {cid: len(data) for cid, data in all_class_data.items()}
    print(f"Class distribution before subsampling: {class_counts_before}")

    if target_samples_per_class is not None:
        print(f"Targeting {target_samples_per_class} samples per class.")
        for class_id, data in all_class_data.items():
            if len(data) > target_samples_per_class:
                # Randomly sample if more than target
                sampled_class_data = random.sample(data, target_samples_per_class)
            else:
                # Use all available if less than or equal to target
                sampled_class_data = data
            
            for image_path, _ in sampled_class_data:
                sampled_images.add(image_path)
    elif total_target_images is not None:
        print(f"Targeting a total of approximately {total_target_images} images.")
        
        if not all_class_data:
            print("No class data to subsample.")
            return sampled_images

        # Calculate total annotations to determine proportions
        total_annotations = sum(len(data) for data in all_class_data.values())
        if total_annotations == 0:
            print("No annotations found for subsampling.")
            return sampled_images

        # Determine target images per class based on proportion
        target_images_per_class_proportional = {}
        for class_id, data in all_class_data.items():
            proportion = len(data) / total_annotations
            target_images_per_class_proportional[class_id] = int(proportion * total_target_images)
        
        # Ensure a minimum number of images per class (e.g., 5, or all if less than 5)
        # This helps to level up underrepresented classes.
        min_images_per_class = 5
        for class_id in all_class_data.keys():
            target_images_per_class_proportional[class_id] = max(
                min_images_per_class,
                target_images_per_class_proportional.get(class_id, 0)
            )
            # Cap at actual available samples
            target_images_per_class_proportional[class_id] = min(
                target_images_per_class_proportional[class_id],
                len(all_class_data[class_id])
            )

        print(f"Target images per class (proportional with leveling): {target_images_per_class_proportional}")

        # Perform sampling
        for class_id, data in all_class_data.items():
            num_to_sample = target_images_per_class_proportional.get(class_id, 0)
            if num_to_sample > 0:
                sampled_class_data = random.sample(data, num_to_sample)
                for image_path, _ in sampled_class_data:
                    sampled_images.add(image_path)
    else:
        print("No subsampling parameters provided. Returning all images.")
        for class_id, data in all_class_data.items():
            for image_path, _ in data:
                sampled_images.add(image_path)

    print(f"Subsampling complete. Selected {len(sampled_images)} unique images.")
    return sampled_images

def copy_image(source_path, destination_base_path, source_base_path):
    """Copies a single image, preserving its relative directory structure."""
    try:
        relative_path = os.path.relpath(source_path, source_base_path)
        destination_path = os.path.join(destination_base_path, relative_path)
        
        # Ensure destination directory exists
        destination_dir = os.path.dirname(destination_path)
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir, exist_ok=True)
        
        shutil.copy2(source_path, destination_path)
        return True, f"Copied {source_path} to {destination_path}"
    except Exception as e:
        return False, f"Failed to copy {source_path}: {e}"

def copy_selected_images(selected_image_paths, destination_base_path, source_base_path="/mnt/e/objects365/images/train"):
    """
    Copies a list of selected image paths to a new destination,
    preserving their relative directory structure, using multithreading.
    """
    print(f"Starting image copying to {destination_base_path} for {len(selected_image_paths)} images.")
    copied_count = 0
    failed_count = 0

    if not selected_image_paths:
        print("No images to copy.")
        return

    # Ensure the destination base path exists
    if not os.path.exists(destination_base_path):
        try:
            os.makedirs(destination_base_path, exist_ok=True)
            print(f"Created destination directory: {destination_base_path}")
        except Exception as e:
            print(f"Failed to create destination directory {destination_base_path}: {e}")
            return

    with ThreadPoolExecutor(max_workers=get_workers_count()) as executor:
        future_to_image = {
            executor.submit(copy_image, img_path, destination_base_path, source_base_path): img_path
            for img_path in selected_image_paths
        }
        for future in as_completed(future_to_image):
            img_path = future_to_image[future]
            success, message = future.result()
            if success:
                copied_count += 1
                if copied_count % 100 == 0:  # Progress feedback every 100 copies
                    print(f"Progress: {copied_count} images copied...")
            else:
                failed_count += 1
                print(message) # Log error messages

    print(f"Finished copying images. Successfully copied: {copied_count}, Failed: {failed_count}.")