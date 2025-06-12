import os
import json
import random
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

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
        print(f"❌ Error reading {file_path}: {e}")
    return annotations

def collect_labels_from_patch(patch_path, base_images_path):
    """
    Collects label information for all .txt files in a given patch directory
    and maps them to their corresponding image paths.
    Returns a list of (image_path, annotations) tuples.
    """
    patch_annotations = []
    txt_files_count = 0
    found_images_count = 0
    
    for root, _, files in os.walk(patch_path):
        for file_name in files:
            if file_name.endswith('.txt'):
                txt_files_count += 1
                label_file_path = os.path.join(root, file_name)
                annotations = parse_label_file(label_file_path)
                if annotations:
                    # Construct the corresponding image path
                    relative_path = os.path.relpath(label_file_path, patch_path)
                    image_name = file_name.replace('.txt', '.jpg')
                    
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
                        found_images_count += 1
                    else:
                        print(f"⚠️  Warning: Image not found for label file {label_file_path} at {full_image_path}")
    
    return patch_annotations, txt_files_count, found_images_count

def discover_and_read_labels(labels_base_path, images_base_path):
    """
    Discovers all patchX_a/b directories and reads their label files in parallel.
    Returns a dictionary where keys are class IDs and values are lists of
    (image_path, bounding_box_coords) for that class.
    """
    print(f"🔍 Scanning labels in: {labels_base_path}")
    all_class_data = defaultdict(list)
    patch_dirs = []

    # Find all patchX_a and patchX_b directories
    for root, dirs, _ in os.walk(labels_base_path):
        for d in dirs:
            if d.startswith('patch') and ('_a' in d or '_b' in d):
                patch_dirs.append(os.path.join(root, d))
    
    if not patch_dirs:
        print(f"❌ No patch directories found in {labels_base_path}. Please check the path and structure.")
        return all_class_data

    print(f"📂 Found {len(patch_dirs)} patch directories")
    print(f"⚙️  Processing with {get_workers_count()} workers...")

    # Thread-safe counters
    progress_lock = threading.Lock()
    total_txt_files = 0
    total_found_images = 0
    processed_patches = 0

    with ThreadPoolExecutor(max_workers=get_workers_count()) as executor:
        # Submit all tasks
        future_to_patch = {
            executor.submit(collect_labels_from_patch, patch_dir, images_base_path): patch_dir
            for patch_dir in patch_dirs
        }
        
        # Process results with progress bar
        with tqdm(total=len(patch_dirs), desc="🔄 Processing patches", unit="patch") as pbar:
            for future in as_completed(future_to_patch):
                patch_dir = future_to_patch[future]
                try:
                    patch_annotations, txt_count, img_count = future.result()
                    
                    with progress_lock:
                        total_txt_files += txt_count
                        total_found_images += img_count
                        processed_patches += 1
                    
                    for image_path, annotations_for_image in patch_annotations:
                        for class_id, bbox_coords in annotations_for_image:
                            all_class_data[class_id].append((image_path, bbox_coords))
                    
                    pbar.set_postfix({
                        'TXT files': total_txt_files,
                        'Images found': total_found_images,
                        'Classes': len(all_class_data)
                    })
                    pbar.update(1)
                    
                except Exception as exc:
                    print(f'❌ {patch_dir} generated an exception: {exc}')
                    pbar.update(1)
    
    print(f"\n✅ Finished collecting all labels!")
    print(f"📊 Summary:")
    print(f"   • Processed patches: {processed_patches}/{len(patch_dirs)}")
    print(f"   • Total label files: {total_txt_files}")
    print(f"   • Total images found: {total_found_images}")
    print(f"   • Total classes discovered: {len(all_class_data)}")
    
    return all_class_data

def perform_subsampling(all_class_data, target_samples_per_class=None, total_target_images=None):
    """
    Performs subsampling on the collected class data.
    """
    print("\n🎯 Starting subsampling...")
    sampled_images = set()
    class_counts_before = {cid: len(data) for cid, data in all_class_data.items()}
    
    # Show class distribution summary
    total_classes = len(class_counts_before)
    total_images_before = sum(class_counts_before.values())
    print(f"📈 Dataset overview:")
    print(f"   • Total classes: {total_classes}")
    print(f"   • Total images: {total_images_before}")
    
    # Show top 10 most represented classes
    top_classes = sorted(class_counts_before.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"   • Top 10 classes by count: {dict(top_classes)}")

    if target_samples_per_class is not None:
        print(f"🎯 Targeting {target_samples_per_class} samples per class...")
        
        with tqdm(total=len(all_class_data), desc="🔄 Subsampling classes", unit="class") as pbar:
            for class_id, data in all_class_data.items():
                if len(data) > target_samples_per_class:
                    # Randomly sample if more than target
                    sampled_class_data = random.sample(data, target_samples_per_class)
                else:
                    # Use all available if less than or equal to target
                    sampled_class_data = data
                
                for image_path, _ in sampled_class_data:
                    sampled_images.add(image_path)
                
                pbar.set_postfix({'Selected images': len(sampled_images)})
                pbar.update(1)
                
    elif total_target_images is not None:
        print(f"🎯 Targeting approximately {total_target_images} total images...")
        
        if not all_class_data:
            print("❌ No class data to subsample.")
            return sampled_images

        # Calculate total annotations to determine proportions
        total_annotations = sum(len(data) for data in all_class_data.values())
        if total_annotations == 0:
            print("❌ No annotations found for subsampling.")
            return sampled_images

        print("📊 Calculating proportional distribution...")
        
        # Determine target images per class based on proportion
        target_images_per_class_proportional = {}
        for class_id, data in all_class_data.items():
            proportion = len(data) / total_annotations
            target_images_per_class_proportional[class_id] = int(proportion * total_target_images)
        
        # Ensure a minimum number of images per class
        min_images_per_class = 5
        print(f"🔧 Applying minimum {min_images_per_class} images per class...")
        
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

        # Show sampling plan for top classes
        top_targets = sorted(target_images_per_class_proportional.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"📋 Sampling plan (top 10 classes): {dict(top_targets)}")

        # Perform sampling
        with tqdm(total=len(all_class_data), desc="🔄 Sampling images", unit="class") as pbar:
            for class_id, data in all_class_data.items():
                num_to_sample = target_images_per_class_proportional.get(class_id, 0)
                if num_to_sample > 0:
                    sampled_class_data = random.sample(data, num_to_sample)
                    for image_path, _ in sampled_class_data:
                        sampled_images.add(image_path)
                
                pbar.set_postfix({'Selected images': len(sampled_images)})
                pbar.update(1)
    else:
        print("⚠️  No subsampling parameters provided. Returning all images...")
        for class_id, data in all_class_data.items():
            for image_path, _ in data:
                sampled_images.add(image_path)

    print(f"\n✅ Subsampling complete!")
    print(f"📊 Results:")
    print(f"   • Selected {len(sampled_images)} unique images")
    print(f"   • Reduction: {total_images_before} → {len(sampled_images)} ({len(sampled_images)/total_images_before*100:.1f}%)")
    
    return sampled_images

def calculate_class_statistics_optimized(selected_image_paths, all_class_data):
    """
    Calcola le statistiche delle classi in modo ottimizzato usando multithreading.
    """
    print("📊 Calculating final class distribution (optimized)...")
    
    # Crea un mapping inverso: image_path -> set di class_ids
    image_to_classes = defaultdict(set)
    
    # Prima fase: costruisci il mapping inverso con multithreading
    def process_class_chunk(class_items):
        local_mapping = defaultdict(set)
        for class_id, data_list in class_items:
            for img_path, _ in data_list:
                local_mapping[img_path].add(class_id)
        return local_mapping
    
    # Dividi le classi in chunks per il multithreading
    class_items = list(all_class_data.items())
    chunk_size = max(1, len(class_items) // get_workers_count())
    chunks = [class_items[i:i + chunk_size] for i in range(0, len(class_items), chunk_size)]
    
    print(f"🔄 Building image-to-class mapping with {len(chunks)} chunks...")
    
    with ThreadPoolExecutor(max_workers=get_workers_count()) as executor:
        future_to_chunk = {executor.submit(process_class_chunk, chunk): chunk for chunk in chunks}
        
        with tqdm(total=len(chunks), desc="🗂️  Building mapping", unit="chunk") as pbar:
            for future in as_completed(future_to_chunk):
                local_mapping = future.result()
                # Unisci i risultati
                for img_path, class_set in local_mapping.items():
                    image_to_classes[img_path].update(class_set)
                pbar.update(1)
    
    # Seconda fase: conta le classi per le immagini selezionate
    class_counts = defaultdict(int)
    
    print(f"🔢 Counting classes for {len(selected_image_paths)} selected images...")
    with tqdm(selected_image_paths, desc="📊 Counting classes", unit="image") as pbar:
        for img_path in pbar:
            if img_path in image_to_classes:
                for class_id in image_to_classes[img_path]:
                    class_counts[class_id] += 1
    
    return dict(class_counts)

def create_image_lists_with_subsampling(base_dir, split, output_file, 
                                      target_samples_per_class=None, 
                                      total_target_images=None,
                                      save_class_stats=True):
    """
    Crea un file di testo con l'elenco delle immagini per un determinato split,
    applicando subsampling basato sulle classi delle etichette.
    """
    print(f"\n🚀 Starting image list creation for '{split}' split with subsampling...")
    print(f"📁 Base directory: {base_dir}")
    print(f"💾 Output file: {output_file}")
    
    labels_base_path = os.path.join(base_dir, "labels", split)
    images_base_path = os.path.join(base_dir, "images", split)
    
    # Step 1: Discover and read all labels
    print(f"\n📋 Step 1/4: Discovering and reading labels...")
    all_class_data = discover_and_read_labels(labels_base_path, images_base_path)
    
    if not all_class_data:
        print(f"❌ No class data found for split '{split}'. Check the dataset structure.")
        return
    
    # Step 2: Perform subsampling
    print(f"\n📋 Step 2/4: Performing subsampling...")
    selected_image_paths = perform_subsampling(
        all_class_data, 
        target_samples_per_class=target_samples_per_class,
        total_target_images=total_target_images
    )
    
    # Step 3: Save image paths to output file
    print(f"\n📋 Step 3/4: Saving image paths...")
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Created output directory: {output_dir}")
    
    print(f"💾 Writing {len(selected_image_paths)} paths to file...")
    with open(output_file, 'w') as f:
        for img_path in tqdm(sorted(selected_image_paths), desc="💾 Writing paths", unit="path"):
            f.write(f"{os.path.abspath(img_path)}\n")
    
    print(f"✅ Saved {len(selected_image_paths)} image paths to: {output_file}")
    
    # Step 4: Save class statistics if requested (OTTIMIZZATO)
    if save_class_stats:
        print(f"\n📋 Step 4/4: Generating class statistics (optimized)...")
        stats_file = output_file.replace('.txt', '_class_stats.json')
        
        # Usa la versione ottimizzata per il calcolo delle statistiche
        class_counts = calculate_class_statistics_optimized(selected_image_paths, all_class_data)
        
        stats = {
            "total_images": len(selected_image_paths),
            "total_classes": len(class_counts),
            "class_distribution": class_counts,
            "subsampling_params": {
                "target_samples_per_class": target_samples_per_class,
                "total_target_images": total_target_images
            }
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✅ Saved class statistics to: {stats_file}")
    
    print(f"\n🎉 Image list creation completed successfully!")
    print(f"📊 Final summary:")
    print(f"   • Split: {split}")
    print(f"   • Total images selected: {len(selected_image_paths)}")
    print(f"   • Output file: {output_file}")

def create_image_lists(base_dir, split, output_file, num_threads=None):
    """
    Versione originale: crea un file di testo con l'elenco di TUTTE le immagini 
    per un determinato split senza subsampling.
    """
    if num_threads is None:
        num_threads = get_workers_count()
    
    print(f"\n🚀 Starting image list creation for '{split}' split (NO subsampling)...")
    
    base_path = Path(base_dir) / "images" / split
    patch_dirs = [d for d in os.listdir(base_path) if (base_path / d).is_dir()]

    print(f"📂 Found {len(patch_dirs)} patch directories")
    print(f"⚙️  Using {num_threads} threads...")

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Created output directory: {output_dir}")

    # Thread-safe writing using a lock
    write_lock = threading.Lock()
    total_images = 0

    with open(output_file, 'w') as f:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_patch = {
                executor.submit(process_patch, patch_dir, base_path): patch_dir
                for patch_dir in patch_dirs
            }

            with tqdm(total=len(patch_dirs), desc="🔄 Processing patches", unit="patch") as pbar:
                for future in as_completed(future_to_patch):
                    patch_name = future_to_patch[future]
                    try:
                        images_in_patch = future.result()
                        with write_lock:
                            for img_path in images_in_patch:
                                f.write(f"{img_path}\n")
                                total_images += 1
                        
                        pbar.set_postfix({'Total images': total_images})
                        pbar.update(1)
                    except Exception as e:
                        print(f"❌ Error processing patch {patch_name}: {e}")
                        pbar.update(1)

    print(f"\n✅ Image list creation completed!")
    print(f"📊 Summary:")
    print(f"   • Split: {split}")
    print(f"   • Total images: {total_images}")
    print(f"   • Output file: {output_file}")

def process_patch(patch_dir, base_path):
    """
    Processa una singola patch e restituisce i percorsi delle immagini.
    """
    image_paths = []
    patch_path = base_path / patch_dir
    if patch_path.is_dir():
        for img in patch_path.glob("*.jpg"):
            image_paths.append(str(img.absolute()))
    return image_paths

if __name__ == "__main__":
    base_dir = "/mnt/e/objects365"
    
    print("🎯 Objects365 Dataset Processing")
    print("=" * 50)
    
    # Esempio di utilizzo con subsampling
    print("\n📋 Example 1: Target samples per class")
    create_image_lists_with_subsampling(
        base_dir, 
        "train", 
        f"{base_dir}/train_images_subsampled_100_per_class.txt",
        target_samples_per_class=100
    )
    
    print("\n📋 Example 2: Target total images")
    create_image_lists_with_subsampling(
        base_dir, 
        "train", 
        f"{base_dir}/train_images_subsampled_10k_total.txt",
        total_target_images=10000
    )
    
    print("\n📋 Example 3: All images (no subsampling)")
    create_image_lists(base_dir, "train", f"{base_dir}/train_images_all.txt")