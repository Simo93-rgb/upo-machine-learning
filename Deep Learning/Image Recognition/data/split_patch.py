import os
import shutil
from math import ceil
from concurrent.futures import ThreadPoolExecutor, as_completed

def split_single_patch(patch_dir, patch):
    patch_path = os.path.join(patch_dir, patch)
    if not os.path.isdir(patch_path):
        return f"Patch non trovata: {patch}"
    files = sorted(os.listdir(patch_path))
    if not files:
        return f"Patch vuota: {patch}"
    mid = ceil(len(files) / 2)
    files_a = files[:mid]
    files_b = files[mid:]
    patch_a = f"{patch}_a"
    patch_b = f"{patch}_b"
    patch_a_path = os.path.join(patch_dir, patch_a)
    patch_b_path = os.path.join(patch_dir, patch_b)
    os.makedirs(patch_a_path, exist_ok=True)
    os.makedirs(patch_b_path, exist_ok=True)
    for f in files_a:
        shutil.move(os.path.join(patch_path, f), os.path.join(patch_a_path, f))
    for f in files_b:
        shutil.move(os.path.join(patch_path, f), os.path.join(patch_b_path, f))
    if not os.listdir(patch_path):
        os.rmdir(patch_path)
    return f"Divisa {patch} in {patch_a} e {patch_b}"

def dividi_patch_in_due_multithread(patch_dir, max_workers=4):
    patch_names = sorted([d for d in os.listdir(patch_dir) if d.startswith("patch") and os.path.isdir(os.path.join(patch_dir, d))])
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(split_single_patch, patch_dir, patch) for patch in patch_names]
        for future in as_completed(futures):
            print(future.result())

# Esempio d'uso:
# dividi_patch_in_due_multithread("/mnt/e/objects365/patch_txts/train", max_workers=8)