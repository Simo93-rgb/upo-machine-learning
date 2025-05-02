import os
import cv2
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Dimensione coerente con YOLOv8
img_size = (640, 640)

def process_file(root, fname, orig_images, orig_labels, cached_images, cached_labels):
    # Filtra solo le immagini
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        return

    stem = os.path.splitext(fname)[0]

    # rel_path = percorso relativo rispetto alla root di orig_images (include la patch)
    rel_path = os.path.relpath(root, start=orig_images)

    # Percorsi di origine
    img_path   = os.path.join(root, fname)
    label_path = os.path.join(orig_labels, rel_path, stem + ".txt")

    # Percorsi di destinazione (.pt per l'immagine, .txt per la label)
    cached_img_path   = os.path.join(cached_images, rel_path, stem + ".pt")
    cached_label_path = os.path.join(cached_labels, rel_path, stem + ".txt")

    # Crea le directory di destinazione (patch comprese)
    os.makedirs(os.path.dirname(cached_img_path), exist_ok=True)
    os.makedirs(os.path.dirname(cached_label_path), exist_ok=True)

    # Carica e preprocessa l'immagine
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Impossibile leggere: {img_path}")
        return

    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Converti in tensore PyTorch e normalizza su [0,1]
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)

    # Salva il tensore con torch.save()
    torch.save(img_tensor, cached_img_path)

    # Copia le label (pass-through .txt YOLO)
    if os.path.exists(label_path):
        with open(label_path, "r") as f_in, open(cached_label_path, "w") as f_out:
            f_out.write(f_in.read())
    else:
        print(f"⚠️ Label non trovata per: {img_path}")

if __name__ == "__main__":
    # Cicla su train/val/test, mantenendo le patch
    for fase in ["train", "val", "test"]:
        orig_images   = f"/mnt/d/objects365/images/{fase}"
        orig_labels   = f"/mnt/d/objects365/labels/{fase}"
        cached_images = f"/mnt/e/yolo-fast/images/{fase}"
        cached_labels = f"/mnt/e/yolo-fast/labels/{fase}"

        # Raccogli tutte le immagini, patch incluse
        file_list = []
        for root, _, files in os.walk(orig_images):
            for file in files:
                file_list.append((root, file))

        # Preprocessing parallelo
        with ThreadPoolExecutor() as executor:
            list(tqdm(
                executor.map(
                    lambda args: process_file(
                        args[0], args[1],
                        orig_images, orig_labels,
                        cached_images, cached_labels
                    ),
                    file_list
                ),
                total=len(file_list),
                desc=f"Preprocessing {fase}"
            ))
