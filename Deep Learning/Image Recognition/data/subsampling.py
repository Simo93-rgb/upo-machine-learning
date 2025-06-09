import os
from pathlib import Path
import random

def create_subsample_image_list(base_dir, output_dir, train_patches=2, val_patches=1):
    """
    Crea un subsample dei file di coordinate (file .txt) per train e val selezionando un numero limitato di patch.
    
    Args:
        base_dir (str): Percorso della directory principale del dataset originale.
        output_dir (str): Percorso della directory dove salvare i file di output.
        train_patches (int): Numero di patch da includere nel train set.
        val_patches (int): Numero di patch da includere nel val set.
    """
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Directory delle immagini
    train_images_dir = base_dir / "images" / "train"
    val_images_dir = base_dir / "images" / "val"

    # File di output
    train_output_file = output_dir / "train_images.txt"
    val_output_file = output_dir / "val_images.txt"

    # Seleziona patch casuali per train
    train_patches_list = random.sample(
        [d for d in os.listdir(train_images_dir) if (train_images_dir / d).is_dir()],
        train_patches
    )
    with open(train_output_file, "w") as f:
        for patch in train_patches_list:
            patch_path = train_images_dir / patch
            for img in patch_path.glob("*.jpg"):
                f.write(f"{img.absolute()}\n")
            print(f"Patch train aggiunta: {patch}")

    # Seleziona patch casuali per val
    val_patches_list = random.sample(
        [d for d in os.listdir(val_images_dir) if (val_images_dir / d).is_dir()],
        val_patches
    )
    with open(val_output_file, "w") as f:
        for patch in val_patches_list:
            patch_path = val_images_dir / patch
            for img in patch_path.glob("*.jpg"):
                f.write(f"{img.absolute()}\n")
            print(f"Patch val aggiunta: {patch}")

    print("Subsample dei file di coordinate creato con successo!")

if __name__ == "__main__":
    # Esegui la creazione del subsample
    base_dir = "/mnt/e/objects365"  # Percorso del dataset originale
    output_dir = "/mnt/e/objects365_subsample_4"  # Percorso per salvare i file di coordinate
    create_subsample_image_list(base_dir, output_dir, train_patches=4, val_patches=1)