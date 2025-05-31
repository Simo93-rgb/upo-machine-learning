import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_patch(patch_dir, base_path, f):
    """
    Processa una singola patch e scrive i percorsi delle immagini nel file di output.
    Args:
        patch_dir (str): Nome della directory della patch.
        base_path (Path): Percorso base delle immagini.
        f (file object): File di output aperto in modalità scrittura.
    """
    patch_path = base_path / patch_dir
    if patch_path.is_dir():
        for img in patch_path.glob("*.jpg"):
            f.write(f"{img.absolute()}\n")
        print(f"Completata patch: {patch_dir}")

def create_image_lists(base_dir, split, output_file, num_threads=4):
    """
    Crea un file di testo con l'elenco delle immagini per un determinato split (train, val, test).
    Utilizza il multithreading per processare le patch in parallelo.

    Args:
        base_dir (str): Percorso della directory principale del dataset.
        split (str): Split del dataset (train, val, test).
        output_file (str): Percorso del file di output.
        num_threads (int): Numero di thread da utilizzare per il multithreading.
    """
    base_path = Path(base_dir) / "images" / split
    patch_dirs = [d for d in os.listdir(base_path) if (base_path / d).is_dir()]

    print(f"Inizio creazione lista immagini per '{split}'...")
    print(f"Numero di patch da processare: {len(patch_dirs)}")

    with open(output_file, 'w') as f:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(process_patch, patch_dir, base_path, f): patch_dir
                for patch_dir in patch_dirs
            }

            for future in as_completed(futures):
                patch_name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Errore durante la processazione della patch {patch_name}: {e}")

    print(f"Creazione lista immagini per '{split}' completata. File salvato in: {output_file}")

def process_patch_to_txt(patch_dir, base_path, split, output_dir):
    """
    Crea un file TXT per una singola patch, contenente i percorsi delle immagini di quella patch.
    """
    patch_path = base_path / patch_dir
    out_txt = Path(output_dir) / f"{split}_{patch_dir}.txt"
    with open(out_txt, "w") as f:
        for img in patch_path.glob("*.jpg"):
            f.write(f"{img.absolute()}\n")
    print(f"Creato: {out_txt}")

def create_patch_image_lists(base_dir, split, output_dir, num_threads=8):
    """
    Crea un file TXT per ogni patch, contenente i percorsi delle immagini di quella patch.
    Args:
        base_dir (str): Percorso della directory principale del dataset.
        split (str): Split del dataset (train, val, test).
        output_dir (str): Directory dove salvare i file TXT per ogni patch.
        num_threads (int): Numero di thread da utilizzare per il multithreading.
    """
    base_path = Path(base_dir) / "images" / split
    patch_dirs = [d for d in os.listdir(base_path) if (base_path / d).is_dir()]
    os.makedirs(output_dir, exist_ok=True)

    print(f"Inizio creazione file patch TXT per '{split}' con {num_threads} thread...")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(process_patch_to_txt, patch_dir, base_path, split, output_dir)
            for patch_dir in patch_dirs
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Errore durante la processazione di una patch: {e}")
    print(f"Completata creazione file patch TXT per '{split}'.")

if __name__ == "__main__":
    # Crea gli elenchi
    base_dir = "/mnt/e/objects365"
    threads = 16
    create_image_lists(base_dir, "train", f"{base_dir}/train_images.txt", num_threads=threads)
    create_image_lists(base_dir, "val", f"{base_dir}/val_images.txt", num_threads=threads)
    create_image_lists(base_dir, "test", f"{base_dir}/test_images.txt", num_threads=threads)