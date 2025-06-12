import os
import shutil
from pathlib import Path
import random
from concurrent.futures import ThreadPoolExecutor

def move_patch_with_images(patch_name, val_images_dir, test_images_dir, val_labels_dir, test_labels_dir):
    """
    Sposta una singola patch di immagini e le relative etichette dalla directory di validazione a quella di test.
    Args:
        patch_name (str): Nome della patch da spostare.
        val_images_dir (Path): Directory delle immagini di validazione.
        test_images_dir (Path): Directory delle immagini di test.
        val_labels_dir (Path): Directory delle etichette di validazione.
        test_labels_dir (Path): Directory delle etichette di test.
    """
    # Sposta le immagini
    src_patch_images = val_images_dir / patch_name
    dst_patch_images = test_images_dir / patch_name
    shutil.move(src_patch_images, dst_patch_images)
    print(f"Spostata cartella immagini: {patch_name}")

    # Sposta le etichette (se esistono)
    src_patch_labels = val_labels_dir / patch_name
    dst_patch_labels = test_labels_dir / patch_name
    if os.path.exists(src_patch_labels):
        shutil.move(src_patch_labels, dst_patch_labels)
        print(f"Spostata cartella labels: {patch_name}")

def create_test_set_with_images(base_dir, test_size_ratio=0.1, num_threads=4):
    """
    Crea una cartella 'test' e sposta una frazione delle immagini dalla cartella 'val' in parallelo.
    Assume una struttura base_dir/images/{train, val} e base_dir/labels/{train, val}.

    Args:
        base_dir (str): Percorso della directory principale del dataset.
        test_size_ratio (float): Frazione delle immagini di validation da spostare nel test set.
        num_threads (int): Numero di thread da utilizzare per il multithreading.
    """
    val_images_dir = Path(base_dir) / "images" / "val"
    test_images_dir = Path(base_dir) / "images" / "test"
    val_labels_dir = Path(base_dir) / "labels" / "val"
    test_labels_dir = Path(base_dir) / "labels" / "test"

    # Crea le directory test/images e test/labels se non esistono
    test_images_dir.mkdir(parents=True, exist_ok=True)
    test_labels_dir.mkdir(parents=True, exist_ok=True)

    # Ottieni la lista di tutte le sottocartelle (patch) in val/images
    patch_dirs = [d for d in os.listdir(val_images_dir) if (val_images_dir / d).is_dir()]
    num_patches = len(patch_dirs)
    num_test_patches = int(num_patches * test_size_ratio)
    test_patches = random.sample(patch_dirs, num_test_patches)

    print(f"Spostando {num_test_patches} patch ({test_size_ratio*100}%) da validation a test...")

    # Usa ThreadPoolExecutor per spostare le patch in parallelo
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(
                move_patch_with_images,
                patch_name,
                val_images_dir,
                test_images_dir,
                val_labels_dir,
                test_labels_dir,
            )
            for patch_name in test_patches
        ]

        # Attendi il completamento di tutti i task
        for future in futures:
            future.result()

    print("Creazione test set completata.")


def move_tar_patch(patch_name, val_images_dir, test_images_dir, val_labels_dir, test_labels_dir):
    """
    Sposta un singolo file tar.gz e la relativa cartella di etichette dalla validation al test set.
    Args:
        patch_name (str): Nome della patch (senza estensione tar.gz).
        val_images_dir (Path): Directory delle immagini di validazione.
        test_images_dir (Path): Directory delle immagini di test.
        val_labels_dir (Path): Directory delle etichette di validazione.
        test_labels_dir (Path): Directory delle etichette di test.
    """
    # Per le immagini, sposta il file tar.gz
    src_image_file = val_images_dir / f"{patch_name}.tar.gz"
    dst_image_file = test_images_dir / f"{patch_name}.tar.gz"
    
    if os.path.exists(src_image_file):
        shutil.move(src_image_file, dst_image_file)
        print(f"Spostato file immagine: {patch_name}.tar.gz")
    else:
        print(f"File immagine non trovato: {patch_name}.tar.gz")
    
    # Per le etichette, sposta la cartella (se esiste)
    src_patch_labels = val_labels_dir / patch_name
    dst_patch_labels = test_labels_dir / patch_name
    
    if os.path.exists(src_patch_labels) and os.path.isdir(src_patch_labels):
        shutil.move(src_patch_labels, dst_patch_labels)
        print(f"Spostata cartella labels: {patch_name}")
    else:
        print(f"Cartella labels non trovata: {patch_name}")

def create_test_set(base_dir, test_size_ratio=0.1, num_threads=4, debug=False):
    """
    Crea una cartella 'test' e sposta una frazione dei file tar.gz e le relative cartelle labels.
    Assume una struttura con file .tar.gz in images/val e cartelle in labels/val.
    
    Args:
        base_dir (str): Percorso della directory principale del dataset.
        test_size_ratio (float): Frazione delle immagini di validation da spostare nel test set.
        num_threads (int): Numero di thread da utilizzare per il multithreading.
    """
    val_images_dir = Path(base_dir) / "images" / "val"
    test_images_dir = Path(base_dir) / "images" / "test"
    val_labels_dir = Path(base_dir) / "labels" / "val"
    test_labels_dir = Path(base_dir) / "labels" / "test"
    if debug:
        # DEBUG: Verifica che le directory esistano
        print(f"Verifico directory val_images_dir: {val_images_dir}")
        if not val_images_dir.exists():
            print(f"ERRORE: La directory {val_images_dir} non esiste!")
            return
        
        # DEBUG: Elenca il contenuto della directory
        print(f"Contenuto della directory {val_images_dir}:")
        for item in val_images_dir.iterdir():
            print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
        
        # DEBUG: Verifica specificamente i file .tar.gz
        tarfiles = list(val_images_dir.glob("*.tar.gz"))
        print(f"File .tar.gz trovati: {len(tarfiles)}")
        if len(tarfiles) > 0:
            print(f"Primi 5 file .tar.gz: {[f.name for f in tarfiles[:5]]}")

    # Crea le directory test/images e test/labels se non esistono
    test_images_dir.mkdir(parents=True, exist_ok=True)
    test_labels_dir.mkdir(parents=True, exist_ok=True)

    # Trova tutti i file tar.gz nella directory images/val
    tar_files = []
    for f in val_images_dir.glob("*.tar.gz"):
        # Estrai il nome della patch rimuovendo TUTTE le estensioni
        patch_name = f.name.split('.')[0]  # prende solo 'patch40' da 'patch40.tar.gz'
        tar_files.append(patch_name)   
         
    # Se non ci sono file tar.gz, mostra un avviso
    if not tar_files:
        print("Attenzione: Nessun file .tar.gz trovato nella directory images/val")
        return
    
    num_patches = len(tar_files)
    num_test_patches = int(num_patches * test_size_ratio)
    test_patches = random.sample(tar_files, num_test_patches)

    print(f"Spostando {num_test_patches} patch ({test_size_ratio*100}%) da validation a test...")

    # Usa ThreadPoolExecutor per spostare le patch in parallelo
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(
                move_tar_patch,
                patch_name,
                val_images_dir,
                test_images_dir,
                val_labels_dir,
                test_labels_dir,
            )
            for patch_name in test_patches
        ]

        # Attendi il completamento di tutti i task
        for future in futures:
            future.result()

    print("Creazione test set completata.")
if __name__ == "__main__":
    # Esegui la creazione del test set
    base_dir = "/mnt/d/objects365"  # Assicurati che questo sia il percorso corretto
    create_test_set(base_dir, test_size_ratio=0.1, num_threads=8)