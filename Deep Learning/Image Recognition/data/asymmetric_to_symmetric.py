import os
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor

def move_label_files(src_dir, dest_dir):
    """
    Sposta tutti i file di etichetta da una directory sorgente a una directory di destinazione.
    Args:
        src_dir (Path): Directory sorgente contenente i file di etichetta.
        dest_dir (Path): Directory di destinazione.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    for label_file in src_dir.glob("*.txt"):
        dest_file = dest_dir / label_file.name
        if dest_file.exists():
            print(f"File già esistente, salto: {dest_file}")
            continue
        shutil.move(str(label_file), str(dest_file))
        print(f"Spostato: {label_file} -> {dest_file}")

def reorganize_labels(base_dir):
    """
    Riorganizza le etichette per corrispondere alla struttura delle immagini.
    Sposta i file da labels/{train,val}/{v1,v2}/patch* a labels/{train,val}/patch*.

    Args:
        base_dir (str): Percorso della directory principale del dataset.
    """
    tasks = []
    with ThreadPoolExecutor() as executor:
        for split in ["train", "val"]:
            labels_dir = Path(base_dir) / "labels" / split
            for version in ["v1", "v2"]:
                version_dir = labels_dir / version
                if version_dir.exists():
                    for patch_dir in version_dir.iterdir():
                        if patch_dir.is_dir():
                            # Destinazione: labels/{split}/patch*
                            dest_dir = labels_dir / patch_dir.name
                            # Controlla se la directory di destinazione è già completa
                            if dest_dir.exists() and any(dest_dir.glob("*.txt")):
                                print(f"Directory già processata, salto: {dest_dir}")
                                continue
                            # Aggiungi il task per spostare i file
                            tasks.append(executor.submit(move_label_files, patch_dir, dest_dir))

                    # Rimuovi la directory vuota v1 o v2 solo se tutte le patch sono state elaborate
                    if not any(version_dir.iterdir()):
                        shutil.rmtree(version_dir)
                        print(f"Rimossa directory: {version_dir}")
                else:
                    print(f"Directory non trovata, salto: {version_dir}")

        # Attendi il completamento di tutti i task
        for task in tasks:
            task.result()

    print("Riorganizzazione delle etichette completata.")

# Esegui la riorganizzazione
base_dir = "/mnt/d/objects365"  # Percorso principale del dataset
reorganize_labels(base_dir)