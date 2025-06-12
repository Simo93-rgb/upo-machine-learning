import os
from torch.utils.data import DataLoader
import webdataset as wds
from PIL import Image
import io


class DataLoaderManager:
    def __init__(self, base_images, base_labels, shard_counts, num_orig_classes):
        """
        Inizializza il gestore del DataLoader.

        Args:
            base_images (str): Percorso base delle immagini.
            base_labels (str): Percorso base delle etichette.
            shard_counts (dict): Numero di shard per ogni split (train, val, test).
            num_orig_classes (int): Numero di classi originali.
        """
        self.base_images = base_images
        self.base_labels = base_labels
        self.shard_counts = shard_counts
        self.num_orig_classes = num_orig_classes
        self.unknown_idx = num_orig_classes  # Classe "Unknown"

    def get_shards(self, split: str):
        """
        Ottiene la lista di shard per uno specifico split.

        Args:
            split (str): Split del dataset (train, val, test).

        Returns:
            list: Lista di percorsi agli shard.
        """
        return [
            os.path.join(self.base_images, split, f"patch_{i}.tar.gz")
            for i in range(1, self.shard_counts[split] + 1)
        ]

    def decode_and_remap(self, jpg_bytes, txt_bytes):
        """
        Decodifica un'immagine e rimappa le etichette.

        Args:
            jpg_bytes (bytes): Bytes dell'immagine JPEG.
            txt_bytes (bytes): Bytes del file di etichette.

        Returns:
            tuple: Immagine decodificata e stringa delle etichette rimappate.
        """
        # Decodifica immagine
        img = Image.open(io.BytesIO(jpg_bytes)).convert("RGB").resize((640, 640))
        # Rimappa le etichette
        new_txt = []
        for line in txt_bytes.decode().strip().splitlines():
            parts = line.split()
            cls = int(parts[0])
            # Rimappa la classe 365 → UNKNOWN_IDX
            if cls == self.num_orig_classes:
                cls = self.unknown_idx
            new_txt.append(" ".join([str(cls)] + parts[1:]))
        return img, "\n".join(new_txt)

    def make_loader(self, split: str, batch_size: int, workers: int, shuffle: bool):
        """
        Crea un DataLoader per uno specifico split.

        Args:
            split (str): Split del dataset (train, val, test).
            batch_size (int): Dimensione del batch.
            workers (int): Numero di worker per il DataLoader.
            shuffle (bool): Se mescolare i dati.

        Returns:
            DataLoader: DataLoader configurato.
        """
        shards = self.get_shards(split)
        ds = wds.WebDataset(shards, handler=wds.warn_and_continue, shardshuffle=False)
        if shuffle:
            ds = ds.shuffle(1000)
        ds = (
            ds.decode("rgb")                   # Legge il JPEG→bytes
              .to_tuple("jpg", "txt")          # Estrae jpg e txt
              .map_tuple(self.decode_and_remap)  # Applica il decode + remap
        )
        return DataLoader(ds, batch_size=batch_size, num_workers=workers, pin_memory=True)


# Esempio di utilizzo
if __name__ == "__main__":
    BASE_IMAGES = "/mnt/d/objects365/images"
    BASE_LABELS = "/mnt/d/objects365/labels"
    SHARD_COUNTS = {
        "train": 50,
        "val": 43,
        "test": len(os.listdir("/mnt/d/objects365/images/test"))
    }
    NUM_ORIG_CLASSES = 365

    # Inizializza il gestore
    manager = DataLoaderManager(BASE_IMAGES, BASE_LABELS, SHARD_COUNTS, NUM_ORIG_CLASSES)

    # Crea un DataLoader per il training
    train_loader = manager.make_loader("train", batch_size=16, workers=4, shuffle=True)