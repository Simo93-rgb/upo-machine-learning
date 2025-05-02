import os
import torch
import webdataset as wds
from PIL import Image
from pathlib import Path
import numpy as np
from ultralytics.data.augment import Compose
from ultralytics.data.utils import check_det_dataset

class OptimizedDetectionLoader:
    def __init__(self, base_path, yaml_path, shard_counts, num_orig_classes, img_size=640):
        """
        Args:
            base_path (str): Percorso base del dataset (es. "/mnt/d/objects365_subsample_2")
            yaml_path (str): Percorso completo del file YAML di configurazione
            shard_counts (dict): Numero di shard per split
            num_orig_classes (int): Numero classi originali
        """
        self.base_path = Path(base_path)
        self.yaml_path = Path(yaml_path)
        self._validate_paths()
        
        self.shard_counts = shard_counts
        self.unknown_idx = num_orig_classes
        self.img_size = img_size

    def _validate_paths(self):
        """Verifica l'esistenza di tutti i percorsi critici"""
        required_paths = [
            self.base_path / "images",
            self.base_path / "labels",
            self.yaml_path
        ]
        
        missing = [str(p) for p in required_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Paths missing:\n" + "\n".join(missing) + 
                "\nVerify dataset structure and YAML file paths."
            )
        
    def get_shard_patterns(self, split):
        """Genera pattern degli shard corretti"""
        num_shards = self.shard_counts[split]
        return [
            f"{self.base_path}/images/{split}/patch_{{000..{num_shards-1:03d}}}.tar.gz"
        ]

    def pipeline(self, split, batch_size, augment=True):
        """Pipeline corretta"""
        patterns = self.get_shard_patterns(split)
        
        return wds.DataPipeline(
            wds.SimpleShardList(patterns),
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.decode("pilrgb"),
            wds.to_tuple("jpg", "txt"),
            wds.map_tuple(
                self._apply_augmentations(augment),
                self._process_labels
            ),
            wds.batched(batch_size, collation_fn=self.collate_fn)
        )

    def _apply_augmentations(self, augment):
        """Applica le augmentations di YOLO mantenendo la compatibilità"""
        def _augment(image):
            if augment:
                # Converti in formato compatibile con YOLO Augment
                img = np.array(image)
                transformed = self.augmentations({
                    "img": img,
                    "ori_shape": img.shape[:2],
                    "resized_shape": (self.img_size, self.img_size)
                })
                return transformed["img"]
            return image.resize((self.img_size, self.img_size))
        return _augment

    def _process_labels(self, label):
        """Processa le label nel formato YOLO ottimizzato"""
        labels = []
        for line in label.decode().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
                
            # Rimappatura classe
            cls = int(parts[0])
            if cls >= self.num_orig_classes:
                cls = self.unknown_idx
                
            # Conversione a formato tensoriale
            labels.append(torch.tensor([cls] + list(map(float, parts[1:]))))
        
        return torch.stack(labels) if labels else torch.empty(0, 5)

    def collate_fn(self, batch):
        """Funzione di collation ottimizzata per GPU"""
        images, labels = zip(*batch)
        return (
            torch.stack(images).to(torch.float32).permute(0, 3, 1, 2) / 255.0,
            [label.to(torch.float32) for label in labels]
        )

    def verify_dataset(self):
        """Versione corretta della validazione del dataset"""
        try:
            # Usa il percorso YAML invece del dizionario
            dataset_info = check_det_dataset(self.yaml_path)
            print("✅ Dataset verification passed!")
            print(f"Dataset info: {dataset_info}")
            return True
        except Exception as e:
            print(f"❌ Dataset verification failed: {str(e)}")
            return False

if __name__ == "__main__":
# Configurazione corretta
    loader = OptimizedDetectionLoader(
        base_path="/mnt/d/objects365_subsample_2",
        yaml_path="/mnt/d/objects365_subsample_2/dataset.yaml",  # Deve esistere
        shard_counts={"train": 50, "val": 10},
        num_orig_classes=364
    )
    if loader.verify_dataset():
        print("Proceeding with training...")
    else:
        print("Fix dataset issues before proceeding.")

    train_loader = loader.pipeline("train", batch_size=64, augment=True)
    val_loader = loader.pipeline("val", batch_size=32, augment=False)