from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER
from .tar_gz_yolo_dataset import TarGzYOLODataset
# --- Classe Trainer Personalizzata ---
class TarGzTrainer(DetectionTrainer):
    """
    Trainer YOLOv8 personalizzato che utilizza TarGzYOLODataset.
    """
    def build_dataset(self, img_path, mode='train', batch=None):
        """
        Costruisce e restituisce un oggetto TarGzYOLODataset.
        """
        gs = int(max(self.model.stride))
        LOGGER.info(f"Costruzione TarGzYOLODataset per la modalità '{mode}'...")
        try:
             # Passa gli argomenti necessari a TarGzYOLODataset
             dataset = TarGzYOLODataset(
                 img_path=img_path, # Percorso letto dal file txt (es train_images.txt)
                 imgsz=self.args.imgsz,
                 batch_size=batch,
                 augment=mode == 'train',  # Applica augment solo per train
                 hyp=self.args,
                 rect=mode != 'train',  # Usa rect solo per val/test per efficienza
                 cache=self.args.cache,
                 stride=gs,
                 pad=0.0 if mode == 'train' else 0.5, # Padding diverso
                 prefix=f"{mode}: ",
                 data=self.data,  # Passa il dizionario dati caricato dal YAML!
                 split=mode       # Specifica lo split corrente!
             )
             return dataset
        except Exception as e:
             LOGGER.error(f"Errore durante la costruzione di TarGzYOLODataset: {e}", exc_info=True)
             raise e # Rilancia l'eccezione per fermare il training