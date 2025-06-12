import tarfile
import io
import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
# Assicurati che il path di importazione sia corretto per la tua installazione/versione
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import IMG_FORMATS
from ultralytics.utils import LOGGER, TQDM

# --- Classe Dataset Personalizzata (Corretta) ---
class TarGzYOLODataset(YOLODataset):
    """
    Dataset YOLOv8 personalizzato per caricare immagini direttamente da archivi tar.gz.
    (Versione con fix per l'argomento 'split' in __init__)
    """
    def __init__(self, *args, **kwargs):
        # 1. Estrai gli argomenti personalizzati necessari PRIMA di chiamare super()
        #    Usiamo .pop() per estrarre E rimuovere da kwargs, così non vengono passati a super()
        self.split = kwargs.pop('split', 'train') # Estrai e rimuovi 'split'
        # L'argomento 'data' è invece richiesto da BaseDataset, quindi lo lasciamo in kwargs
        self.data = kwargs.get('data', None)
        if not self.data or 'path' not in self.data:
             raise ValueError("Il dizionario 'data' con la chiave 'path' deve essere fornito a TarGzYOLODataset.")

        # 2. Definisci i percorsi basati sugli argomenti estratti
        base_data_path = Path(self.data['path'])
        self.img_archive_dir = base_data_path / "images" / self.split
        self.label_base_dir = base_data_path / "labels" / self.split

        # Cache per tenere aperti i file tar
        self.open_tar_files = {}
        LOGGER.info(f"[{self.split.upper()}] TarGz Dataset Init: Img archives dir: {self.img_archive_dir}")
        LOGGER.info(f"[{self.split.upper()}] TarGz Dataset Init: Labels base dir: {self.label_base_dir}")

        # 3. Chiama l'inizializzatore della classe base
        #    Ora kwargs non contiene più 'split' (se esisteva)
        #    Passiamo *args e il dizionario kwargs modificato
        try:
            super().__init__(*args, **kwargs)
        except TypeError as e:
             LOGGER.error(f"Errore durante la chiamata a super().__init__ in TarGzYOLODataset: {e}")
             LOGGER.error(f"Argomenti passati a super().__init__: args={args}, kwargs={kwargs}")
             raise e
        # Nota: L'init della classe base popolerà self.im_files e cercherà le etichette

    # --- Metodi _get_open_tar, load_image, __del__ rimangono invariati ---
    # (Includili qui come nella versione precedente)

    def _get_open_tar(self, archive_path):
        """Helper per ottenere o aprire un oggetto tarfile, mettendolo in cache."""
        archive_path_str = str(archive_path)
        tar = self.open_tar_files.get(archive_path_str)
        if tar is None:
            if not archive_path.exists():
                LOGGER.error(f"Errore: Archivio non trovato {archive_path}")
                return None
            try:
                tar = tarfile.open(archive_path, 'r:gz')
                self.open_tar_files[archive_path_str] = tar
                #LOGGER.info(f"Aperto e messo in cache tarfile: {archive_path}")
            except tarfile.ReadError as e:
                LOGGER.error(f"Errore: Impossibile leggere l'archivio {archive_path}: {e}")
                return None
            except Exception as e:
                LOGGER.error(f"Errore: Impossibile aprire l'archivio {archive_path}: {e}")
                return None
        return tar

    def load_image(self, i):
        """
        Sovrascrive il metodo base per caricare l'i-esima immagine dall'archivio tar.gz corretto.
        Restituisce: img (ndarray BGR), (h0, w0), (h, w)
        """
        # Verifica se l'indice è valido e se il file è già stato marcato come None
        if i >= len(self.im_files) or self.im_files[i] is None:
             # LOGGER.warning(f"Tentativo di caricare immagine non valida all'indice {i}")
             return None, (0, 0), (0, 0)

        im_file_str = self.im_files[i] # Percorso 'teorico' dal file txt
        im_file_path = Path(im_file_str)

        # 1. Estrai le parti necessarie dal percorso teorico
        try:
            parts = im_file_path.parts
            split_index = -1
            for idx, part in enumerate(parts):
                 if part == "images":
                     split_index = idx + 1
                     break
            if split_index == -1 or split_index + 1 >= len(parts):
                 # Prova a gestire path relativi o diversi? Per ora assumiamo la struttura base
                 raise ValueError(f"Formato path immagine non riconosciuto o incompleto: {im_file_str}")

            patch_dir_name = parts[split_index + 1]
            image_filename = im_file_path.name # Usa .name per ottenere solo il nome file

            # 2. Costruisci il percorso dell'archivio e il nome del membro interno
            archive_filename = f"{patch_dir_name}.tar.gz"
            archive_path = self.img_archive_dir / archive_filename
            image_member_name = f"{patch_dir_name}/{image_filename}" # Path dentro il tar

        except Exception as e:
             LOGGER.error(f"Errore nel parsing del path immagine {im_file_str}: {e}")
             self.im_files[i] = None # Marca come non valido
             return None, (0, 0), (0, 0)

        # 3. Apri/Ottieni l'archivio tar dalla cache
        tar = self._get_open_tar(archive_path)
        if tar is None:
            self.im_files[i] = None
            return None, (0, 0), (0, 0)

        # 4. Leggi l'immagine dall'archivio
        img = None
        shape0 = (0, 0) # Shape originale (h, w)
        try:
            member = tar.getmember(image_member_name)
            img_file_bytes = tar.extractfile(member)
            if img_file_bytes:
                img_bytes = img_file_bytes.read()
                img_np = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR) # Legge come BGR
                if img is None:
                     # Logga un warning specifico per il fallimento di imdecode
                     LOGGER.warning(f"⚠️ cv2.imdecode ha restituito None per {image_member_name} da {archive_path}")
                     self.im_files[i] = None # Segna come non valido
                     return None, (0, 0), (0, 0)
                shape0 = img.shape[:2]
            else:
                 # extractfile ha restituito None? Improbabile ma gestiamolo
                 LOGGER.warning(f"⚠️ tar.extractfile ha restituito None per {image_member_name} in {archive_path}")
                 self.im_files[i] = None # Segna come non valido
                 return None, (0, 0), (0, 0)

        except KeyError:
            # LOGGER.warning(f"⚠️ Immagine '{image_member_name}' non trovata nell'archivio {archive_path}")
            self.im_files[i] = None
            return None, (0, 0), (0, 0)
        except tarfile.ReadError as e:
             LOGGER.error(f"❌ Errore di lettura tar durante l'estrazione di {image_member_name} da {archive_path}: {e}")
             self.im_files[i] = None
             return None, (0, 0), (0, 0)
        except Exception as e:
            LOGGER.error(f"❌ Errore caricando/decodificando {image_member_name} da {archive_path}: {e}")
            self.im_files[i] = None
            return None, (0, 0), (0, 0)

        return img, shape0, img.shape[:2]

    def __del__(self):
        # Usa hasattr per sicurezza, __del__ può essere chiamato in stati strani
        split_name = self.split.upper() if hasattr(self, 'split') and self.split else 'UNKNOWN SPLIT'
        num_files = len(self.open_tar_files)
        # Non loggare se non ci sono file aperti, riduce l'output
        if num_files > 0:
             LOGGER.info(f"[{split_name}] Chiusura di {num_files} file tar in cache...")
             count = 0
             for path, tar in self.open_tar_files.items():
                 try:
                     if hasattr(tar, 'close') and callable(tar.close):
                          tar.close()
                          count += 1
                 except Exception as e:
                     #LOGGER.warning(f"  - Errore chiudendo {path}: {e}")
                     pass # Ignora l'errore durante la chiusura
             #LOGGER.info(f"  Chiusi {count} file tar.")
        self.open_tar_files = {}