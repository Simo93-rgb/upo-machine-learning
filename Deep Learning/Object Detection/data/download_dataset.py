import os
import requests
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tarfile import is_tarfile
from zipfile import ZipFile, is_zipfile
import tarfile  # Import tarfile for handling tar archives
from tqdm import tqdm  # Import tqdm for progress bars
from concurrent.futures import ThreadPoolExecutor

def unzip_file(file, path=None, exclude=(".DS_Store", "__MACOSX"), threads=8):
    """
    Unzip a *.zip file to path/, excluding files containing strings in the exclude list.
    Uses multithreading to speed up extraction.
    """
    if path is None:
        path = Path(file).parent  # default path

    with ZipFile(file) as zipObj:
        # Filtra i file da escludere
        file_list = [f for f in zipObj.namelist() if all(x not in f for x in exclude)]

        # Controlla se tutti i file sono già stati estratti
        all_extracted = all((Path(path) / f).exists() for f in file_list)
        if all_extracted:
            print(f"Skipping extraction: All files from {file} are already present in {path}")
            return

        def extract_member(member):
            zipObj.extract(member, path=path)

        # Usa ThreadPoolExecutor per estrarre i file in parallelo
        with ThreadPoolExecutor(max_workers=threads) as executor:
            list(executor.map(extract_member, file_list))

    print(f"Unzipped {len(file_list)} files from {file} to {path}")

def download_with_resume(url, dest, retry=3):
    """
    Download a file with resume capability and show progress.
    """
    headers = {}
    # Assicurati che il file esista prima di aprirlo in modalità append
    if not os.path.exists(dest):
        open(dest, 'wb').close()  # Crea un file vuoto

    if os.path.exists(dest):
        headers['Range'] = f"bytes={os.path.getsize(dest)}-"
    with requests.get(url, headers=headers, stream=True) as r:
        total_size = int(r.headers.get('content-length', 0)) + os.path.getsize(dest)
        if r.status_code == 416:  # File already fully downloaded
            print(f"{dest} is already fully downloaded.")
            return
        elif r.status_code not in (200, 206):
            raise Exception(f"Failed to download {url}: {r.status_code}")
        with open(dest, "ab", buffering=64 * 1024) as f, tqdm(
            desc=f"Downloading {Path(dest).name}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            initial=os.path.getsize(dest),
        ) as bar:
            for chunk in r.iter_content(chunk_size=65536):  # 64 KB chunks
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def download(url, dir=".", unzip=True, delete=False, threads=8, retry=3):
    """
    Multithreaded file download and unzip function with progress bars.
    """
    def download_one(url, dir):
        success = True
        f = Path(dir) / Path(url).name
        marker = f.with_suffix(f.suffix + ".downloaded")  # Marker file

        # Controlla se il file è già stato scaricato e processato
        if marker.exists():
            print(f"Skipping download and extraction: {f} is already processed.")
            return

        # Controlla se il file compresso esiste già
        if f.exists():
            print(f"File already exists: {f}. Skipping download.")
        else:
            # Scarica il file
            os.makedirs(f.parent, exist_ok=True)
            print(f"Starting download: {url} -> {f}")
            for i in range(retry + 1):
                try:
                    download_with_resume(url, f)
                    success = True
                    break
                except Exception as e:
                    print(f"⚠️ Download failure: {e}, retrying {i + 1}/{retry}...")
                    success = False
            if not success:
                print(f"❌ Failed to download {url} after {retry} retries.")
                return

        # Scompattamento del file, se necessario
        if unzip and (f.suffix == ".gz" or is_zipfile(f) or is_tarfile(f)):
            print(f"Unzipping {f}...")
            if is_zipfile(f):
                unzip_file(f, dir, threads)  # unzip
            elif is_tarfile(f):
                tar_dir = Path(f.parent)
                with tarfile.open(f, "r:*") as tar:
                    all_extracted = all((tar_dir / member.name).exists() for member in tar.getmembers())
                    if all_extracted:
                        print(f"Skipping extraction: All files from {f} are already present in {tar_dir}")
                    else:
                        tar.extractall(path=tar_dir)
                        print(f"Unzipped tar file {f} to {tar_dir}")
            elif f.suffix == ".gz":
                gz_file = Path(f.parent) / f.stem
                if gz_file.exists():
                    print(f"Skipping extraction: {gz_file} already exists")
                else:
                    os.system(f"gunzip -k {f}")  # unzip
                    print(f"Unzipped {f} to {gz_file}")

        # Elimina il file compresso, se richiesto
        if delete and f.exists():
            f.unlink()  # Rimuove il file compresso
            print(f"Deleted compressed file: {f}")

        # Crea il file marker
        marker.touch()
        print(f"Created marker file: {marker}")

    dir = Path(dir)
    os.makedirs(dir, exist_ok=True)  # make directory
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))  # multithreaded
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)


# if __name__ == "__main__":
#     threads = 16  # Number of threads for downloading
#     # Set the base directory to the local disk E (adjusted for WSL)
#     base_dir = os.path.join("/mnt/e", "object365")

#     # Ensure the base directory exists
#     os.makedirs(base_dir, exist_ok=True)

#     # Make Directories
#     for p in ["images", "labels"]:
#         for q in ["train", "val"]:
#             os.makedirs(os.path.join(base_dir, p, q), exist_ok=True)

#     # Train, Val Splits
#     for split, patches in [("train", 50 + 1), ("val", 43 + 1)]:
#         print(f"Processing {split} in {patches} patches ...")
#         images = os.path.join(base_dir, "images", split)
#         labels = os.path.join(base_dir, "labels", split)
#         # Download
#         url = f"https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/{split}/"
#         if split == "train":
#             download(
#                 [f"{url}zhiyuan_objv2_{split}.tar.gz"], dir=base_dir, delete=True
#             )  # annotations json
#             download(
#                 [f"{url}patch{i}.tar.gz" for i in range(patches)],
#                 dir=images,
#                 delete=True,
#                 threads=threads,
#             )
#         elif split == "val":
#             download(
#                 [f"{url}zhiyuan_objv2_{split}.json"], dir=base_dir, delete=True
#             )  # annotations json
#             download(
#                 [f"{url}images/v1/patch{i}.tar.gz" for i in range(15 + 1)],
#                 dir=images,
#                 delete=False,
#                 threads=threads,
#             )
#             download(
#                 [f"{url}images/v2/patch{i}.tar.gz" for i in range(16, patches)],
#                 dir=images,
#                 delete=True,
#                 threads=threads,
#             )
if __name__ == "__main__":
    threads = 16  # Numero di thread per il download
    # Imposta la directory base sul disco locale E (adattato per WSL)
    base_dir = os.path.join("/mnt/e", "object365")

    # Assicurati che la directory base esista
    os.makedirs(base_dir, exist_ok=True)

    # Crea le directory
    for p in ["images", "labels"]:
        for q in ["train", "val"]:
            os.makedirs(os.path.join(base_dir, p, q), exist_ok=True)

    # Train, Val Splits
    for split, patches in [("train", 50 + 1), ("val", 43 + 1)]:
        print(f"Processing {split} in {patches} patches ...")
        images = os.path.join(base_dir, "images", split)
        labels = os.path.join(base_dir, "labels", split)

        # URL base
        url = f"https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/{split}/"

        # Scarica le annotazioni (labels)
        if split == "train":
            download(
                [f"{url}zhiyuan_objv2_{split}.tar.gz"], dir=base_dir, delete=True
            )  # Annotazioni JSON per il training
        elif split == "val":
            download(
                [f"{url}zhiyuan_objv2_{split}.json"], dir=base_dir, delete=True
            )  # Annotazioni JSON per la validazione

        # Scarica le immagini
        if split == "train":
            download(
                [f"{url}patch{i}.tar.gz" for i in range(patches)],
                dir=images,
                delete=True,
                threads=threads,
            )
        elif split == "val":
            download(
                [f"{url}images/v1/patch{i}.tar.gz" for i in range(15 + 1)],
                dir=images,
                delete=False,
                threads=threads,
            )
            download(
                [f"{url}images/v2/patch{i}.tar.gz" for i in range(16, patches)],
                dir=images,
                delete=True,
                threads=threads,
            )

        # Estrai le annotazioni (se necessario)
        annotations_file = os.path.join(base_dir, f"zhiyuan_objv2_{split}.tar.gz" if split == "train" else f"zhiyuan_objv2_{split}.json")
        if os.path.exists(annotations_file):
            # Verifica se le annotazioni sono già state estratte
            if split == "train" and is_tarfile(annotations_file):
                extracted_marker = Path(annotations_file).with_suffix(".extracted")
                if extracted_marker.exists():
                    print(f"Annotations already extracted to {labels}")
                else:
                    print(f"Unzipping annotations: {annotations_file}")
                    with tarfile.open(annotations_file, "r:*") as tar:
                        tar.extractall(path=labels)
                    print(f"Annotations extracted to {labels}")
                    extracted_marker.touch()  # Crea un marker per future esecuzioni
            # Per il set di validazione, copia il JSON nella directory delle labels
            elif split == "val" and annotations_file.endswith(".json"):
                import shutil
                labels_json = os.path.join(labels, os.path.basename(annotations_file))
                if not os.path.exists(labels_json):
                    print(f"Copying annotations to {labels}")
                    shutil.copy(annotations_file, labels)
                else:
                    print(f"Annotations already in place: {labels_json}")