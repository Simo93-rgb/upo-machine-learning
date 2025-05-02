import os
import sys
import tarfile
import numpy as np
from pathlib import Path
from tqdm import tqdm
import requests
from typing import List, Union, Tuple, Optional, Any
import concurrent.futures
from zipfile import ZipFile, is_zipfile
from itertools import repeat
from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor
import argparse

# Verify that pycocotools is installed
try:
    from pycocotools.coco import COCO
except ImportError:
    print("Installation of pycocotools required. Run: pip install pycocotools>=2.0")
    sys.exit(1)

def is_tarfile(filename: Union[str, Path]) -> bool:
    """
    Checks if the given file is a valid tar file.
    Args:
        filename (str or Path): The path to the file to check.
    Returns:
        bool: True if the file exists and is a tar file, False otherwise.
    """
    
    return tarfile.is_tarfile(filename) if os.path.exists(filename) else False

def unzip_file(file: Union[str, Path], 
               path: Optional[Union[str, Path]] = None, 
               exclude: Tuple[str, ...] = (".DS_Store", "__MACOSX"), 
               threads: int = 1) -> None:
    """
    Extracts the contents of a ZIP file to a specified directory, with support for multithreaded extraction
    and optional exclusion of specific files or directories.

    Args:
        file (str or Path): The path to the ZIP file to be extracted.
        path (str or Path, optional): The directory where the files will be extracted. 
            Defaults to the parent directory of the ZIP file.
        exclude (tuple, optional): A tuple of substrings. Files or directories containing any of these 
            substrings will be excluded from extraction. Defaults to (".DS_Store", "__MACOSX").
        threads (int, optional): The number of threads to use for parallel extraction. Defaults to 8.

    Returns:
        None

    Prints:
        A message indicating whether the extraction was skipped (if all files are already present) 
        or the number of files successfully extracted.
    """

    if path is None:
        path = Path(file).parent  # default path

    with ZipFile(file) as zipObj:
        # Filter files to exclude
        file_list = [f for f in zipObj.namelist() if all(x not in f for x in exclude)]

        # Check if all files have already been extracted
        all_extracted = all((Path(path) / f).exists() for f in file_list)
        if all_extracted:
            print(f"Skipping extraction: All files from {file} are already present in {path}")
            return

        def extract_member(member: str) -> None:
            zipObj.extract(member, path=path)

        # Use ThreadPoolExecutor to extract files in parallel
        with ThreadPoolExecutor(max_workers=threads) as executor:
            list(executor.map(extract_member, file_list))

    print(f"Unzipped {len(file_list)} files from {file} to {path}")

def download_with_resume(url: str, dest: Union[str, Path], retry: int = 3) -> None:
    """
    Downloads a file from a given URL with support for resuming partial downloads.

    Args:
        url (str): The URL of the file to download.
        dest (str or Path): The destination file path where the downloaded file will be saved.
        retry (int, optional): The number of retry attempts in case of failure. Defaults to 3.

    Raises:
        Exception: If the download fails with a status code other than 200, 206, or 416.

    Returns:
        None

    Notes:
        - If the file already exists, the function will attempt to resume the download
          from where it left off using the HTTP Range header.
        - If the file is already fully downloaded (HTTP 416 status code), the function
          will print a message and exit without downloading again.
        - The function uses a progress bar from the `tqdm` library to display the download progress.
    """

    headers = {}
    # Make sure the file exists before opening it in append mode
    if not os.path.exists(dest):
        open(dest, 'wb').close()  # Create an empty file

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

def download_one(url: str, 
                 dir: Union[str, Path], 
                 delete: bool = False, 
                 unzip: bool = True, 
                 threads: int = 8, 
                 retry: int = 3) -> None:
    """
    Downloads a file from a given URL, optionally unzips it, and manages retries and cleanup.
    Args:
        url (str): The URL of the file to download.
        dir (str or Path): The directory where the file will be saved.
        delete (bool, optional): If True, deletes the compressed file after extraction. Defaults to False.
        unzip (bool, optional): If True, extracts the file if it is a .gz, .zip, or .tar file. Defaults to True.
        threads (int, optional): Number of threads to use for extraction (if applicable). Defaults to 8.
        retry (int, optional): Number of retry attempts for downloading the file in case of failure. Defaults to 3.
    Returns:
        None
    Behavior:
        - Skips downloading if a marker file indicating successful processing already exists.
        - Skips downloading if the compressed file already exists.
        - Downloads the file with retry logic in case of failures.
        - Extracts the file if it is a .gz, .zip, or .tar file and `unzip` is True.
        - Deletes the compressed file after extraction if `delete` is True.
        - Creates a marker file to indicate successful processing.
    Notes:
        - The function uses `download_with_resume` for downloading files with resume support.
        - Extraction of .zip and .tar files is handled using `unzip_file` and `tarfile` respectively.
        - For .gz files, the `gunzip` command is used.
    """
    
    success = True
    f = Path(dir) / Path(url).name
    marker = f.with_suffix(f.suffix + ".downloaded")  # Marker file

    # Check if the file has already been downloaded and processed
    if marker.exists() and unzip==False:
        print(f"Skipping download and extraction: {f} is already processed.")
        return
        
    # Check if the compressed file already exists
    if f.exists():
        print(f"File already exists: {f}. Skipping download.")
    else:
        # Download the file
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

    # Extract the file if necessary
    if unzip and (f.suffix == ".gz" or is_zipfile(f) or is_tarfile(f)):
        print(f"Unzipping {f}...")
        if is_zipfile(f):
            unzip_file(f, dir, threads=1)  # unzip
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

    # Delete the compressed file if requested
    if delete and f.exists():
        f.unlink()  # Remove the compressed file
        print(f"Deleted compressed file: {f}")

    # Create the marker file
    marker.touch()
    print(f"Created marker file: {marker}")

def download(url: Union[str, List[str], Path], 
             dir: Union[str, Path] = ".", 
             unzip: bool = True, 
             delete: bool = False, 
             threads: int = 16, 
             retry: int = 3) -> None:
    """
    Downloads a file or multiple files from the given URL(s) to the specified directory.
    Args:
        url (str or list or Path): The URL or list of URLs to download.
        dir (str or Path, optional): The directory where the downloaded files will be saved. Defaults to the current directory ".".
        unzip (bool, optional): If True, automatically unzips the downloaded file(s) if they are compressed. Defaults to True.
        delete (bool, optional): If True, deletes the original compressed file(s) after unzipping. Defaults to False.
        threads (int, optional): The number of threads to use for downloading files. Defaults to 8.
        retry (int, optional): The number of retry attempts for failed downloads. Defaults to 3.
    Returns:
        None
    Notes:
        - If `threads` is greater than 1, the function uses multithreading to download multiple files concurrently.
        - If a single URL is provided as a string, it is treated as a single file download.
        - The `download_one` function is used internally to handle individual file downloads.
    """
    
    dir = Path(dir)
    os.makedirs(dir, exist_ok=True)  # make directory
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(x[0], x[1], delete=delete, unzip=unzip, threads=threads, retry=retry), 
                 zip(url, repeat(dir)))  # multithreaded        
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir, delete=delete, unzip=unzip, threads=threads, retry=retry)

def xyxy2xywhn(xyxy: np.ndarray, 
               w: Union[int, float] = 640, 
               h: Union[int, float] = 640, 
               clip: bool = False, 
               eps: float = 0.0) -> np.ndarray:
    """
    Converts bounding box coordinates from (x_min, y_min, x_max, y_max) format 
    to normalized (x_center, y_center, width, height) format.
    Args:
        xyxy (numpy.ndarray): Array of shape (N, 4) containing bounding box 
            coordinates in (x_min, y_min, x_max, y_max) format.
        w (float or int, optional): Width of the image. Defaults to 640.
        h (float or int, optional): Height of the image. Defaults to 640.
        clip (bool, optional): If True, clips the bounding box coordinates 
            to be within the image dimensions. Defaults to False.
        eps (float, optional): Small epsilon value to prevent clipping 
            to exactly the image boundary. Defaults to 0.0.
    Returns:
        numpy.ndarray: Array of shape (N, 4) containing bounding box 
        coordinates in normalized (x_center, y_center, width, height) format.
    """
    
    if clip:
        xyxy[:, 0] = np.maximum(0, np.minimum(xyxy[:, 0], w - eps))
        xyxy[:, 1] = np.maximum(0, np.minimum(xyxy[:, 1], h - eps))
        xyxy[:, 2] = np.maximum(0, np.minimum(xyxy[:, 2], w - eps))
        xyxy[:, 3] = np.maximum(0, np.minimum(xyxy[:, 3], h - eps))
    
    y = xyxy.copy()
    y[:, 0] = ((xyxy[:, 0] + xyxy[:, 2]) / 2) / w  # x center
    y[:, 1] = ((xyxy[:, 1] + xyxy[:, 3]) / 2) / h  # y center
    y[:, 2] = (xyxy[:, 2] - xyxy[:, 0]) / w        # width
    y[:, 3] = (xyxy[:, 3] - xyxy[:, 1]) / h        # height
    return y

def process_annotations_to_yolo(annotations_path: Path, labels: Path, split: str, num_threads: int = 16) -> None:
    """
    Process COCO annotations and convert them to YOLO format using multithreading.
    Args:
        annotations_path (Path): Path to the COCO annotations JSON file.
        labels (Path): Path to the labels directory where YOLO annotations will be saved.
        split (str): Dataset split (e.g., "train" or "val").
        num_threads (int): Number of threads to use for processing.
    """
    if not annotations_path.exists():
        print(f"Annotations file not found: {annotations_path}")
        return

    print(f"Converting {split} annotations to YOLO format...")
    coco = COCO(annotations_path)
    names = [x["name"] for x in coco.loadCats(coco.getCatIds())]

    def process_image(im):
        try:
            width, height = im["width"], im["height"]
            path = Path(im["file_name"])  # image file name

            # Determine patch directory by looking at image path structure
            relative_path = path.parent.relative_to("images")  # Remove "images" from the relative path

            # Create corresponding directory in labels if it doesn't exist
            label_dir = labels / relative_path
            label_dir.mkdir(parents=True, exist_ok=True)

            # Save label file in the same patch directory structure
            label_file = label_dir / path.with_suffix(".txt").name
            if label_file.exists():  # Skip if the label file already exists
                return

            with open(label_file, "a", encoding="utf-8") as file:
                catIds = coco.getCatIds()
                annIds = coco.getAnnIds(imgIds=im["id"], catIds=catIds, iscrowd=None)
                if not annIds:
                    return
                for a in coco.loadAnns(annIds):
                    x, y, w, h = a["bbox"]  # bounding box in xywh (xy is top-left corner)
                    xyxy = np.array([[x, y, x + w, y + h]])  # convert to xyxy format
                    x, y, w, h = xyxy2xywhn(xyxy, w=width, h=height, clip=True)[0]  # convert to xywhn
                    file.write(f"{a['category_id']} {x:.5f} {y:.5f} {w:.5f} {h:.5f}\n")
        except Exception as e:
            print(f"Error processing image {im['file_name']}: {e}")

    # Use ThreadPoolExecutor for multithreaded processing
    imgIds = coco.getImgIds()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(process_image, coco.loadImgs(imgIds)), total=len(imgIds), desc=f"Processing {split}"))

    print(f"Finished processing {split} annotations to YOLO format.")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Download and process the Objects365 dataset.")
    parser.add_argument("--threads", type=int, default=20, help="Number of threads for downloading (default: 20)")
    parser.add_argument("--unzip", action="store_true", help="Unzip files after downloading (default: False)")
    parser.add_argument("--delete", action="store_true", help="Delete compressed files after unzipping (default: False)")
    parser.add_argument("--base_dir", type=str, default="/mnt/d/objects365", help="Base directory for dataset (default: /mnt/d/objects365)")
    parser.add_argument("--process_annotationr", action="store_true", help="process_annotation after downloading (default: False)")
    args = parser.parse_args()

    threads = args.threads
    unzip = args.unzip
    delete = args.delete
    process_annotation = args.process_annotationr
    base_dir = Path(args.base_dir)
    # Set the base directory on local disk E (adapted for WSL)
    base_dir = Path("/mnt/d/objects365")

    # Make sure the base directory exists
    base_dir.mkdir(parents=True, exist_ok=True)

    # Create directories
    for p in ["images", "labels"]:
        for q in ["train", "val"]:
            (base_dir / p / q).mkdir(parents=True, exist_ok=True)

    # Train, Val Splits
    for split, patches in [("train", 50 + 1), ("val", 43 + 1)]:
        print(f"Processing {split} in {patches} patches ...")
        images, labels = base_dir / "images" / split, base_dir / "labels" / split

        # Base URL
        url = f"https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/{split}/"

        # Download annotations
        if split == "train":
            download([f"{url}zhiyuan_objv2_{split}.tar.gz"], dir=base_dir, delete=False)
        elif split == "val":
            download([f"{url}zhiyuan_objv2_{split}.json"], dir=base_dir, delete=False)

        # Download images
        if split == "train":
            print(f"Downloading training images ({patches} patches)...")
            download(
                [f"{url}patch{i}.tar.gz" for i in range(patches)],
                dir=images,
                delete=delete,
                threads=threads,
                unzip=unzip
            )
        elif split == "val":
            print("Downloading validation images v1...")
            download(
                [f"{url}images/v1/patch{i}.tar.gz" for i in range(15 + 1)],
                dir=images,
                delete=delete,
                threads=threads,
                unzip=unzip
            )
            print("Downloading validation images v2...")
            download(
                [f"{url}images/v2/patch{i}.tar.gz" for i in range(16, patches)],
                dir=images,
                delete=delete,
                threads=threads,
                unzip=unzip
            )

        if process_annotation:
            # Process annotations in YOLO format
            print(f"Processing annotations for {split}...")
            annotations_path = base_dir / f"zhiyuan_objv2_{split}.json"
            if split == "train" and not annotations_path.exists():
                tar_path = base_dir / f"zhiyuan_objv2_{split}.tar.gz"
                if tar_path.exists():
                    # Check if annotations have already been extracted
                    extracted_marker = tar_path.with_suffix(".extracted")
                    if extracted_marker.exists():
                        print(f"Annotations already extracted from {tar_path}")
                    else:
                        with tarfile.open(tar_path, "r:*") as tar:
                            tar.extractall(path=base_dir)
                        print(f"Annotations extracted from {tar_path}")
                        extracted_marker.touch()  # Create marker for future runs
            # Process annotations in YOLO format
            print(f"Processing annotations for {split}...")
            annotations_path = base_dir / f"zhiyuan_objv2_{split}.json"
            process_annotations_to_yolo(annotations_path, labels, split, num_threads=threads)

    print("\nDownload and processing of Objects365 dataset completed!")
    print(f"Dataset saved in: {base_dir}")
    print("Number of categories: 365")
    print("Total training images: approximately 1.7 million")
    print("Total validation images: approximately 80,000")
    print("\nWARNING: The Objects365 dataset is available for academic purposes only.")
    print("Read the license in the official documentation: https://www.objects365.org/download.html")