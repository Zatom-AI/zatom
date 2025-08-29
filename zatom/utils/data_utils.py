import os
import tarfile

import requests
from tqdm import tqdm

from zatom.utils import pylogger
from zatom.utils.typing_utils import typecheck

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@typecheck
def download_file(url: str, output_path: str, verbose: bool = True):
    """Download a file in streaming mode.

    Args:
        url: The URL of the file to download.
        output_path: The local path where the file should be saved.
    """
    if verbose:
        log.info(f"Downloading {output_path}...")

    with requests.get(url, stream=True, timeout=5) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        block_size = 1024 * 1024  # 1 MB
        progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)

        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=block_size):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        progress_bar.close()

        if total_size != 0 and progress_bar.n != total_size:
            if verbose:
                log.warning("Download size mismatch!")


@typecheck
def extract_tar_gz(file_path: str, extract_to: str, verbose: bool = True):
    """Extract a `tar.gz` file.

    Args:
        file_path: The path to the tar.gz file.
        extract_to: The directory to extract the contents to.
    """
    if verbose:
        log.info(f"Extracting {file_path} to {extract_to}...")

    os.makedirs(extract_to, exist_ok=True)
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=extract_to)  # nosec

    if verbose:
        log.info(f"Extracted {file_path} to {extract_to}.")
