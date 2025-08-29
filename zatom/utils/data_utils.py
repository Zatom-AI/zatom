import os
import tarfile

from zatom.utils import pylogger
from zatom.utils.typing_utils import typecheck

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


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
