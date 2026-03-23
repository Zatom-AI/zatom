"""This script provides simple utilities to download any or all datasets used for Zatom.

Examples:
    Download a single dataset:
        $ python zatom/data/download_data.py --dataset qm9

    Download all datasets:
        $ python zatom/data/download_data.py --dataset all

    Use in in Python dataset class or notebook:
        from zatom.data.download_data import get_zatom_dataset
        qm9_path = get_zatom_dataset("qm9", root="data/")
        print(f"Dataset saved to: {qm9_path}")
"""

import argparse

from zatom.utils.data_utils import hf_download_repo
from zatom.utils.typing_utils import typecheck

_avail_datasets = [
    "qm9",
    "geom",
    "qmof",
    "mptrj",
    "mp_20",
    "matbench",
    "omol25",
]
_datasets_requiring_raw_files = [
    "qm9",
    "mp_20",
    "matbench",
    "omol25",
]  # Datasets that require raw files to be downloaded

# Set global dataset path
_dataset_path = "data/"


@typecheck
def get_zatom_dataset(dataset_name: str, root: str = _dataset_path) -> str:
    """Download a specific dataset from an anonymous organization on Hugging Face.

    NOTE: This function can be added to the dataset loaders to automatically
    download the dataset if it is not found in the local path.

    Args:
        dataset_name: Name of the dataset to download. Must be one of the available
            datasets: ["qm9", "geom", "qmof", "mptrj", "mp_20", "matbench", "omol25"].
        root: Local directory to save the downloaded dataset. Default is "data/".

    Returns:
        The local path where the dataset is saved.
    """
    assert (
        dataset_name in _avail_datasets
    ), f"Dataset {dataset_name} not found. Available datasets: {_avail_datasets}"
    dataset_path = hf_download_repo(
        repo_id=f"ANON/{dataset_name}",
        local_root=root,  # Root directory to save the dataset
        name_by_subdir=True,  # Each dataset repo is saved in a subdirectory named after the dataset
        ignore_files=(
            None if dataset_name in _datasets_requiring_raw_files else ["raw.tar.gz"]
        ),  # Ignore unnecessary raw files to save space
    )
    return dataset_path


def download_all_datasets():
    """Download all datasets from an anonymous organization on Hugging Face."""
    for dataset_name in _avail_datasets:
        print(f"Downloading {dataset_name} dataset...")
        data_path = get_zatom_dataset(dataset_name)
        print(f"Dataset {dataset_name} downloaded to: {data_path}")
    print("All datasets downloaded successfully.")


def main():
    """Main function to parse command-line arguments and download the specified dataset(s)."""
    parser = argparse.ArgumentParser(
        description="Download datasets from an anonymous organization on Hugging Face"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=f"Name of dataset to download. Available: {', '.join(_avail_datasets)}, or 'all' to download all datasets.",
    )

    args = parser.parse_args()

    if args.dataset.lower() == "all":
        download_all_datasets()
        print("Downloading all datasets...")
    else:
        print(f"Downloading {args.dataset} dataset...")
        data_path = get_zatom_dataset(args.dataset, root=_dataset_path)
        print(f"Dataset {args.dataset} downloaded to: {data_path}")


if __name__ == "__main__":
    main()
