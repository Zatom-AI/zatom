import os
import warnings
from typing import Callable, List, Literal, Optional

import torch
from fairchem.core.datasets import AseDBDataset
from torch_geometric.data import Data, Dataset, download_url

from zatom.utils import pylogger
from zatom.utils.data_utils import extract_tar_gz

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class OMol25(Dataset):
    """The OMol25 dataset from FAIR at Meta, as a PyG Dataset.

    In order to create a torch_geometric.data.Dataset, you need to implement five fundamental methods:
    - Dataset.raw_file_names(): A list of files in the raw_dir which needs to be found in order to skip the download.
    - Dataset.processed_file_names(): A list of files in the processed_dir which needs to be found in order to skip the processing.
    - Dataset.download(): Downloads raw data into raw_dir.
    - Dataset.len(): Returns the number of examples in the dataset.
    - Dataset.get(): Gets a single example from the dataset.

    Args:
        root: Root directory where the dataset should be saved.
        transform: A function/transform that takes in an
            `torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: `None`)
        pre_transform: A function/transform that takes in
            an `torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: `None`)
        pre_filter: A function that takes in an
            `torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: `None`)
        force_reload: Whether to re-process the dataset.
            (default: `False`)
        split: The dataset split to load (train, val, test).
            (default: `train`)
        subset: The (training) dataset subset to load ("" or "_4M").
            (default: `""`)
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        split: Literal["train", "val", "test"] = "train",
        subset: Literal["", "_4M"] = "",
    ) -> None:
        self.split = split
        self.subset = subset
        self.dataset_path = os.path.join(root, "raw", f"{split}{subset}")

        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)

        self.dataset = AseDBDataset({"src": self.dataset_path})

    @property
    def raw_file_names(self) -> List[str]:
        """Return the list of raw file names."""
        return (
            os.listdir(self.dataset_path)
            if os.path.exists(self.dataset_path)
            else [f"{self.split}{self.subset}.tar.gz"]
        )

    @property
    def processed_file_names(self) -> List[str]:
        """Return the list of processed file names."""
        return (
            os.listdir(self.dataset_path)
            if os.path.exists(self.dataset_path)
            else [f"{self.split}{self.subset}.tar.gz"]
        )

    def download(self) -> None:
        """Download the dataset."""
        # Dataset files
        url = f"https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/{self.split}{self.subset}.tar.gz"
        output_file = os.path.join(self.root, "raw", f"{self.split}{self.subset}.tar.gz")

        # Skip download and extraction if raw data directory exists
        if os.path.exists(self.dataset_path) and os.listdir(self.dataset_path):
            return

        # Download raw data if necessary
        if os.path.exists(output_file):
            log.info(f"OMol25 data file {output_file} already exists. Directly extracting.")
        else:
            download_url(url, os.path.dirname(output_file))

        # Extract raw data
        extract_tar_gz(output_file, os.path.dirname(self.dataset_path))

    def len(self) -> int:
        """Return the number of examples."""
        return len(self.dataset)

    def get(self, idx) -> Data:
        """Get a single example."""
        atoms = self.dataset.get_atoms(idx)

        num_atoms = len(atoms)
        atoms_to_keep = torch.ones((num_atoms,), dtype=torch.bool)

        data = Data(
            id=f"omol25:{atoms.info['source']}",
            atom_types=torch.LongTensor(atoms.get_atomic_numbers()[atoms_to_keep]),
            pos=torch.Tensor(atoms.positions[atoms_to_keep]),
            frac_coords=torch.zeros_like(torch.Tensor(atoms.positions[atoms_to_keep])),
            cell=torch.zeros((1, 3, 3)),
            lattices=torch.zeros(1, 6),
            lattices_scaled=torch.zeros(1, 6),
            lengths=torch.zeros(1, 3),
            lengths_scaled=torch.zeros(1, 3),
            angles=torch.zeros(1, 3),
            angles_radians=torch.zeros(1, 3),
            num_atoms=torch.LongTensor([num_atoms]),
            num_nodes=torch.LongTensor([num_atoms]),  # Special attribute used for PyG batching
            spacegroup=torch.zeros(1, dtype=torch.long),  # Null spacegroup
            token_idx=torch.arange(num_atoms),
            dataset_idx=torch.tensor(
                [1], dtype=torch.long
            ),  # 1 --> Indicates non-periodic/molecule
        )

        if self.pre_filter is not None and not self.pre_filter(data):
            return data
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        return data
