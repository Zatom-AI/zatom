"""Copyright (c) Meta Platforms, Inc.

and affiliates.
"""

import os
import pickle  # nosec
import warnings
from typing import Callable, List, Literal, Optional

import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset, download_url
from tqdm import tqdm

from zatom.utils import pylogger

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class GEOM(InMemoryDataset):
    """GEOM-Drugs dataset as a PyG InMemoryDataset.

    In order to create a torch_geometric.data.InMemoryDataset, you need to implement four fundamental methods:
    - InMemoryDataset.raw_file_names(): A list of files in the raw_dir which needs to be found in order to skip the download.
    - InMemoryDataset.processed_file_names(): A list of files in the processed_dir which needs to be found in order to skip the processing.
    - InMemoryDataset.download(): Downloads raw data into raw_dir.
    - InMemoryDataset.process(): Processes raw data and saves it into the processed_dir.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            `torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: `None`)
        pre_transform (callable, optional): A function/transform that takes in
            an `torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: `None`)
        pre_filter (callable, optional): A function that takes in an
            `torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: `None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: `False`)
        load: (bool, optional): Whether to load the processed dataset into memory.
        split: The dataset split to load (train, val, test).
            (default: `train`)
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        load: bool = True,
        split: Literal["train", "val", "test"] = "train",
    ) -> None:
        self.split = split
        self.dataset_path = os.path.join(root, "raw", split)
        self.pickle_file = os.path.join(self.dataset_path, f"{self.split}_data.pickle")
        self.processed_file = os.path.join(root, "processed", f"{self.split}.pt")
        self.processed_smiles_file = os.path.join(root, "processed", f"{self.split}_smiles.pt")
        self.processed_num_nodes_file = os.path.join(
            root, "processed", f"{self.split}_num_nodes.pt"
        )

        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)

        if load:
            self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """Return the list of raw file names."""
        return [os.path.basename(self.pickle_file)]

    @property
    def processed_file_names(self) -> List[str]:
        """Return the list of processed file names."""
        return [
            os.path.basename(self.processed_file),
            os.path.basename(self.processed_smiles_file),
            os.path.basename(self.processed_num_nodes_file),
        ]

    def download(self) -> None:
        """Download the dataset."""
        # Dataset files
        url = f"https://bits.csb.pitt.edu/files/geom_raw/{self.split}_data.pickle"

        # Skip download if pickle file already exists
        if os.path.exists(self.pickle_file):
            return

        # Download raw data if necessary
        download_url(url, os.path.dirname(self.pickle_file))

    def process(self) -> None:
        """Preprocess the dataset."""
        with open(self.pickle_file, "rb") as f:
            raw_data_list = pickle.load(f)  # nosec

        log.info(f"Loaded {len(raw_data_list)} GEOM-Drugs entries from {self.split} set")

        pyg_data_list = []
        smiles_list = []
        num_nodes_list = []
        for entry_idx, entry in tqdm(
            enumerate(raw_data_list),
            total=len(raw_data_list),
            desc="Processing GEOM-Drugs entries",
        ):

            smiles, data = entry
            for conformer_idx, mol in enumerate(data):
                if conformer_idx >= 5:
                    break  # NOTE: GEOM-Drugs only considers the first five 3D conformations

                N = mol.GetNumAtoms()

                # 3D coordinates
                pos = mol.GetConformer().GetPositions()
                pos = torch.tensor(pos, dtype=torch.float32)

                # Atom types
                atomic_number = []
                for atom in mol.GetAtoms():
                    atomic_number.append(atom.GetAtomicNum())
                z = torch.tensor(atomic_number, dtype=torch.long)

                # Metadata
                smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                id = f"geom_{self.split}_{entry_idx}_{conformer_idx}"

                pyg_data = Data(
                    id=id,
                    atom_types=z,
                    pos=pos,
                    frac_coords=torch.zeros_like(pos),
                    cell=torch.zeros((1, 3, 3)),
                    lattices=torch.zeros(1, 6),
                    lattices_scaled=torch.zeros(1, 6),
                    lengths=torch.zeros(1, 3),
                    lengths_scaled=torch.zeros(1, 3),
                    angles=torch.zeros(1, 3),
                    angles_radians=torch.zeros(1, 3),
                    num_atoms=torch.LongTensor([N]),
                    num_nodes=torch.LongTensor([N]),  # Special attribute used for PyG batching
                    spacegroup=torch.zeros(1, dtype=torch.long),  # Null spacegroup
                    token_idx=torch.arange(N),
                    dataset_idx=torch.tensor(
                        [4], dtype=torch.long
                    ),  # 4 --> Indicates non-periodic/molecule
                )

                if self.pre_filter is not None and not self.pre_filter(pyg_data):
                    continue
                if self.pre_transform is not None:
                    pyg_data = self.pre_transform(pyg_data)

                pyg_data_list.append(pyg_data)
                smiles_list.append(smiles)
                num_nodes_list.append(N)

        # Save the data
        log.info(f"Saving {len(pyg_data_list)} GEOM-Drugs entries to {self.split} set")
        self.save(pyg_data_list, self.processed_file)
        torch.save(smiles_list, self.processed_smiles_file)
        torch.save(torch.tensor(num_nodes_list), self.processed_num_nodes_file)
