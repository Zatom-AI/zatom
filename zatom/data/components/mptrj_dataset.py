import os
import warnings
from typing import Callable, List, Literal, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from zatom.data.components.preprocessing_utils import preprocess_parquet
from zatom.utils.typing_utils import typecheck

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)


class MPtrj(InMemoryDataset):
    """The MPtrj dataset of Materials Project trajectories, as a PyG InMemoryDataset.

    In order to create a torch_geometric.data.InMemoryDataset, you need to implement four fundamental methods:
    - InMemoryDataset.raw_file_names(): A list of files in the raw_dir which needs to be found in order to skip the download.
    - InMemoryDataset.processed_file_names(): A list of files in the processed_dir which needs to be found in order to skip the processing.
    - InMemoryDataset.download(): Downloads raw data into raw_dir.
    - InMemoryDataset.process(): Processes raw data and saves it into the processed_dir.

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
        load: Whether to load the processed dataset into memory.
            (default: `True`)
        force_reload: Whether to re-process the dataset.
            (default: `False`)
        shift: Shift value for energy normalization.
            (default: `0.0`)
        scale: Scale value for energy normalization.
            (default: `1.0`)
        force_scale: Scale value for force normalization.
            (default: `1.0`)
        split: The dataset split to load (train, val, test).
            (default: `train`)
    """

    @typecheck
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        load: bool = True,
        force_reload: bool = False,
        shift: float = 0.0,
        scale: float = 1.0,
        force_scale: float = 1.0,
        split: Literal["train", "val", "test"] = "train",
    ) -> None:
        self.shift = shift
        self.scale = scale
        self.force_scale = force_scale
        self.split = split
        self.split_files = {
            "train": [
                os.path.join("data", f"{self.split}-0000{i}-of-00008.parquet") for i in range(8)
            ],
            "val": [os.path.join("data", f"{self.split}-00000-of-00001.parquet")],
            "test": [os.path.join("data", f"{self.split}-00000-of-00001.parquet")],
        }[split]

        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)

        if load:
            self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """Return the list of raw file names."""
        return self.split_files

    @property
    def processed_file_names(self) -> List[str]:
        """Return the list of processed file names."""
        return [f"{self.split}.pt"]

    def download(self) -> None:
        """Download the dataset."""
        from huggingface_hub import hf_hub_download

        for split_file in self.split_files:
            hf_hub_download(  # nosec
                repo_id="nimashoghi/mptrj",
                filename=split_file,
                repo_type="dataset",
                local_dir=os.path.join(self.root, "raw"),
            )

    def process(self) -> None:
        """Process the dataset."""
        if os.path.exists(os.path.join(self.root, "raw", f"{self.split}.pt")):
            cached_data = torch.load(
                os.path.join(self.root, "raw", f"{self.split}.pt"), weights_only=False
            )  # nosec
        else:
            parquet_files = [
                os.path.join(self.root, "raw", split_file) for split_file in self.split_files
            ]
            parquet_df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
            cached_data = preprocess_parquet(
                parquet_df,
                prop_list=["energy_referenced", "forces"],
                num_workers=32,
            )
            torch.save(cached_data, os.path.join(self.root, "raw", f"{self.split}.pt"))

        data_list = []
        for data_dict in cached_data:
            # Extract attributes from data_dict
            graph_arrays = data_dict["graph_arrays"]

            atom_types = graph_arrays["atom_types"]
            frac_coords = graph_arrays["frac_coords"]
            cell = graph_arrays["cell"]
            lattices = graph_arrays["lattices"]
            lengths = graph_arrays["lengths"]
            angles = graph_arrays["angles"]
            num_atoms = graph_arrays["num_atoms"]

            # Prepare target values (energy and forces)
            energy = torch.tensor([data_dict["energy_referenced"].astype(np.float32)])
            forces = torch.from_numpy(np.stack(data_dict["forces"]).astype(np.float32))

            y = torch.cat([energy.repeat(num_atoms, 1), forces], dim=-1)

            # Normalize the lengths of lattice vectors, which makes
            # lengths for materials of different sizes at same scale
            _lengths = lengths / float(num_atoms) ** (1 / 3)
            # Convert angles of lattice vectors to be in radians
            _angles = np.radians(angles)
            # Add scaled lengths and angles to graph arrays
            graph_arrays["length_scaled"] = _lengths
            graph_arrays["angles_radians"] = _angles
            graph_arrays["lattices_scaled"] = np.concatenate([_lengths, _angles])

            data = Data(
                id=data_dict["mp_id"],
                atom_types=torch.LongTensor(atom_types),
                frac_coords=torch.Tensor(frac_coords),
                cell=torch.Tensor(cell).unsqueeze(0),
                lattices=torch.Tensor(lattices).unsqueeze(0),
                lattices_scaled=torch.Tensor(graph_arrays["lattices_scaled"]).unsqueeze(0),
                lengths=torch.Tensor(lengths).view(1, -1),
                lengths_scaled=torch.Tensor(graph_arrays["length_scaled"]).view(1, -1),
                angles=torch.Tensor(angles).view(1, -1),
                angles_radians=torch.Tensor(graph_arrays["angles_radians"]).view(1, -1),
                num_atoms=torch.LongTensor([num_atoms]),
                num_nodes=torch.LongTensor([num_atoms]),  # special attribute used for PyG batching
                token_idx=torch.arange(num_atoms),
                dataset_idx=torch.tensor(
                    [5], dtype=torch.long
                ),  # 5 --> Indicates periodic/crystal
                y=y,
            )
            # 3D coordinates (NOTE: do not zero-center prior to graph construction)
            data.pos = torch.einsum(
                "bi,bij->bj",
                data.frac_coords,
                torch.repeat_interleave(data.cell, data.num_atoms, dim=0),
            )
            # Space group number
            data.spacegroup = torch.LongTensor([data_dict["spacegroup"]])

            # Dummy charge and spin (not used with MPtrj currently)
            data.charge = torch.tensor(0, dtype=torch.float32)
            data.spin = torch.tensor(0, dtype=torch.long)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, os.path.join(self.root, "processed", f"{self.split}.pt"))

    @typecheck
    def get(self, idx: int) -> Data:
        """Get the data object at index idx and normalize its energies and forces.

        Args:
            idx: Index of the data object to retrieve.

        Returns:
            The data object at index idx.
        """
        data = super().get(idx)

        energy = data.y[0, 0:1]
        forces = data.y[:, 1:4]

        energy = (energy - self.shift) / self.scale
        forces = forces / self.force_scale

        num_atoms = data.num_atoms.item()
        data.y = torch.cat([energy.repeat(num_atoms, 1), forces], dim=-1)

        data.charge = torch.tensor(0, dtype=torch.float32)
        data.spin = torch.tensor(0, dtype=torch.long)

        return data
