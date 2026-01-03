"""Copyright (c) Meta Platforms, Inc.

and affiliates.
"""

import os
import warnings
from typing import Callable, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from zatom.data.components.preprocessing_utils import preprocess
from zatom.utils.typing_utils import typecheck

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)


class MP20(InMemoryDataset):
    """The MP20 dataset from Materials Project, as a PyG InMemoryDataset.

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
        force_reload: Whether to re-process the dataset.
            (default: `False`)
    """

    @typecheck
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """Return the list of raw file names."""
        return ["all.csv"]

    @property
    def processed_file_names(self) -> List[str]:
        """Return the list of processed file names."""
        return ["mp20.pt"]

    def download(self) -> None:
        """Download the dataset."""
        from huggingface_hub import hf_hub_download

        hf_hub_download(  # nosec
            repo_id="chaitjo/MP20_ADiT",
            filename=os.path.join("raw", "all.csv"),
            repo_type="dataset",
            local_dir=self.root,
        )

    def process(self) -> None:
        """Process the dataset."""
        if os.path.exists(os.path.join(self.root, "raw", "all.pt")):
            cached_data = torch.load(
                os.path.join(self.root, "raw", "all.pt"), weights_only=False
            )  # nosec
        else:
            cached_data = preprocess(
                os.path.join(self.root, "raw", "all.csv"),
                niggli=True,
                primitive=False,
                graph_method="crystalnn",
                prop_list=["formation_energy_per_atom"],
                use_space_group=True,
                tol=0.1,
                num_workers=32,
            )
            torch.save(cached_data, os.path.join(self.root, "raw", "all.pt"))

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
                    [0], dtype=torch.long
                ),  # 0 --> Indicates periodic/crystal
            )
            # 3D coordinates (NOTE: do not zero-center prior to graph construction)
            data.pos = torch.einsum(
                "bi,bij->bj",
                data.frac_coords,
                torch.repeat_interleave(data.cell, data.num_atoms, dim=0),
            )
            # Space group number
            data.spacegroup = torch.LongTensor([data_dict["spacegroup"]])

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, os.path.join(self.root, "processed", "mp20.pt"))
