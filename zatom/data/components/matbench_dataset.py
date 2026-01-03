import json
import os
from pathlib import Path
from typing import Callable, List, Optional, Union

import torch
from pymatgen.core import Structure
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from zatom.data.components.preprocessing_utils import (
    cart_to_frac_coords,
    lattice_matrix_to_params,
)
from zatom.utils.data_utils import hf_download_file
from zatom.utils.typing_utils import typecheck

MATBENCH_QM9_TARGET_NAME_TO_LITERATURE_SCALE = {
    # NOTE: For now, limited unit conversions are made for Matbench tasks
    # Reference: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html
    "mu": 1.0,
    "alpha": 1.0,
    "homo": 1.0,
    "lumo": 1.0,
    "gap": 1000.0,  # 1 electronvolt (eV) = 1000 millielectronvolts (meV)
    "r2": 1.0,
    "zpve": 1.0,
    "U0": 1.0,
    "U": 1.0,
    "H": 1.0,
    "G": 1.0,
    "Cv": 1.0,
    "U0_atom": 1.0,
    "U_atom": 1.0,
    "H_atom": 1.0,
    "G_atom": 1.0,
    "A": 1.0,
    "B": 1.0,
    "C": 1.0,
}


@typecheck
def get_matbench_data_path(task_name: str, local_root: Optional[Union[str, Path]] = None) -> str:
    """Download and return path to Matbench data for a given task name.

    Args:
        task_name: Name of the Matbench task.
        local_root: Optional local root directory to store the dataset. If None, will use default cache directory.

    Returns:
        Path to the downloaded Matbench dataset.
    """
    path = hf_download_file(
        repo_id="Ty-Perez/matbench_properties",
        filename=f"{task_name}.tar.gz",
        local_root=local_root,
    )
    return str(path)


class Matbench(InMemoryDataset):
    """Base class for Matbench datasets.

    In order to create a torch_geometric.data.InMemoryDataset, you need to implement four fundamental methods:
    - InMemoryDataset.raw_file_names(): A list of files in the raw_dir which needs to be found in order to skip the download.
    - InMemoryDataset.processed_file_names(): A list of files in the processed_dir which needs to be found in order to skip the processing.
    - InMemoryDataset.download(): Downloads raw data into raw_dir.
    - InMemoryDataset.process(): Processes raw data and saves it into the processed_dir.

    Available Tasks:
        - matbench_dielectric
        - matbench_expt_gap
        - matbench_expt_is_metal
        - matbench_glass
        - matbench_jdft2d
        - matbench_log_gvrh
        - matbench_log_kvrh
        - matbench_mp_e_form
        - matbench_mp_gap
        - matbench_mp_is_metal
        - matbench_perovskites
        - matbench_phonons
        - matbench_steels

    Available Splits:
        - train
        - test

    Args:
        root: Root directory where the dataset is stored.
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
        load: Whether to load the processed dataset.
            (default: `True`)
        force_reload: Whether to re-process the dataset.
            (default: `False`)
        task_name: Name of the Matbench task to load.
        split: Data split to use ("train" or "test").
        fold_idx: Index of the training fold to use.
        shift: Optional tensor by which to shift the target properties.
        scale: Optional tensor by which to scale the target properties.
        dataset_dir: Directory where the Matbench dataset is stored. If None, will download the dataset.
    """

    avail_splits = ["train", "test"]
    avail_tasks = {
        # NOTE: Maps from Matbench task name to QM9 task index
        "matbench_dielectric": 0,  # mu
        "matbench_expt_gap": 1,  # alpha
        "matbench_expt_is_metal": 2,  # homo
        "matbench_glass": 3,  # lumo
        "matbench_mp_gap": 4,  # gap
        "matbench_jdft2d": 5,  # r2
        "matbench_log_gvrh": 6,  # zpve
        "matbench_log_kvrh": 7,  # U0
        "matbench_mp_e_form": 8,  # U
        "matbench_mp_is_metal": 9,  # H
        "matbench_perovskites": 10,  # G
        "matbench_phonons": 11,  # Cv
        "matbench_steels": 12,  # U0_atom
        # ... Add more tasks as needed
        "13?": 13,  # U_atom
        "14?": 14,  # H_atom
        "15?": 15,  # G_atom
        "16?": 16,  # A
        "17?": 17,  # B
        "18?": 18,  # C
    }

    @typecheck
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        load: bool = True,
        force_reload: bool = False,
        task_name: str = "matbench_mp_gap",
        split: str = "train",
        fold_idx: int = 0,
        shift: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        dataset_dir: Optional[str] = None,
    ):
        assert (
            task_name in self.avail_tasks and "?" not in task_name
        ), f"Task {task_name} is not one of the available tasks: {[k for k in self.avail_tasks if '?' not in k]}"

        if dataset_dir is None:
            dataset_dir = get_matbench_data_path(task_name, local_root=root)

        self.dataset_dir = dataset_dir
        assert os.path.exists(
            dataset_dir
        ), f"Dataset directory for Matbench {dataset_dir} must exist."

        data_files = [f for f in os.listdir(dataset_dir) if f.endswith(".pt")]
        summary_files = [f for f in os.listdir(dataset_dir) if f.endswith(".json")]

        assert len(data_files) > 0, f"No data files found in {dataset_dir}."
        assert (
            len(summary_files) == 1
        ), f"Expected one JSON file in {dataset_dir}, but found {len(summary_files)}. Please remove all JSON files except `summary.json`."

        # Load summary file
        summary_file = os.path.join(dataset_dir, summary_files[0])
        with open(summary_file) as f:
            summary = json.load(f)

        # Collect metadata
        self.mb_task = summary["task_name"]
        self.mb_target_name = summary["target_key"]
        self.metadata = summary["metadata"]
        self.all_sample_ids = summary["all_ids"]
        self.targets_dict = summary["targets"]

        assert len(self.all_sample_ids) == len(
            data_files
        ), f"Number of data files {len(data_files)} does not match number of sample IDs {len(self.all_sample_ids)}."

        # Parse training folds
        fold_idx = str(int(fold_idx))
        self.all_folds = summary["training_folds"]
        avail_folds = list(self.all_folds.keys())

        assert (
            fold_idx in avail_folds
        ), f"fold_idx {fold_idx} not in available folds: {avail_folds}"
        assert (
            split in self.avail_splits
        ), f"split {split} not in available splits: {self.avail_splits}"

        self.split = split
        self.fold_idx = fold_idx
        self.shift = shift
        self.scale = scale
        self.data = self.all_folds[fold_idx][split]

        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)

        if load:
            self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """Return the list of raw file names."""
        return (
            os.listdir(self.dataset_dir)
            if os.path.exists(self.dataset_dir)
            else [f"{self.mb_task}.tar.gz"]
        )

    @property
    def processed_file_names(self) -> List[str]:
        """Return the list of processed file names."""
        return [f"{self.mb_task}_{self.split}_{self.fold_idx}.pt"]

    def download(self) -> None:
        """Download the dataset."""
        # NOTE: This is instead handled by `get_matbench_data_path()`
        pass

    @typecheck
    def info(self) -> str:
        """Return basic information about dataset retrieved from metadata.

        Returns:
            String containing basic information about the dataset.
        """
        out = []
        for key, value in self.metadata.items():
            if key == "bibtex_refs":
                # pass
                continue
            out.append(f"{key}: {value}\n")
        return "".join(out)

    @typecheck
    def __repr__(self) -> str:
        """Return basic information about dataset retrieved from metadata.

        Returns:
            String containing basic information about the dataset.
        """
        out_str = [f"Name: {self.mb_task}\n"]
        out_str += [f"Fold {self.fold_idx}: {self.split}\n"]
        out_str += [f"size: {len(self.data)} / {len(self.all_sample_ids)}\n"]
        return "".join(out_str)

    @typecheck
    def load_sample(self, name: str) -> tuple:
        """Load a single sample from the dataset.

        Args:
            name: Name of the sample to load.

        Returns:
            A tuple containing the structure and target value.
        """
        path = os.path.join(self.dataset_dir, f"{name}.pt")
        sample_dict = torch.load(path, weights_only=True)

        structure = Structure.from_dict(sample_dict["structure"])
        target = sample_dict["target"]

        return structure, target

    def process(self) -> None:
        """Process the dataset."""
        data_list = []
        for idx in tqdm(range(len(self.data))):
            sample_name = self.data[idx]

            structure, target = self.load_sample(sample_name)
            num_atoms = len(structure)

            # Prepare target values (properties)
            y = torch.tensor([[torch.nan] * len(self.avail_tasks)], dtype=torch.float32)
            y[0, self.avail_tasks[self.mb_task]] = torch.tensor(target, dtype=torch.float32)

            # Calculate and store the lattice params for use elsewhere
            a, b, c, alpha, beta, gamma = lattice_matrix_to_params(structure.lattice.matrix)
            lengths = torch.tensor([a, b, c], dtype=torch.float32)
            angles = torch.tensor([alpha, beta, gamma], dtype=torch.float32)
            lattices = torch.cat([lengths, angles])

            # --- Perform conversions using the ORIGINAL lattice matrix ---
            original_lattice_matrix = torch.tensor(
                structure.lattice.matrix, dtype=torch.float32
            ).reshape(1, 3, 3)
            original_cart_coords = torch.tensor(structure.cart_coords, dtype=torch.float32)

            frac_coords = cart_to_frac_coords(
                cart_coords=original_cart_coords,
                lattices=original_lattice_matrix,
                num_atoms=num_atoms,
            )

            # Normalize the lengths of lattice vectors, which makes
            # lengths for materials of different sizes at same scale
            _lengths = lengths / float(num_atoms) ** (1 / 3)
            # Convert angles of lattice vectors to be in radians
            _angles = torch.deg2rad(angles)
            # Add scaled lengths and angles to graph arrays
            lengths_scaled = _lengths
            angles_radians = _angles
            lattices_scaled = torch.cat([_lengths, _angles])

            data = Data(
                id=f"matbench:{self.mb_task}_{idx}",
                atom_types=torch.tensor(structure.atomic_numbers, dtype=torch.long),
                # pos=original_cart_coords,
                frac_coords=frac_coords,
                cell=original_lattice_matrix,
                # pbc=torch.tensor(structure.lattice.pbc, dtype=torch.bool).reshape(1, 3),
                lattices=lattices.unsqueeze(0),
                lattices_scaled=lattices_scaled.unsqueeze(0),
                lengths=lengths.view(1, -1),
                lengths_scaled=lengths_scaled.view(1, -1),
                angles=angles.view(1, -1),
                angles_radians=angles_radians.view(1, -1),
                num_atoms=torch.LongTensor([num_atoms]),
                num_nodes=torch.LongTensor([num_atoms]),  # special attribute used for PyG batching
                token_idx=torch.arange(num_atoms),
                dataset_idx=torch.tensor(
                    [6], dtype=torch.long
                ),  # 6 --> Indicates periodic/crystal
                y=y,
            )

            # 3D coordinates (NOTE: do not zero-center prior to graph construction)
            data.pos = torch.einsum(
                "bi,bij->bj",
                data.frac_coords,
                torch.repeat_interleave(data.cell, data.num_atoms, dim=0),
            )

            # Dummy space group number
            data.spacegroup = torch.tensor([1], dtype=torch.long)

            # Dummy charge and spin (not used with Matbench currently)
            data.charge = torch.tensor(0, dtype=torch.float32)
            data.spin = torch.tensor(0, dtype=torch.long)

            data_list.append(data)

        self.save(
            data_list,
            os.path.join(
                self.root, "processed", f"{self.mb_task}_{self.split}_{self.fold_idx}.pt"
            ),
        )

    @typecheck
    def get(self, idx: int) -> Data:
        """Get the data object at index idx and normalize its properties.

        Args:
            idx: Index of the data object to retrieve.

        Returns:
            The data object at index idx.
        """
        data = super().get(idx)

        if self.shift is not None and self.scale is not None:
            data.y = (data.y - self.shift) / self.scale

        data.charge = torch.tensor(0, dtype=torch.float32)
        data.spin = torch.tensor(0, dtype=torch.long)

        return data
