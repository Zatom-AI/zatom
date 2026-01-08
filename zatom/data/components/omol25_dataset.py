import os
import threading
import warnings
from typing import Callable, List, Literal, Optional

import numpy as np
import torch
from fairchem.core.datasets import AseDBDataset
from torch_geometric.data import Data, Dataset, download_url

from zatom.utils import pylogger
from zatom.utils.data_utils import extract_tar_gz, normalize_energy
from zatom.utils.typing_utils import typecheck

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


# Constants
ELEMENT_SYMBOLS = [
    # NOTE: Taken from https://github.com/niazoys/PlatonicTransformers/blob/fe2cebb780d94d5fb207e975194a5996c843863e/datasets/omol.py#L101C9-L111C22
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
]

# A simple length cache for database
LENGTH_CACHE = {}
LENGTH_LOCK = threading.Lock()


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
        energy_coefficients: Optional per-element energy coefficients for energy normalization.
            If `None`, energy and forces will be set to zero.
            (default: `None`)
        shift: Shift value for energy normalization.
            (default: `0.0`)
        scale: Scale value for energy normalization.
            (default: `1.0`)
    """

    @typecheck
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        split: Literal["train", "val", "test"] = "train",
        subset: Literal["", "_4M"] = "",
        energy_coefficients: Optional[dict] = None,
        shift: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        self.split = split
        self.subset = subset
        self.energy_coefficients = energy_coefficients
        self.shift = shift
        self.scale = scale
        self.dataset_path = os.path.join(root, "raw", f"{split}{subset}")

        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)

        # Avoid creating AseDBDataset here - defer to workers
        # Instead, just get dataset length safely
        self.dataset = None  # NOTE: Each thread/process will have its own instance

        # Get dataset length safely (using cache)
        cache_key = self.dataset_path
        with LENGTH_LOCK:
            if cache_key in LENGTH_CACHE:
                dataset_length = LENGTH_CACHE[cache_key]
            else:
                # One-time initialization in the main process only for length
                # NOTE: This instance will be discarded immediately after getting length
                tmp_dataset = AseDBDataset({"src": self.dataset_path})
                dataset_length = len(tmp_dataset)
                LENGTH_CACHE[cache_key] = dataset_length
                del tmp_dataset

        self.dataset_length = dataset_length
        self.dataset_info = {
            "name": "omol25",
            "atom_encoder": {
                symbol: i + 1 for i, symbol in enumerate(ELEMENT_SYMBOLS)
            },  # NOTE: Start from 1, not 0
            "atom_decoder": ELEMENT_SYMBOLS,
        }

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

    def _init_dataset(self):
        """Safely initialize dataset in the worker process when needed."""
        if self.dataset is None:
            self.dataset = AseDBDataset({"src": self.dataset_path})
        return self.dataset

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

    @typecheck
    def len(self) -> int:
        """Return the number of examples."""
        return self.dataset_length

    def _get_energy_and_forces(self, atoms):
        """Safely get energy and forces from atoms object, handling cases where calculator might not be attached.
        
        This is necessary because when loading atoms from AseDBDataset in DataLoader worker processes,
        the calculator object might not be properly attached, even though the energy/forces data is stored
        in the database. This method tries multiple fallback strategies to retrieve the data.
        """
        # Try to get energy - handle cases where calculator might not be attached
        energy = None
        try:
            energy = atoms.get_potential_energy()
        except RuntimeError as e:
            if "no calculator" in str(e).lower() or "has no calculator" in str(e).lower():
                # Try to get energy from calculator results if calc exists but isn't properly set
                if atoms.calc is not None:
                    if hasattr(atoms.calc, 'results') and 'energy' in atoms.calc.results:
                        energy = atoms.calc.results['energy']
                    elif hasattr(atoms.calc, 'get_potential_energy'):
                        # Some calculators might have the method but not be properly initialized
                        try:
                            energy = atoms.calc.get_potential_energy(atoms)
                        except Exception:
                            pass
                # Try to get from atoms.info
                if energy is None and 'energy' in atoms.info:
                    energy = atoms.info['energy']
                # If we still don't have energy, log warning and return None to signal fallback needed
                if energy is None:
                    # This can happen when calculator is not properly attached in worker processes
                    log.warning(
                        f"Could not retrieve energy from atoms object. "
                        f"Calculator attached: {atoms.calc is not None}, "
                        f"Calculator results available: {hasattr(atoms.calc, 'results') if atoms.calc is not None else False}, "
                        f"Info keys: {list(atoms.info.keys())}. "
                        f"Returning None to trigger fallback to dummy values."
                    )
                    return None, None
            else:
                raise
        
        # Try to get forces - handle cases where calculator might not be attached
        forces = None
        try:
            forces = atoms.get_forces()
        except RuntimeError as e:
            if "no calculator" in str(e).lower() or "has no calculator" in str(e).lower():
                # Try to get forces from calculator results if calc exists but isn't properly set
                if atoms.calc is not None:
                    if hasattr(atoms.calc, 'results') and 'forces' in atoms.calc.results:
                        forces = atoms.calc.results['forces']
                    elif hasattr(atoms.calc, 'get_forces'):
                        # Some calculators might have the method but not be properly initialized
                        try:
                            forces = atoms.calc.get_forces(atoms)
                        except Exception:
                            pass
                # If we still don't have forces, return None to signal fallback needed
                if forces is None:
                    log.warning(
                        f"Could not retrieve forces from atoms object. "
                        f"Calculator attached: {atoms.calc is not None}, "
                        f"Calculator results available: {hasattr(atoms.calc, 'results') if atoms.calc is not None else False}, "
                        f"Info keys: {list(atoms.info.keys())}. "
                        f"Returning None to trigger fallback to dummy values."
                    )
                    return None, None
            else:
                raise
        
        return energy, forces

    @typecheck
    def get(self, idx) -> Data:
        """Get a single example."""
        dataset = self._init_dataset()

        # Get atoms from AseDBDataset - this handles the multi-file database properly
        # and should attach calculator with energy/forces results
        atoms = dataset.get_atoms(idx)

        num_atoms = len(atoms)
        atoms_to_keep = torch.ones((num_atoms,), dtype=torch.bool)

        y = torch.zeros((num_atoms, 4), dtype=torch.float32)  # Dummy energy + forces

        if self.energy_coefficients is not None:
            # Try to get energy and forces
            energy = None
            forces = None

            # Method 1: Get from calculator results if available
            # The calculator should be attached by AseDBDataset with energy/forces in results
            if atoms.calc is not None and hasattr(atoms.calc, 'results') and atoms.calc.results:
                if 'energy' in atoms.calc.results:
                    energy = atoms.calc.results['energy']
                if 'forces' in atoms.calc.results:
                    forces = atoms.calc.results['forces']

            # Method 2: Final fallback using _get_energy_and_forces helper
            # This tries atoms.get_potential_energy() and other methods
            if energy is None or forces is None:
                energy_calc, forces_calc = self._get_energy_and_forces(atoms)
                if energy is None:
                    energy = energy_calc
                if forces is None:
                    forces = forces_calc

            # Only proceed if we have both energy and forces
            if energy is not None and forces is not None:
                energy = np.array([energy], dtype=np.float32)
                energy[0] = normalize_energy(atoms, energy[0], self.energy_coefficients)

                energy = torch.from_numpy(energy)
                forces = torch.from_numpy(forces.astype(np.float32))

                energy = (energy - self.shift) / self.scale
                forces = forces / self.scale

                y = torch.cat([energy.repeat(num_atoms, 1), forces], dim=-1)
            else:
                # CRITICAL: If energy_coefficients is set, we MUST have valid energy/forces for training
                # Raising an error here prevents silently corrupting the model with dummy values
                has_calc_results = (atoms.calc is not None and
                                   hasattr(atoms.calc, 'results') and
                                   atoms.calc.results is not None)
                raise RuntimeError(
                    f"Failed to retrieve energy/forces for sample at index {idx}. "
                    f"energy_coefficients is set (training mode), but could not access data. "
                    f"Diagnostics:\n"
                    f"  - Atoms has calculator: {atoms.calc is not None}\n"
                    f"  - Calculator has results: {has_calc_results}\n"
                    f"  - Calc results keys: {list(atoms.calc.results.keys()) if has_calc_results else 'N/A'}\n"
                    f"  - Retrieved energy: {energy is not None}\n"
                    f"  - Retrieved forces: {forces is not None}\n"
                    f"This usually indicates calculator results were not preserved in DataLoader workers."
                )

        spin_graph = atoms.info.get("spin", atoms.info.get("multiplicity", 0))  # Int multiplicity
        charge_graph = atoms.info.get("charge", 0)

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
                [3], dtype=torch.long
            ),  # 3 --> Indicates non-periodic/molecule
            y=y,
            charge=torch.tensor(charge_graph, dtype=torch.float32),
            spin=torch.tensor(spin_graph, dtype=torch.long),
        )

        if self.pre_filter is not None and not self.pre_filter(data):
            return data
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        return data
