"""Adapted from https://github.com/facebookresearch/all-atom-diffusion-transformer."""

import os
from functools import partial
from typing import Sequence

import numpy as np
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from pymatgen.core import Molecule
from pymatgen.io.babel import BabelMolAdaptor
from torch.utils.data import ConcatDataset
from torch_geometric.data import Data
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from zatom.data.components.geom_dataset import GEOM
from zatom.data.components.matbench_dataset import Matbench
from zatom.data.components.mp20_dataset import MP20
from zatom.data.components.mptrj_dataset import MPtrj
from zatom.data.components.omol25_dataset import OMol25
from zatom.data.components.qmof150_dataset import QMOF150
from zatom.utils import pylogger
from zatom.utils.data_utils import (
    get_matbench_stats,
    get_mptrj_stats,
    get_omol25_per_atom_energy_and_stats,
)
from zatom.utils.typing_utils import typecheck

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

EV_TO_MEV = 1000.0  # 1 electronvolt (eV) = 1000 millielectronvolts (meV)
QM9_TARGET_NAME_TO_LITERATURE_SCALE = {
    # eVs are converted to meVs where applicable
    # Reference: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html
    "mu": 1.0,
    "alpha": 1.0,
    "homo": EV_TO_MEV,
    "lumo": EV_TO_MEV,
    "gap": EV_TO_MEV,
    "r2": 1.0,
    "zpve": EV_TO_MEV,
    "U0": EV_TO_MEV,
    "U": EV_TO_MEV,
    "H": EV_TO_MEV,
    "G": EV_TO_MEV,
    "Cv": 1.0,
    "U0_atom": EV_TO_MEV,
    "U_atom": EV_TO_MEV,
    "H_atom": EV_TO_MEV,
    "G_atom": EV_TO_MEV,
    "A": 1.0,
    "B": 1.0,
    "C": 1.0,
}
QM9_TARGETS = list(QM9_TARGET_NAME_TO_LITERATURE_SCALE.keys())
QM9_TARGET_NAME_TO_IDX = {name: i for i, name in enumerate(QM9_TARGETS)}


@typecheck
def qm9_custom_transform(data: Data, removeHs: bool = True) -> Data:
    """Custom transformation for the QM9 dataset.

    Args:
        data: Input data object.
        removeHs: Whether to remove hydrogen atoms.

    Returns:
        Data: Transformed data object.
    """
    atoms_to_keep = torch.ones_like(data.z, dtype=torch.bool)
    num_atoms = data.num_nodes

    if removeHs:
        atoms_to_keep = data.z != 1
        num_atoms = atoms_to_keep.sum().item()

    # PyG object attributes consistent with CrystalDataset
    return Data(
        id=f"qm9_{data.name}",
        atom_types=data.z[atoms_to_keep],
        pos=data.pos[atoms_to_keep],
        frac_coords=torch.zeros_like(data.pos[atoms_to_keep]),
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
        dataset_idx=torch.tensor([1], dtype=torch.long),  # 1 --> Indicates non-periodic/molecule
        y=data.y,
        charge=torch.tensor(0, dtype=torch.float32),
        spin=torch.tensor(0, dtype=torch.long),
    )


@typecheck
def global_property_custom_transform(data: Data, num_properties: int) -> Data:
    """Custom global property transformation for a dataset.

    Args:
        data: Input data object.
        num_properties: Number of global properties.

    Returns:
        Data: Transformed data object.
    """
    # PyG object attributes consistent with CrystalDataset
    data.y = torch.tensor(
        [[torch.nan] * num_properties], dtype=torch.float32
    )  # Dummy target property
    data.charge = torch.tensor(0, dtype=torch.float32)
    data.spin = torch.tensor(0, dtype=torch.long)
    return data


class JointDataModule(LightningDataModule):
    """`LightningDataModule` for jointly training on 3D atomic datasets:

    Datasets supported:
    - MP20: crystal structures
    - QM9: small molecules
    - QMOF150: metal-organic frameworks
    - OMol25: chemically diverse molecules
    - GEOM-Drugs: drug-like molecules
    - MPtrj: materials trajectories
    - Matbench: materials properties benchmark

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    @typecheck
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
    ) -> None:
        super().__init__()

        # This line allows to access init params with 'self.hparams' attribute
        # Also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    @typecheck
    def setup(self, stage: str | None = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        Args:
            stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # QM9 dataset
        qm9_dataset = QM9(
            root=self.hparams.datasets.qm9.root,
            transform=partial(qm9_custom_transform, removeHs=self.hparams.datasets.qm9.removeHs),
        ).shuffle()
        # Create generative modeling train, val, test splits (n.b., same as ADiT)
        self.qm9_train_dataset = qm9_dataset[:100000]
        self.qm9_val_dataset = qm9_dataset[100000:118000]
        self.qm9_test_dataset = qm9_dataset[118000:]
        # # Save `num_nodes` histogram and SMILES strings for sampling from generative models
        # num_nodes = torch.tensor([data["num_nodes"] for data in qm9_dataset])
        # smiles = (
        #     [
        #         BabelMolAdaptor(Molecule(species=data["atom_types"], coords=data["pos"])).pybel_mol.write("smi").strip()
        #         for data in qm9_dataset
        #     ]
        # )
        # torch.save(
        #     torch.bincount(num_nodes),
        #     os.path.join(self.hparams.datasets.qm9.root, "num_nodes_bincount.pt"),
        # )
        # torch.save(smiles, os.path.join(self.hparams.datasets.qm9.root, "smiles.pt"))
        # Select target properties if specified
        qm9_target_names = self.hparams.datasets.qm9.global_property
        if qm9_target_names is not None:
            assert all(
                name in QM9_TARGET_NAME_TO_IDX for name in qm9_target_names
            ), f"QM9 target properties '{qm9_target_names}' not recognized. Must be a list of the properties listed in https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html."
            qm9_dataset = QM9(
                root=self.hparams.datasets.qm9.root,
                transform=partial(
                    qm9_custom_transform, removeHs=self.hparams.datasets.qm9.removeHs
                ),
            )
            qm9_target_idx = torch.tensor(
                [QM9_TARGET_NAME_TO_IDX[name] for name in qm9_target_names]
            )
            qm9_dataset.data.y = torch.where(
                torch.isin(
                    torch.arange(
                        qm9_dataset.data.y.size(1), device=qm9_dataset.data.y.device
                    ).unsqueeze(0),
                    qm9_target_idx,
                ),
                qm9_dataset.data.y,
                float("nan"),
            )
            log.info(
                f"QM9 target properties set to '{qm9_target_names}' (indices {qm9_target_idx.tolist()})"
                f" with mean {qm9_dataset.data.y[:, qm9_target_idx].mean(dim=0).tolist()} and std {qm9_dataset.data.y[:, qm9_target_idx].std(dim=0).tolist()}."
            )
            # Create property prediction train/val/test splits (n.b., same as Platonic Transformer)
            qm9_random_state = np.random.RandomState(seed=42)
            qm9_perm = torch.from_numpy(qm9_random_state.permutation(np.arange(130831)))
            qm9_train_idx, qm9_val_idx, qm9_test_idx = (
                qm9_perm[:110000],
                qm9_perm[110000:120000],
                qm9_perm[120000:],
            )
            # Normalize property prediction targets per data sample using training set statistics
            self.qm9_train_prop_mean = qm9_dataset.data.y[qm9_train_idx].mean(dim=0, keepdim=True)
            self.qm9_train_prop_std = qm9_dataset.data.y[qm9_train_idx].std(dim=0, keepdim=True)
            qm9_dataset.data.y = (
                qm9_dataset.data.y - self.qm9_train_prop_mean
            ) / self.qm9_train_prop_std
            self.qm9_train_dataset = qm9_dataset[qm9_train_idx]
            self.qm9_val_dataset = qm9_dataset[qm9_val_idx]
            self.qm9_test_dataset = qm9_dataset[qm9_test_idx]
        # Retain subset of dataset; can be used to train on only one dataset, too
        qm9_train_subset_size = int(
            len(self.qm9_train_dataset) * self.hparams.datasets.qm9.proportion
        )
        self.qm9_train_dataset = self.qm9_train_dataset[:qm9_train_subset_size]
        self.qm9_val_dataset = self.qm9_val_dataset[
            : max(
                qm9_train_subset_size,
                int(len(self.qm9_val_dataset) * self.hparams.datasets.qm9.proportion),
            )
        ]
        self.qm9_test_dataset = self.qm9_test_dataset[
            : max(
                qm9_train_subset_size,
                int(len(self.qm9_test_dataset) * self.hparams.datasets.qm9.proportion),
            )
        ]

        # MP20 dataset
        global_property_custom_transform_fn = partial(
            global_property_custom_transform, num_properties=qm9_dataset.data.y.shape[1]
        )  # Dummy property
        mp20_dataset = MP20(
            root=self.hparams.datasets.mp20.root,
            transform=global_property_custom_transform_fn,
        )  # .shuffle()
        # # Save num_nodes and spacegroup histograms for sampling from generative models
        # num_nodes = torch.tensor([data["num_nodes"] for data in mp20_dataset])
        # spacegroups = torch.tensor([data["spacegroup"] for data in mp20_dataset])
        # torch.save(
        #     torch.bincount(num_nodes),
        #     os.path.join(self.hparams.datasets.mp20.root, "num_nodes_bincount.pt"),
        # )
        # torch.save(
        #     torch.bincount(spacegroups),
        #     os.path.join(self.hparams.datasets.mp20.root, "spacegroups_bincount.pt"),
        # )
        # Create train, val, test splits
        self.mp20_train_dataset = mp20_dataset[:27138]
        self.mp20_val_dataset = mp20_dataset[27138 : 27138 + 9046]
        self.mp20_test_dataset = mp20_dataset[27138 + 9046 :]
        # Retain subset of dataset; can be used to train on only one dataset, too
        mp20_train_subset_size = int(
            len(self.mp20_train_dataset) * self.hparams.datasets.mp20.proportion
        )
        self.mp20_train_dataset = self.mp20_train_dataset[:mp20_train_subset_size]
        self.mp20_val_dataset = self.mp20_val_dataset[
            : max(
                mp20_train_subset_size,
                int(len(self.mp20_val_dataset) * self.hparams.datasets.mp20.proportion),
            )
        ]
        self.mp20_test_dataset = self.mp20_test_dataset[
            : max(
                mp20_train_subset_size,
                int(len(self.mp20_test_dataset) * self.hparams.datasets.mp20.proportion),
            )
        ]

        # QMOF150 dataset
        qmof150_dataset = QMOF150(
            root=self.hparams.datasets.qmof150.root,
            transform=global_property_custom_transform_fn,
        ).shuffle()
        # Save num_nodes and spacegroup histogram for sampling from generative models
        # num_nodes = torch.tensor([data["num_nodes"] for data in qmof150_dataset])
        # spacegroups = torch.tensor([data["spacegroup"] for data in qmof150_dataset])
        # torch.save(
        #     torch.bincount(num_nodes),
        #     os.path.join(self.hparams.datasets.qmof150.root, "num_nodes_bincount.pt"),
        # )
        # torch.save(
        #     torch.bincount(spacegroups),
        #     os.path.join(self.hparams.datasets.qmof150.root, "spacegroups_bincount.pt"),
        # )
        # Create train, val, test splits
        self.qmof150_train_dataset = qmof150_dataset[2048:]
        self.qmof150_val_dataset = qmof150_dataset[:1024]
        self.qmof150_test_dataset = qmof150_dataset[1024:2048]
        # Retain subset of dataset; can be used to train on only one dataset, too
        qmof150_train_subset_size = int(
            len(self.qmof150_train_dataset) * self.hparams.datasets.qmof150.proportion
        )
        self.qmof150_train_dataset = self.qmof150_train_dataset[:qmof150_train_subset_size]
        self.qmof150_val_dataset = self.qmof150_val_dataset[
            : max(
                qmof150_train_subset_size,
                int(len(self.qmof150_val_dataset) * self.hparams.datasets.qmof150.proportion),
            )
        ]
        self.qmof150_test_dataset = self.qmof150_test_dataset[
            : max(
                qmof150_train_subset_size,
                int(len(self.qmof150_test_dataset) * self.hparams.datasets.qmof150.proportion),
            )
        ]

        # OMol25 dataset
        # Create train, val, test splits
        self.omol25_train_dataset = OMol25(
            root=self.hparams.datasets.omol25.root,
            split="train",
            subset=self.hparams.datasets.omol25.subset,
        )  # .shuffle()
        self.omol25_val_dataset = OMol25(root=self.hparams.datasets.omol25.root, split="val")
        self.omol25_test_dataset = OMol25(root=self.hparams.datasets.omol25.root, split="test")
        # # Save `num_nodes` histogram and SMILES strings for sampling from generative models
        # num_nodes = torch.tensor(
        #     [
        #         data["num_nodes"]
        #         for dataset in [
        #             self.omol25_train_dataset,
        #             self.omol25_val_dataset,
        #             self.omol25_test_dataset,
        #         ]
        #         for data in dataset
        #     ]
        # )
        # smiles = (
        #     [
        #         BabelMolAdaptor(Molecule(species=data["atom_types"], coords=data["pos"])).pybel_mol.write("smi").strip()
        #         for dataset in [
        #             self.omol25_train_dataset,
        #             self.omol25_val_dataset,
        #             self.omol25_test_dataset,
        #         ]
        #         for data in tqdm(dataset, desc="Saving OMol25 SMILES strings")
        #     ]
        # )
        # torch.save(
        #     torch.bincount(num_nodes),
        #     os.path.join(self.hparams.datasets.omol25.root, "num_nodes_bincount.pt"),
        # )
        # torch.save(smiles, os.path.join(self.hparams.datasets.omol25.root, "smiles.pt"))
        # Normalize energy and force prediction targets per data sample using training set statistics
        if self.hparams.datasets.omol25.global_energy:
            omol25_train_energy_coefficients, omol25_train_dataset_stats = (
                get_omol25_per_atom_energy_and_stats(
                    dataset=self.omol25_train_dataset,
                    coef_path=self.hparams.datasets.omol25.root,
                    include_hof=False,
                    recalculate=False,
                )
            )
            self.omol25_train_dataset.energy_coefficients = omol25_train_energy_coefficients
            self.omol25_val_dataset.energy_coefficients = omol25_train_energy_coefficients
            self.omol25_test_dataset.energy_coefficients = omol25_train_energy_coefficients
            self.omol25_train_dataset.shift = omol25_train_dataset_stats["shift"]
            self.omol25_train_dataset.scale = omol25_train_dataset_stats["scale"]
            self.omol25_val_dataset.shift = omol25_train_dataset_stats["shift"]
            self.omol25_val_dataset.scale = omol25_train_dataset_stats["scale"]
            self.omol25_test_dataset.shift = omol25_train_dataset_stats["shift"]
            self.omol25_test_dataset.scale = omol25_train_dataset_stats["scale"]
        # Retain subset of dataset; can be used to train on only one dataset, too
        omol25_train_subset_size = int(
            len(self.omol25_train_dataset) * self.hparams.datasets.omol25.proportion
        )
        self.omol25_train_dataset = self.omol25_train_dataset[:omol25_train_subset_size]
        self.omol25_val_dataset = self.omol25_val_dataset[
            : max(
                omol25_train_subset_size,
                int(len(self.omol25_val_dataset) * self.hparams.datasets.omol25.proportion),
            )
        ]
        self.omol25_test_dataset = self.omol25_test_dataset[
            : max(
                omol25_train_subset_size,
                int(len(self.omol25_test_dataset) * self.hparams.datasets.omol25.proportion),
            )
        ]

        # GEOM-Drugs dataset
        # Create train, val, test splits
        self.geom_train_dataset = GEOM(
            root=self.hparams.datasets.geom.root,
            transform=global_property_custom_transform_fn,
            load=self.hparams.datasets.geom.proportion > 0.0,
            split="train",
        )  # .shuffle()
        self.geom_val_dataset = GEOM(
            root=self.hparams.datasets.geom.root,
            transform=global_property_custom_transform_fn,
            load=self.hparams.datasets.geom.proportion > 0.0,
            split="val",
        )
        self.geom_test_dataset = GEOM(
            root=self.hparams.datasets.geom.root,
            transform=global_property_custom_transform_fn,
            load=self.hparams.datasets.geom.proportion > 0.0,
            split="test",
        )
        # # Save `num_nodes` histogram and SMILES strings for sampling from generative models
        # num_nodes = torch.cat(
        #     [
        #         torch.load(self.geom_train_dataset.processed_num_nodes_file), # nosec
        #         torch.load(self.geom_val_dataset.processed_num_nodes_file), # nosec
        #         torch.load(self.geom_test_dataset.processed_num_nodes_file), # nosec
        #     ]
        # )
        # smiles = (
        #     [
        #         *torch.load(self.geom_train_dataset.processed_smiles_file), # nosec
        #         *torch.load(self.geom_val_dataset.processed_smiles_file), # nosec
        #         *torch.load(self.geom_test_dataset.processed_smiles_file), # nosec
        #     ]
        # )
        # torch.save(
        #     torch.bincount(num_nodes),
        #     os.path.join(self.hparams.datasets.geom.root, "num_nodes_bincount.pt"),
        # )
        # torch.save(smiles, os.path.join(self.hparams.datasets.geom.root, "smiles.pt"))
        # Retain subset of dataset; can be used to train on only one dataset, too
        geom_train_subset_size = int(
            len(self.geom_train_dataset) * self.hparams.datasets.geom.proportion
        )
        self.geom_train_dataset = self.geom_train_dataset[:geom_train_subset_size]
        self.geom_val_dataset = self.geom_val_dataset[
            : max(
                geom_train_subset_size,
                int(len(self.geom_val_dataset) * self.hparams.datasets.geom.proportion),
            )
        ]
        self.geom_test_dataset = self.geom_test_dataset[
            : max(
                geom_train_subset_size,
                int(len(self.geom_test_dataset) * self.hparams.datasets.geom.proportion),
            )
        ]

        # MPtrj dataset
        # Create train, val, test splits
        self.mptrj_train_dataset = MPtrj(
            root=self.hparams.datasets.mptrj.root,
            load=self.hparams.datasets.mptrj.proportion > 0.0,
            split="train",
        )  # .shuffle()
        self.mptrj_val_dataset = MPtrj(
            root=self.hparams.datasets.mptrj.root,
            load=self.hparams.datasets.mptrj.proportion > 0.0,
            split="val",
        )
        self.mptrj_test_dataset = MPtrj(
            root=self.hparams.datasets.mptrj.root,
            load=self.hparams.datasets.mptrj.proportion > 0.0,
            split="test",
        )
        # # Save num_nodes histogram for sampling from generative models
        # num_nodes = torch.tensor(
        #     [
        #         data["num_nodes"]
        #         for dataset in [
        #             self.mptrj_train_dataset,
        #             self.mptrj_val_dataset,
        #             self.mptrj_test_dataset,
        #         ]
        #         for data in dataset
        #     ]
        # )
        # spacegroups = torch.tensor(
        #     [
        #         data["spacegroup"]
        #         for dataset in [
        #             self.mptrj_train_dataset,
        #             self.mptrj_val_dataset,
        #             self.mptrj_test_dataset,
        #         ]
        #         for data in dataset
        #     ]
        # )
        # torch.save(
        #     torch.bincount(num_nodes),
        #     os.path.join(self.hparams.datasets.mptrj.root, "num_nodes_bincount.pt"),
        # )
        # torch.save(
        #     torch.bincount(spacegroups),
        #     os.path.join(self.hparams.datasets.mptrj.root, "spacegroups_bincount.pt"),
        # )
        # Normalize energy and force prediction targets per data sample using training set statistics
        if self.hparams.datasets.mptrj.global_energy:
            mptrj_train_dataset_stats = get_mptrj_stats(
                dataset=self.mptrj_train_dataset,
                coef_path=self.hparams.datasets.mptrj.root,
                recalculate=False,
            )
            self.mptrj_train_dataset.shift = mptrj_train_dataset_stats["shift"]
            self.mptrj_train_dataset.scale = mptrj_train_dataset_stats["scale"]
            self.mptrj_val_dataset.shift = mptrj_train_dataset_stats["shift"]
            self.mptrj_val_dataset.scale = mptrj_train_dataset_stats["scale"]
            self.mptrj_test_dataset.shift = mptrj_train_dataset_stats["shift"]
            self.mptrj_test_dataset.scale = mptrj_train_dataset_stats["scale"]
        # Retain subset of dataset; can be used to train on only one dataset, too
        mptrj_train_subset_size = int(
            len(self.mptrj_train_dataset) * self.hparams.datasets.mptrj.proportion
        )
        self.mptrj_train_dataset = self.mptrj_train_dataset[:mptrj_train_subset_size]
        self.mptrj_val_dataset = self.mptrj_val_dataset[
            : max(
                mptrj_train_subset_size,
                int(len(self.mptrj_val_dataset) * self.hparams.datasets.mptrj.proportion),
            )
        ]
        self.mptrj_test_dataset = self.mptrj_test_dataset[
            : max(
                mptrj_train_subset_size,
                int(len(self.mptrj_test_dataset) * self.hparams.datasets.mptrj.proportion),
            )
        ]

        # Matbench dataset
        # Create train, val, test splits
        self.matbench_train_dataset = Matbench(
            root=self.hparams.datasets.matbench.root,
            load=self.hparams.datasets.matbench.proportion > 0.0,
            task_name=self.hparams.datasets.matbench.global_property,
            split="train",
        )  # .shuffle()
        self.matbench_val_dataset = Matbench(
            root=self.hparams.datasets.matbench.root,
            load=self.hparams.datasets.matbench.proportion > 0.0,
            task_name=self.hparams.datasets.matbench.global_property,
            split="train",  # NOTE: Matbench does not have a val split, so use train split instead
        )
        self.matbench_test_dataset = Matbench(
            root=self.hparams.datasets.matbench.root,
            load=self.hparams.datasets.matbench.proportion > 0.0,
            task_name=self.hparams.datasets.matbench.global_property,
            split="test",
        )
        # # Save num_nodes histogram for sampling from generative models
        # num_nodes = torch.tensor(
        #     [
        #         data["num_nodes"]
        #         for dataset in [
        #             self.matbench_train_dataset,
        #             self.matbench_val_dataset,
        #             self.matbench_test_dataset,
        #         ]
        #         for data in dataset
        #     ]
        # )
        # spacegroups = torch.tensor(
        #     [
        #         data["spacegroup"]
        #         for dataset in [
        #             self.matbench_train_dataset,
        #             self.matbench_val_dataset,
        #             self.matbench_test_dataset,
        #         ]
        #         for data in dataset
        #     ]
        # )
        # torch.save(
        #     torch.bincount(num_nodes),
        #     os.path.join(self.hparams.datasets.matbench.root, "num_nodes_bincount.pt"),
        # )
        # torch.save(
        #     torch.bincount(spacegroups),
        #     os.path.join(self.hparams.datasets.matbench.root, "spacegroups_bincount.pt"),
        # )
        # Normalize property prediction targets per data sample using training set statistics
        matbench_train_dataset_stats = get_matbench_stats(
            dataset=self.matbench_train_dataset,
            task_name=self.hparams.datasets.matbench.global_property,
            coef_path=self.hparams.datasets.matbench.root,
            recalculate=False,
        )
        self.matbench_train_dataset.shift = matbench_train_dataset_stats["shift"]
        self.matbench_train_dataset.scale = matbench_train_dataset_stats["scale"]
        self.matbench_val_dataset.shift = matbench_train_dataset_stats["shift"]
        self.matbench_val_dataset.scale = matbench_train_dataset_stats["scale"]
        self.matbench_test_dataset.shift = matbench_train_dataset_stats["shift"]
        self.matbench_test_dataset.scale = matbench_train_dataset_stats["scale"]
        # Retain subset of dataset; can be used to train on only one dataset, too
        matbench_train_subset_size = int(
            len(self.matbench_train_dataset) * self.hparams.datasets.matbench.proportion
        )
        self.matbench_train_dataset = self.matbench_train_dataset[:matbench_train_subset_size]
        self.matbench_val_dataset = self.matbench_val_dataset[
            : max(
                matbench_train_subset_size,
                # NOTE: Using smaller proportion for val (i.e., duplicate train) set to mitigate overfitting
                int(
                    len(self.matbench_val_dataset)
                    * self.hparams.datasets.matbench.proportion
                    * 0.1
                ),
            )
        ]
        self.matbench_test_dataset = self.matbench_test_dataset[
            : max(
                matbench_train_subset_size,
                int(len(self.matbench_test_dataset) * self.hparams.datasets.matbench.proportion),
            )
        ]

        if stage is None or stage in ["fit", "validate"]:
            self.train_dataset = ConcatDataset(
                [
                    self.mp20_train_dataset,
                    self.qm9_train_dataset,
                    self.qmof150_train_dataset,
                    self.omol25_train_dataset,
                    self.geom_train_dataset,
                    self.mptrj_train_dataset,
                    self.matbench_train_dataset,
                ]
            )
            log.info(
                f"Training dataset: {len(self.train_dataset)} samples (MP20: {len(self.mp20_train_dataset)}, QM9: {len(self.qm9_train_dataset)}, QMOF150: {len(self.qmof150_train_dataset)}, OMol25: {len(self.omol25_train_dataset)}, GEOM: {len(self.geom_train_dataset)}, MPtrj: {len(self.mptrj_train_dataset)}, Matbench: {len(self.matbench_train_dataset)})"
            )
            log.info(f"MP20 validation dataset: {len(self.mp20_val_dataset)} samples")
            log.info(f"QM9 validation dataset: {len(self.qm9_val_dataset)} samples")
            log.info(f"QMOF150 validation dataset: {len(self.qmof150_val_dataset)} samples")
            log.info(f"OMol25 validation dataset: {len(self.omol25_val_dataset)} samples")
            log.info(f"GEOM validation dataset: {len(self.geom_val_dataset)} samples")
            log.info(f"MPtrj validation dataset: {len(self.mptrj_val_dataset)} samples")
            log.info(f"Matbench validation dataset: {len(self.matbench_val_dataset)} samples")

        if stage is None or stage in ["test", "predict"]:
            log.info(f"MP20 test dataset: {len(self.mp20_test_dataset)} samples")
            log.info(f"QM9 test dataset: {len(self.qm9_test_dataset)} samples")
            log.info(f"QMOF150 test dataset: {len(self.qmof150_test_dataset)} samples")
            log.info(f"OMol25 test dataset: {len(self.omol25_test_dataset)} samples")
            log.info(f"GEOM test dataset: {len(self.geom_test_dataset)} samples")
            log.info(f"MPtrj test dataset: {len(self.mptrj_test_dataset)} samples")
            log.info(f"Matbench test dataset: {len(self.matbench_test_dataset)} samples")

    @typecheck
    def train_dataloader(self) -> DataLoader:
        """Create and return the train dataloader.

        Returns:
            The train dataloader.
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size.train,
            num_workers=self.hparams.num_workers.train,
            persistent_workers=self.hparams.num_workers.persistent_workers,
            pin_memory=self.hparams.num_workers.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    @typecheck
    def val_dataloader(self) -> Sequence[DataLoader]:
        """Create and return the validation dataloader.

        Returns:
            A sequence of validation dataloaders.
        """
        return [
            DataLoader(
                dataset=self.mp20_val_dataset,
                batch_size=self.hparams.batch_size.val,
                num_workers=self.hparams.num_workers.val,
                persistent_workers=self.hparams.num_workers.persistent_workers,
                pin_memory=self.hparams.num_workers.pin_memory,
                shuffle=False,
            ),
            DataLoader(
                dataset=self.qm9_val_dataset,
                batch_size=self.hparams.batch_size.val,
                num_workers=self.hparams.num_workers.val,
                persistent_workers=self.hparams.num_workers.persistent_workers,
                pin_memory=self.hparams.num_workers.pin_memory,
                shuffle=False,
            ),
            DataLoader(
                dataset=self.qmof150_val_dataset,
                batch_size=self.hparams.batch_size.val,
                num_workers=self.hparams.num_workers.val,
                persistent_workers=self.hparams.num_workers.persistent_workers,
                pin_memory=self.hparams.num_workers.pin_memory,
                shuffle=False,
            ),
            DataLoader(
                dataset=self.omol25_val_dataset,
                batch_size=self.hparams.batch_size.val,
                num_workers=self.hparams.num_workers.val,
                persistent_workers=self.hparams.num_workers.persistent_workers,
                pin_memory=self.hparams.num_workers.pin_memory,
                shuffle=False,
            ),
            DataLoader(
                dataset=self.geom_val_dataset,
                batch_size=self.hparams.batch_size.val,
                num_workers=self.hparams.num_workers.val,
                persistent_workers=self.hparams.num_workers.persistent_workers,
                pin_memory=self.hparams.num_workers.pin_memory,
                shuffle=False,
            ),
            DataLoader(
                dataset=self.mptrj_val_dataset,
                batch_size=self.hparams.batch_size.val,
                num_workers=self.hparams.num_workers.val,
                persistent_workers=self.hparams.num_workers.persistent_workers,
                pin_memory=self.hparams.num_workers.pin_memory,
                shuffle=False,
            ),
            DataLoader(
                dataset=self.matbench_val_dataset,
                batch_size=self.hparams.batch_size.val,
                num_workers=self.hparams.num_workers.val,
                persistent_workers=self.hparams.num_workers.persistent_workers,
                pin_memory=self.hparams.num_workers.pin_memory,
                shuffle=False,
            ),
        ]

    @typecheck
    def test_dataloader(self) -> Sequence[DataLoader]:
        """Create and return the test dataloader.

        Returns:
            A sequence of test dataloaders.
        """
        return [
            DataLoader(
                dataset=self.mp20_test_dataset,
                batch_size=self.hparams.batch_size.test,
                num_workers=self.hparams.num_workers.test,
                persistent_workers=self.hparams.num_workers.persistent_workers,
                pin_memory=self.hparams.num_workers.pin_memory,
                shuffle=False,
            ),
            DataLoader(
                dataset=self.qm9_test_dataset,
                batch_size=self.hparams.batch_size.test,
                num_workers=self.hparams.num_workers.test,
                persistent_workers=self.hparams.num_workers.persistent_workers,
                pin_memory=self.hparams.num_workers.pin_memory,
                shuffle=False,
            ),
            DataLoader(
                dataset=self.qmof150_test_dataset,
                batch_size=self.hparams.batch_size.test,
                num_workers=self.hparams.num_workers.test,
                persistent_workers=self.hparams.num_workers.persistent_workers,
                pin_memory=self.hparams.num_workers.pin_memory,
                shuffle=False,
            ),
            DataLoader(
                dataset=self.omol25_test_dataset,
                batch_size=self.hparams.batch_size.test,
                num_workers=self.hparams.num_workers.test,
                persistent_workers=self.hparams.num_workers.persistent_workers,
                pin_memory=self.hparams.num_workers.pin_memory,
                shuffle=False,
            ),
            DataLoader(
                dataset=self.geom_test_dataset,
                batch_size=self.hparams.batch_size.test,
                num_workers=self.hparams.num_workers.test,
                persistent_workers=self.hparams.num_workers.persistent_workers,
                pin_memory=self.hparams.num_workers.pin_memory,
                shuffle=False,
            ),
            DataLoader(
                dataset=self.mptrj_test_dataset,
                batch_size=self.hparams.batch_size.test,
                num_workers=self.hparams.num_workers.test,
                persistent_workers=self.hparams.num_workers.persistent_workers,
                pin_memory=self.hparams.num_workers.pin_memory,
                shuffle=False,
            ),
            DataLoader(
                dataset=self.matbench_test_dataset,
                batch_size=self.hparams.batch_size.test,
                num_workers=self.hparams.num_workers.test,
                persistent_workers=self.hparams.num_workers.persistent_workers,
                pin_memory=self.hparams.num_workers.pin_memory,
                shuffle=False,
            ),
        ]
