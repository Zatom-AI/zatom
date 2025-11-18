"""Adapted from https://github.com/facebookresearch/all-atom-diffusion-transformer."""

import os
from functools import partial
from typing import Sequence

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
from zatom.data.components.mp20_dataset import MP20
from zatom.data.components.omol25_dataset import OMol25
from zatom.data.components.qmof150_dataset import QMOF150
from zatom.utils import pylogger
from zatom.utils.data_utils import get_omol25_per_atom_energy_and_stats
from zatom.utils.typing_utils import typecheck

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


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
    )


@typecheck
def global_property_custom_transform(data: Data) -> Data:
    """Custom global property transformation for a dataset.

    Args:
        data: Input data object.
        removeHs: Whether to remove hydrogen atoms.

    Returns:
        Data: Transformed data object.
    """
    # PyG object attributes consistent with CrystalDataset
    data.y = torch.tensor([[torch.nan]], dtype=torch.float32)  # Dummy target property
    return data


class JointDataModule(LightningDataModule):
    """`LightningDataModule` for jointly training on 3D atomic datasets:

    - MP20: crystal structures
    - QM9: small molecules
    - QMOF150: metal-organic frameworks
    - OMol25: chemically diverse molecules

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
        # Normalize property prediction targets per data sample using training set statistics
        qm9_train_dataset = qm9_dataset[:100000]
        qm9_prop_mean = qm9_train_dataset.data.y.mean(dim=0, keepdim=True)
        qm9_prop_std = qm9_train_dataset.data.y.std(dim=0, keepdim=True)
        qm9_dataset.data.y = (qm9_dataset.data.y - qm9_prop_mean) / qm9_prop_std
        # Reference: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html
        if self.hparams.datasets.qm9.global_property == "mu":
            target = 0
            qm9_dataset.data.y = qm9_dataset.data.y[:, target].unsqueeze(-1)
            qm9_prop_mean, qm9_prop_std = (
                qm9_prop_mean[:, target].item(),
                qm9_prop_std[:, target].item(),
            )
            log.info(
                f"QM9 dipole moment (μ) target normalization: mean={qm9_prop_mean:.4f}, std={qm9_prop_std:.4f}"
            )
        elif self.hparams.datasets.qm9.global_property == "alpha":
            target = 1
            qm9_dataset.data.y = qm9_dataset.data.y[:, target].unsqueeze(-1)
            qm9_prop_mean, qm9_prop_std = (
                qm9_prop_mean[:, target].item(),
                qm9_prop_std[:, target].item(),
            )
            log.info(
                f"QM9 isotropic polarizability (α) target normalization: mean={qm9_prop_mean:.4f}, std={qm9_prop_std:.4f}"
            )
        elif self.hparams.datasets.qm9.global_property is not None:
            raise ValueError(
                f"QM9 target property '{self.hparams.datasets.qm9.global_property}' not recognized. Must be one of ('mu', 'alpha') or None."
            )
        else:
            qm9_dataset.data.y = qm9_dataset.data.y[:, 0].unsqueeze(-1)  # Default to dipole moment
        # Create train, val, test splits
        self.qm9_train_dataset = qm9_train_dataset
        self.qm9_val_dataset = qm9_dataset[100000:118000]
        self.qm9_test_dataset = qm9_dataset[118000:]
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
        mp20_dataset = MP20(
            root=self.hparams.datasets.mp20.root,
            transform=global_property_custom_transform,
        )  # .shuffle()
        # # Save num_nodes histogram for sampling from generative models
        # num_nodes = torch.tensor([data["num_nodes"] for data in mp20_dataset])
        # torch.save(
        #     torch.bincount(num_nodes),
        #     os.path.join(self.hparams.datasets.mp20.root, "num_nodes_bincount.pt"),
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
            transform=global_property_custom_transform,
        ).shuffle()
        # # Save num_nodes histogram for sampling from generative models
        # num_nodes = torch.tensor([data["num_nodes"] for data in qmof150_dataset])
        # torch.save(
        #     torch.bincount(num_nodes),
        #     os.path.join(self.hparams.datasets.qmof150.root, "num_nodes_bincount.pt"),
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
            transform=global_property_custom_transform,
            load=self.hparams.datasets.geom.proportion > 0.0,
            split="train",
        )  # .shuffle()
        self.geom_val_dataset = GEOM(
            root=self.hparams.datasets.geom.root,
            transform=global_property_custom_transform,
            load=self.hparams.datasets.geom.proportion > 0.0,
            split="val",
        )
        self.geom_test_dataset = GEOM(
            root=self.hparams.datasets.geom.root,
            transform=global_property_custom_transform,
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

        if stage is None or stage in ["fit", "validate"]:
            self.train_dataset = ConcatDataset(
                [
                    self.mp20_train_dataset,
                    self.qm9_train_dataset,
                    self.qmof150_train_dataset,
                    self.omol25_train_dataset,
                    self.geom_train_dataset,
                ]
            )
            log.info(
                f"Training dataset: {len(self.train_dataset)} samples (MP20: {len(self.mp20_train_dataset)}, QM9: {len(self.qm9_train_dataset)}, QMOF150: {len(self.qmof150_train_dataset)}, OMol25: {len(self.omol25_train_dataset)}, GEOM: {len(self.geom_train_dataset)})"
            )
            log.info(f"MP20 validation dataset: {len(self.mp20_val_dataset)} samples")
            log.info(f"QM9 validation dataset: {len(self.qm9_val_dataset)} samples")
            log.info(f"QMOF150 validation dataset: {len(self.qmof150_val_dataset)} samples")
            log.info(f"OMol25 validation dataset: {len(self.omol25_val_dataset)} samples")
            log.info(f"GEOM validation dataset: {len(self.geom_val_dataset)} samples")

        if stage is None or stage in ["test", "predict"]:
            log.info(f"MP20 test dataset: {len(self.mp20_test_dataset)} samples")
            log.info(f"QM9 test dataset: {len(self.qm9_test_dataset)} samples")
            log.info(f"QMOF150 test dataset: {len(self.qmof150_test_dataset)} samples")
            log.info(f"OMol25 test dataset: {len(self.omol25_test_dataset)} samples")
            log.info(f"GEOM test dataset: {len(self.geom_test_dataset)} samples")

    def train_dataloader(self) -> DataLoader:
        """Create and return the train dataloader.

        Returns:
            The train dataloader.
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size.train,
            num_workers=self.hparams.num_workers.train,
            pin_memory=False,
            shuffle=True,
            drop_last=True,
        )

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
                pin_memory=False,
                shuffle=False,
            ),
            DataLoader(
                dataset=self.qm9_val_dataset,
                batch_size=self.hparams.batch_size.val,
                num_workers=self.hparams.num_workers.val,
                pin_memory=False,
                shuffle=False,
            ),
            DataLoader(
                dataset=self.qmof150_val_dataset,
                batch_size=self.hparams.batch_size.val,
                num_workers=self.hparams.num_workers.val,
                pin_memory=False,
                shuffle=False,
            ),
            DataLoader(
                dataset=self.omol25_val_dataset,
                batch_size=self.hparams.batch_size.val,
                num_workers=self.hparams.num_workers.val,
                pin_memory=False,
                shuffle=False,
            ),
            DataLoader(
                dataset=self.geom_val_dataset,
                batch_size=self.hparams.batch_size.val,
                num_workers=self.hparams.num_workers.val,
                pin_memory=False,
                shuffle=False,
            ),
        ]

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
                pin_memory=False,
                shuffle=False,
            ),
            DataLoader(
                dataset=self.qm9_test_dataset,
                batch_size=self.hparams.batch_size.test,
                num_workers=self.hparams.num_workers.test,
                pin_memory=False,
                shuffle=False,
            ),
            DataLoader(
                dataset=self.qmof150_test_dataset,
                batch_size=self.hparams.batch_size.test,
                num_workers=self.hparams.num_workers.test,
                pin_memory=False,
                shuffle=False,
            ),
            DataLoader(
                dataset=self.omol25_test_dataset,
                batch_size=self.hparams.batch_size.test,
                num_workers=self.hparams.num_workers.test,
                pin_memory=False,
                shuffle=False,
            ),
            DataLoader(
                dataset=self.geom_test_dataset,
                batch_size=self.hparams.batch_size.test,
                num_workers=self.hparams.num_workers.test,
                pin_memory=False,
                shuffle=False,
            ),
        ]
