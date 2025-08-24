import copy
import os
import time
from typing import Any, Dict, Literal, Tuple

import pandas as pd
import torch
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import FSDPStrategy
from omegaconf import DictConfig
from torch.nn import ModuleDict
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
from torchmetrics import MeanMetric
from tqdm import tqdm

from zatom.eval.crystal_generation import CrystalGenerationEvaluator
from zatom.eval.mof_generation import MOFGenerationEvaluator
from zatom.eval.molecule_generation import MoleculeGenerationEvaluator
from zatom.utils import pylogger
from zatom.utils.sampling_utils import sample_top_p
from zatom.utils.training_utils import random_rotation_matrix
from zatom.utils.typing_utils import typecheck

log = pylogger.RankedLogger(__name__)


IDX_TO_DATASET = {
    0: "mp20",
    1: "qm9",
    2: "qmof150",
}
DATASET_TO_IDX = {
    "mp20": 0,  # Periodic
    "qm9": 1,  # Non-periodic
    "qmof150": 0,  # Periodic
}


class EBMLitModule(LightningModule):
    """LightningModule for generative energy-based modeling (EBM) of 3D atomic systems.

    A `LightningModule` implements 6 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def configure_model(self):
    # Define and configure model.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        ecoder: torch.nn.Module,
        interpolant: DictConfig,
        augmentations: DictConfig,
        sampling: DictConfig,
        conditioning: DictConfig,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scheduler_frequency: str,
        compile: bool,
        log_all: int | None,
    ) -> None:
        super().__init__()

        # This line allows to access init params with 'self.hparams' attribute.
        # Also ensures init params will be stored in ckpt.
        self.save_hyperparameters(logger=False)

        # Multimodal input encoder/energy decoder
        self.ecoder = ecoder

        # Interpolant for flow matching-based data corruption
        self.interpolant = interpolant

        # Evaluator objects for computing metrics
        self.val_generation_evaluators = {
            "mp20": CrystalGenerationEvaluator(
                dataset_cif_list=pd.read_csv(
                    os.path.join(self.hparams.sampling.data_dir, "mp_20", "raw", "all.csv")
                )["cif"].tolist()
            ),
            "qm9": MoleculeGenerationEvaluator(
                dataset_smiles_list=torch.load(  # nosec
                    os.path.join(self.hparams.sampling.data_dir, "qm9", "smiles.pt"),
                ),
                removeHs=self.hparams.sampling.removeHs,
            ),
            "qmof150": MOFGenerationEvaluator(),
        }
        self.test_generation_evaluators = copy.deepcopy(self.val_generation_evaluators)

        # Metric objects for calculating and averaging across batches
        self.train_metrics = ModuleDict(
            {
                "atom_types_loss": MeanMetric(),
                "atom_types_initial_loss": MeanMetric(),
                "atom_types_final_step_loss": MeanMetric(),
                "atom_types_initial_final_pred_energies_gap": MeanMetric(),
                "atom_types_ce_loss": MeanMetric(),
                "atom_types_ppl_loss": MeanMetric(),
                "pos_loss": MeanMetric(),
                "pos_initial_loss": MeanMetric(),
                "pos_final_step_loss": MeanMetric(),
                "pos_initial_final_pred_energies_gap": MeanMetric(),
                "pos_mse_loss": MeanMetric(),
                "frac_coords_loss": MeanMetric(),
                "frac_coords_initial_loss": MeanMetric(),
                "frac_coords_final_step_loss": MeanMetric(),
                "frac_coords_initial_final_pred_energies_gap": MeanMetric(),
                "frac_coords_mse_loss": MeanMetric(),
                "lengths_scaled_loss": MeanMetric(),
                "lengths_scaled_initial_loss": MeanMetric(),
                "lengths_scaled_final_step_loss": MeanMetric(),
                "lengths_scaled_initial_final_pred_energies_gap": MeanMetric(),
                "lengths_scaled_mse_loss": MeanMetric(),
                "angles_radians_loss": MeanMetric(),
                "angles_radians_initial_loss": MeanMetric(),
                "angles_radians_final_step_loss": MeanMetric(),
                "angles_radians_initial_final_pred_energies_gap": MeanMetric(),
                "angles_radians_mse_loss": MeanMetric(),
                "loss": MeanMetric(),
                "initial_loss": MeanMetric(),
                "final_step_loss": MeanMetric(),
                "initial_final_pred_energies_gap": MeanMetric(),
                "ce_loss": MeanMetric(),
                "ppl_loss": MeanMetric(),
                "mse_loss": MeanMetric(),
                "dataset_idx": MeanMetric(),
            }
        )
        self.val_metrics = ModuleDict(
            {
                "mp20": ModuleDict(
                    {
                        "atom_types_loss": MeanMetric(),
                        "atom_types_initial_loss": MeanMetric(),
                        "atom_types_final_step_loss": MeanMetric(),
                        "atom_types_initial_final_pred_energies_gap": MeanMetric(),
                        "atom_types_ce_loss": MeanMetric(),
                        "atom_types_ppl_loss": MeanMetric(),
                        "pos_loss": MeanMetric(),
                        "pos_initial_loss": MeanMetric(),
                        "pos_final_step_loss": MeanMetric(),
                        "pos_initial_final_pred_energies_gap": MeanMetric(),
                        "pos_mse_loss": MeanMetric(),
                        "frac_coords_loss": MeanMetric(),
                        "frac_coords_initial_loss": MeanMetric(),
                        "frac_coords_final_step_loss": MeanMetric(),
                        "frac_coords_initial_final_pred_energies_gap": MeanMetric(),
                        "frac_coords_mse_loss": MeanMetric(),
                        "lengths_scaled_loss": MeanMetric(),
                        "lengths_scaled_initial_loss": MeanMetric(),
                        "lengths_scaled_final_step_loss": MeanMetric(),
                        "lengths_scaled_initial_final_pred_energies_gap": MeanMetric(),
                        "lengths_scaled_mse_loss": MeanMetric(),
                        "angles_radians_loss": MeanMetric(),
                        "angles_radians_initial_loss": MeanMetric(),
                        "angles_radians_final_step_loss": MeanMetric(),
                        "angles_radians_initial_final_pred_energies_gap": MeanMetric(),
                        "angles_radians_mse_loss": MeanMetric(),
                        "loss": MeanMetric(),
                        "initial_loss": MeanMetric(),
                        "final_step_loss": MeanMetric(),
                        "initial_final_pred_energies_gap": MeanMetric(),
                        "ce_loss": MeanMetric(),
                        "ppl_loss": MeanMetric(),
                        "mse_loss": MeanMetric(),
                        "valid_rate": MeanMetric(),
                        "struct_valid_rate": MeanMetric(),
                        "comp_valid_rate": MeanMetric(),
                        "unique_rate": MeanMetric(),
                        "novel_rate": MeanMetric(),
                        "sampling_time": MeanMetric(),
                    }
                ),
                "qm9": ModuleDict(
                    {
                        "atom_types_loss": MeanMetric(),
                        "atom_types_initial_loss": MeanMetric(),
                        "atom_types_final_step_loss": MeanMetric(),
                        "atom_types_initial_final_pred_energies_gap": MeanMetric(),
                        "atom_types_ce_loss": MeanMetric(),
                        "atom_types_ppl_loss": MeanMetric(),
                        "pos_loss": MeanMetric(),
                        "pos_initial_loss": MeanMetric(),
                        "pos_final_step_loss": MeanMetric(),
                        "pos_initial_final_pred_energies_gap": MeanMetric(),
                        "pos_mse_loss": MeanMetric(),
                        "frac_coords_loss": MeanMetric(),
                        "frac_coords_initial_loss": MeanMetric(),
                        "frac_coords_final_step_loss": MeanMetric(),
                        "frac_coords_initial_final_pred_energies_gap": MeanMetric(),
                        "frac_coords_mse_loss": MeanMetric(),
                        "lengths_scaled_loss": MeanMetric(),
                        "lengths_scaled_initial_loss": MeanMetric(),
                        "lengths_scaled_final_step_loss": MeanMetric(),
                        "lengths_scaled_initial_final_pred_energies_gap": MeanMetric(),
                        "lengths_scaled_mse_loss": MeanMetric(),
                        "angles_radians_loss": MeanMetric(),
                        "angles_radians_initial_loss": MeanMetric(),
                        "angles_radians_final_step_loss": MeanMetric(),
                        "angles_radians_initial_final_pred_energies_gap": MeanMetric(),
                        "angles_radians_mse_loss": MeanMetric(),
                        "loss": MeanMetric(),
                        "initial_loss": MeanMetric(),
                        "final_step_loss": MeanMetric(),
                        "initial_final_pred_energies_gap": MeanMetric(),
                        "ce_loss": MeanMetric(),
                        "ppl_loss": MeanMetric(),
                        "mse_loss": MeanMetric(),
                        "valid_rate": MeanMetric(),
                        "unique_rate": MeanMetric(),
                        "novel_rate": MeanMetric(),
                        "mol_pred_loaded": MeanMetric(),
                        "sanitization": MeanMetric(),
                        "inchi_convertible": MeanMetric(),
                        "all_atoms_connected": MeanMetric(),
                        "bond_lengths": MeanMetric(),
                        "bond_angles": MeanMetric(),
                        "internal_steric_clash": MeanMetric(),
                        "aromatic_ring_flatness": MeanMetric(),
                        "non-aromatic_ring_non-flatness": MeanMetric(),
                        "double_bond_flatness": MeanMetric(),
                        "internal_energy": MeanMetric(),
                        "sampling_time": MeanMetric(),
                    }
                ),
                "qmof150": ModuleDict(
                    {
                        "atom_types_loss": MeanMetric(),
                        "atom_types_initial_loss": MeanMetric(),
                        "atom_types_final_step_loss": MeanMetric(),
                        "atom_types_initial_final_pred_energies_gap": MeanMetric(),
                        "atom_types_ce_loss": MeanMetric(),
                        "atom_types_ppl_loss": MeanMetric(),
                        "pos_loss": MeanMetric(),
                        "pos_initial_loss": MeanMetric(),
                        "pos_final_step_loss": MeanMetric(),
                        "pos_initial_final_pred_energies_gap": MeanMetric(),
                        "pos_mse_loss": MeanMetric(),
                        "frac_coords_loss": MeanMetric(),
                        "frac_coords_initial_loss": MeanMetric(),
                        "frac_coords_final_step_loss": MeanMetric(),
                        "frac_coords_initial_final_pred_energies_gap": MeanMetric(),
                        "frac_coords_mse_loss": MeanMetric(),
                        "lengths_scaled_loss": MeanMetric(),
                        "lengths_scaled_initial_loss": MeanMetric(),
                        "lengths_scaled_final_step_loss": MeanMetric(),
                        "lengths_scaled_initial_final_pred_energies_gap": MeanMetric(),
                        "lengths_scaled_mse_loss": MeanMetric(),
                        "angles_radians_loss": MeanMetric(),
                        "angles_radians_initial_loss": MeanMetric(),
                        "angles_radians_final_step_loss": MeanMetric(),
                        "angles_radians_initial_final_pred_energies_gap": MeanMetric(),
                        "angles_radians_mse_loss": MeanMetric(),
                        "loss": MeanMetric(),
                        "initial_loss": MeanMetric(),
                        "final_step_loss": MeanMetric(),
                        "initial_final_pred_energies_gap": MeanMetric(),
                        "ce_loss": MeanMetric(),
                        "ppl_loss": MeanMetric(),
                        "mse_loss": MeanMetric(),
                        "valid_rate": MeanMetric(),
                        "unique_rate": MeanMetric(),
                        "has_carbon": MeanMetric(),
                        "has_hydrogen": MeanMetric(),
                        "has_atomic_overlaps": MeanMetric(),
                        "has_overcoordinated_c": MeanMetric(),
                        "has_overcoordinated_n": MeanMetric(),
                        "has_overcoordinated_h": MeanMetric(),
                        "has_undercoordinated_c": MeanMetric(),
                        "has_undercoordinated_n": MeanMetric(),
                        "has_undercoordinated_rare_earth": MeanMetric(),
                        "has_metal": MeanMetric(),
                        "has_lone_molecule": MeanMetric(),
                        "has_high_charges": MeanMetric(),
                        # "is_porous": MeanMetric(),
                        "has_suspicicious_terminal_oxo": MeanMetric(),
                        "has_undercoordinated_alkali_alkaline": MeanMetric(),
                        "has_geometrically_exposed_metal": MeanMetric(),
                        # 'has_3d_connected_graph': MeanMetric(),
                        "all_checks": MeanMetric(),
                        "sampling_time": MeanMetric(),
                    }
                ),
            }
        )
        self.test_metrics = copy.deepcopy(self.val_metrics)

        # Load bincounts for sampling
        self.num_nodes_bincount = {
            "mp20": torch.nn.Parameter(
                torch.load(  # nosec
                    os.path.join(self.hparams.sampling.data_dir, "mp_20", "num_nodes_bincount.pt"),
                    map_location="cpu",
                ),
                requires_grad=False,
            ),
            "qm9": torch.nn.Parameter(
                torch.load(  # nosec
                    os.path.join(self.hparams.sampling.data_dir, "qm9", "num_nodes_bincount.pt"),
                    map_location="cpu",
                ),
                requires_grad=False,
            ),
            "qmof150": torch.nn.Parameter(
                torch.load(  # nosec
                    os.path.join(self.hparams.sampling.data_dir, "qmof", "num_nodes_bincount.pt"),
                    map_location="cpu",
                ),
                requires_grad=False,
            ),
        }
        self.spacegroups_bincount = {
            "mp20": torch.nn.Parameter(
                torch.load(  # nosec
                    os.path.join(
                        self.hparams.sampling.data_dir, "mp_20", "spacegroups_bincount.pt"
                    ),
                    map_location="cpu",
                ),
                requires_grad=False,
            ),
            "qm9": None,
            "qmof150": None,
        }

    @typecheck
    def forward(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Perform a forward pass.

        Args:
            batch: A batch of data (a tuple) containing the input tensors and target labels.

        Returns:
            A dictionary of loss values.
        """
        datasets_dict = self.trainer.datamodule.hparams.datasets

        # Subset distributions when overfitting (one-time only)
        overfitting = self.trainer.overfit_batches == 1
        if overfitting and self.trainer.global_step == 0:
            batch_num_nodes = torch.bincount(batch.batch)
            for dataset_index in batch.dataset_idx.unique():
                num_nodes = batch_num_nodes[batch.dataset_idx == dataset_index]
                spacegroups = batch.spacegroup[batch.dataset_idx == dataset_index]
                dataset_name = IDX_TO_DATASET[dataset_index.item()]

                # Filter `num_nodes_bincount`
                if self.num_nodes_bincount[dataset_name] is not None:
                    bins = torch.arange(self.num_nodes_bincount[dataset_name].size(0))
                    mask = torch.isin(bins, num_nodes.to(bins.device))  # Keep matching bins
                    self.num_nodes_bincount[dataset_name][~mask] = 0

                # Filter `spacegroups_bincount`
                if self.spacegroups_bincount[dataset_name] is not None:
                    bins = torch.arange(self.spacegroups_bincount[dataset_name].size(0))
                    mask = torch.isin(bins, spacegroups.to(bins.device))  # Keep matching bins
                    self.spacegroups_bincount[dataset_name][~mask] = 0

        # Corrupt and densify batch using the interpolant
        self.interpolant.device = self.device
        self.interpolant.max_num_nodes = max(
            len(self.num_nodes_bincount[dataset]) - 1
            for dataset in datasets_dict
            if datasets_dict[dataset].proportion > 0.0
        )
        noisy_dense_batch = self.interpolant.corrupt_batch(batch)

        # Prepare conditioning inputs to forward pass
        use_cfg = self.ecoder.class_dropout_prob > 0
        dataset_idx = batch.dataset_idx + int(
            use_cfg
        )  # 0 -> null class (for classifier-free guidance or CFG)
        # if not self.hparams.conditioning.dataset_idx:
        #     dataset_idx = torch.zeros_like(dataset_idx)
        spacegroup = batch.spacegroup
        if not self.hparams.conditioning.spacegroup:
            spacegroup = torch.zeros_like(batch.spacegroup)

        # Prepare target tensors for loss calculation
        dense_node_is_periodic, _ = to_dense_batch(
            batch.node_is_periodic, batch.batch, max_num_nodes=self.interpolant.max_num_nodes
        )
        dense_atom_types, mask = to_dense_batch(
            batch.atom_types, batch.batch, max_num_nodes=self.interpolant.max_num_nodes
        )
        dense_pos, _ = to_dense_batch(
            batch.pos, batch.batch, max_num_nodes=self.interpolant.max_num_nodes
        )
        dense_frac_coords, _ = to_dense_batch(
            batch.frac_coords, batch.batch, max_num_nodes=self.interpolant.max_num_nodes
        )
        dense_lengths_scaled = batch.lengths_scaled.unsqueeze(
            -2
        )  # Handle these as global features
        dense_angles_radians = batch.angles_radians.unsqueeze(-2)

        dense_atom_types[~mask] = -100  # Mask out padding tokens during loss calculation

        target_tensors = {
            "atom_types": dense_atom_types,
            "pos": dense_pos
            / 10.0,  # Supervise model predictions using nanometers (not Angstroms)
            "frac_coords": dense_frac_coords,
            "lengths_scaled": dense_lengths_scaled,
            "angles_radians": dense_angles_radians,
        }

        # Run energy-based encoder/decoder (i.e., E-coder or `ecoder`)
        loss_dict = self.ecoder.forward_with_loss_wrapper(
            atom_types=noisy_dense_batch["atom_types"],
            pos=noisy_dense_batch["pos"],
            frac_coords=noisy_dense_batch["frac_coords"],
            lengths_scaled=noisy_dense_batch["lengths_scaled"],
            angles_radians=noisy_dense_batch["angles_radians"],
            dataset_idx=dataset_idx,
            spacegroup=spacegroup,
            mask=noisy_dense_batch["token_mask"],
            token_is_periodic=dense_node_is_periodic,
            target_tensors=target_tensors,
            stage=self.trainer.state.stage.value,  # 'train', 'sanity_check', 'validate', 'test', 'predict'
        )

        return loss_dict

    #####################################################################################################

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # By default lightning executes validation step sanity checks before training starts,
        # so it's worth it to make sure validation metrics don't store results from these checks.
        for dataset in self.val_metrics.keys():
            for metric in self.val_metrics[dataset].values():
                metric.reset()

    def on_train_epoch_start(self) -> None:
        """Lightning hook that is called when a training epoch starts."""
        for metric in self.train_metrics.values():
            metric.reset()

    @typecheck
    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        Args:
            batch: A batch of data (a tuple) containing the input tensors and target labels.
            batch_idx: The index of the current batch.

        Returns:
            A tensor of losses between model predictions and targets.
        """
        t_start = time.time()

        with torch.no_grad():
            # Save masks used to apply augmentations
            sample_is_periodic = batch.dataset_idx != DATASET_TO_IDX["qm9"]
            batch.node_is_periodic = sample_is_periodic[batch.batch]

            if self.hparams.augmentations.frac_coords is True:
                if batch.node_is_periodic.any():
                    # Sample random translation vector from batch length distribution / 2
                    random_translation = (
                        torch.normal(
                            torch.abs(batch.lengths.mean(dim=0)),
                            torch.abs(batch.lengths.std(dim=0).nan_to_num(1e-8)),
                        )
                        / 2
                    )
                    # Apply same random translation to all Cartesian coordinates
                    pos_aug = batch.pos + random_translation
                    batch.pos = pos_aug
                    # Compute new fractional coordinates for samples which are periodic
                    cell_per_node_inv = torch.linalg.inv(
                        batch.cell[batch.batch][batch.node_is_periodic]
                    )
                    frac_coords_aug = torch.einsum(
                        "bi,bij->bj", batch.pos[batch.node_is_periodic], cell_per_node_inv
                    )
                    frac_coords_aug = frac_coords_aug % 1.0
                    batch.frac_coords[batch.node_is_periodic] = frac_coords_aug

            if self.hparams.augmentations.pos is True:
                rot_mat = random_rotation_matrix(validate=True, device=self.device)
                pos_aug = batch.pos @ rot_mat.T
                batch.pos = pos_aug
                cell_aug = batch.cell @ rot_mat.T
                batch.cell = cell_aug
                # # NOTE: Fractional coordinates are rotation-invariant
                # assert torch.allclose(
                #     batch.frac_coords,
                #     torch.einsum("bi,bij->bj", pos_aug, torch.linalg.inv(cell_aug)[batch.batch]) % 1.0,
                #     rtol=1e-3,
                #     atol=1e-3,
                # )

        # Forward pass with loss calculation
        loss_dict = self.forward(batch)

        # Log relative proportions of datasets in batch
        loss_dict["dataset_idx"] = batch.dataset_idx.detach().flatten()

        # Update and log train metrics
        for k, v in loss_dict.items():
            self.train_metrics[k](v.detach())
            self.log(
                f"train/{k}",
                self.train_metrics[k],
                on_step=True,
                on_epoch=False,
                prog_bar=True if k == "loss" else False,
            )

        # Log metadata metrics
        self.log(
            "global_step",
            torch.tensor(self.global_step, device=self.device, dtype=torch.float),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        # Log throughput metrics
        step_time = time.time() - t_start
        examples_per_second = torch.tensor(
            batch.batch_size / step_time, device=self.device, dtype=torch.float
        )
        example_length = torch.bincount(batch.batch).float().mean()

        self.log(
            "train/examples_per_second",
            examples_per_second,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        self.log(
            "train/example_length",
            example_length,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        # Return loss or backpropagation will fail
        return loss_dict["loss"]

    #####################################################################################################

    def on_validation_epoch_start(self) -> None:
        """Called at the start of the validation epoch."""
        self.on_evaluation_epoch_start(stage="val")

    def validation_step(self, batch: Batch, batch_idx: int, dataloader_idx: int) -> None:
        """Perform a single validation step on a batch of data."""
        self.evaluation_step(batch, batch_idx, dataloader_idx, stage="val")

    def on_validation_epoch_end(self) -> None:
        """Called at the end of the validation epoch."""
        self.on_evaluation_epoch_end(stage="val")

    #####################################################################################################

    def on_test_epoch_start(self) -> None:
        """Called at the start of the test epoch."""
        self.on_evaluation_epoch_start(stage="test")

    def test_step(self, batch: Batch, batch_idx: int, dataloader_idx: int) -> None:
        """Perform a single test step on a batch of data."""
        self.evaluation_step(batch, batch_idx, dataloader_idx, stage="test")

    def on_test_epoch_end(self) -> None:
        """Called at the end of the test epoch."""
        self.on_evaluation_epoch_end(stage="test")

    #####################################################################################################

    @typecheck
    def on_evaluation_epoch_start(self, stage: Literal["val", "test"]) -> None:
        """Lightning hook that is called when a validation/test epoch starts.

        Args:
            stage: The stage of evaluation ('val' or 'test').
        """
        if stage not in ["val", "test"]:
            raise ValueError("The `stage` must be `val` or `test`.")
        metrics = getattr(self, f"{stage}_metrics")
        for dataset in metrics.keys():
            for metric in metrics[dataset].values():
                metric.reset()
        generation_evaluators = getattr(self, f"{stage}_generation_evaluators")
        for dataset in generation_evaluators.keys():
            generation_evaluators[dataset].clear()  # Clear lists for next epoch

    @typecheck
    def evaluation_step(
        self,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int,
        stage: Literal["val", "test"],
    ) -> None:
        """Perform a single evaluation step on a batch of data from the validation/test set.

        Args:
            batch: The input batch of data.
            batch_idx: The index of the batch.
            dataloader_idx: The index of the dataloader.
            stage: The stage of evaluation ('val' or 'test').
        """
        if stage not in ["val", "test"]:
            raise ValueError("The `stage` must be `val` or `test`.")

        metrics = getattr(self, f"{stage}_metrics")[IDX_TO_DATASET[dataloader_idx]]
        generation_evaluator = getattr(self, f"{stage}_generation_evaluators")[
            IDX_TO_DATASET[dataloader_idx]
        ]
        generation_evaluator.device = metrics["loss"].device

        # Save masks used to apply augmentations
        sample_is_periodic = batch.dataset_idx != DATASET_TO_IDX["qm9"]
        batch.node_is_periodic = sample_is_periodic[batch.batch]

        # Forward pass with loss calculation
        loss_dict = self.forward(batch)

        # Update and log per-step eval metrics
        for k, v in loss_dict.items():
            metrics[k](v)
            self.log(
                f"{stage}_{IDX_TO_DATASET[dataloader_idx]}/{k}",
                metrics[k],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                add_dataloader_idx=False,
            )

    def on_evaluation_epoch_end(self, stage: Literal["val", "test"]) -> None:
        """Lightning hook that is called when a validation/test epoch ends."""

        if stage not in ["val", "test"]:
            raise ValueError("The `stage` must be `val` or `test`.")
        metrics = getattr(self, f"{stage}_metrics")
        generation_evaluators = getattr(self, f"{stage}_generation_evaluators")

        for dataset in metrics.keys():
            generation_evaluators[dataset].device = metrics[dataset]["loss"].device
            t_start = time.time()
            for samples_so_far in tqdm(
                range(0, self.hparams.sampling.num_samples, self.hparams.sampling.batch_size),
                desc="    Sampling",
            ):
                # Perform sampling and decoding to crystal structures
                out, batch, _ = self.sample_and_decode(
                    num_nodes_bincount=self.num_nodes_bincount[dataset],
                    spacegroups_bincount=self.spacegroups_bincount[dataset],
                    batch_size=self.hparams.sampling.batch_size,
                    dataset_idx=DATASET_TO_IDX[dataset],
                )
                # Save predictions for metrics and visualisation
                start_idx = 0
                for idx_in_batch, num_atom in enumerate(batch["num_atoms"].tolist()):
                    _atom_types = out["atom_types"].narrow(0, start_idx, num_atom)
                    _atom_types[_atom_types == 0] = 1  # Atom type 0 -> 1 (H) to prevent crash
                    _atom_types[_atom_types == self.interpolant.mask_token_index] = (
                        1  # Mask atom type -> 1 (H) to prevent crash
                    )
                    _pos = out["pos"].narrow(0, start_idx, num_atom) * 10.0  # nm to A
                    _frac_coords = out["frac_coords"].narrow(0, start_idx, num_atom)
                    _lengths = out["lengths_scaled"][idx_in_batch] * float(num_atom) ** (
                        1 / 3
                    )  # Unscale lengths
                    _angles = torch.rad2deg(
                        out["angles_radians"][idx_in_batch]
                    )  # Convert to degrees
                    generation_evaluators[dataset].append_pred_array(
                        {
                            "atom_types": _atom_types.detach().cpu().numpy(),
                            "pos": _pos.detach().cpu().numpy(),
                            "frac_coords": _frac_coords.detach().cpu().numpy(),
                            "lengths": _lengths.detach().cpu().numpy(),
                            "angles": _angles.detach().cpu().numpy(),
                            "sample_idx": samples_so_far
                            + self.global_rank * len(batch["num_atoms"])
                            + idx_in_batch,
                        }
                    )
                    start_idx = start_idx + num_atom
            t_end = time.time()

            # Compute generation metrics
            gen_metrics_dict = generation_evaluators[dataset].get_metrics(
                save=self.hparams.sampling.visualize,
                save_dir=os.path.join(
                    self.hparams.sampling.save_dir, f"{dataset}_{stage}_{self.global_rank}"
                ),
            )
            gen_metrics_dict["sampling_time"] = t_end - t_start
            for k, v in gen_metrics_dict.items():
                metrics[dataset][k](v)
                self.log(
                    f"{stage}_{dataset}/{k}",
                    metrics[dataset][k],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True if k == "valid_rate" else False,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

            if self.hparams.sampling.visualize and isinstance(self.logger, WandbLogger):
                pred_table = generation_evaluators[dataset].get_wandb_table(
                    current_epoch=self.current_epoch,
                    save_dir=os.path.join(
                        self.hparams.sampling.save_dir, f"{dataset}_{stage}_{self.global_rank}"
                    ),
                )
                self.logger.experiment.log(
                    {f"{dataset}_{stage}_samples_table_device{self.global_rank}": pred_table}
                )

    #####################################################################################################

    @typecheck
    def sample_and_decode(
        self,
        num_nodes_bincount: torch.Tensor,
        spacegroups_bincount: torch.Tensor | None,
        batch_size: int,
        dataset_idx: int = 0,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Sample and decode a batch of crystal structures.

        Args:
            num_nodes_bincount: A tensor containing the number of nodes for each crystal structure.
            spacegroups_bincount: A tensor containing the space group information for each crystal structure.
            batch_size: The number of crystal structures to sample.
            dataset_idx: The index of the dataset to sample from.

        Returns:
            A tuple containing the sampled crystal structure modalities, the original batch, and the generated sample modalities for the final MCMC step.
        """
        # Sample random lengths from distribution: (B, 1)
        sample_lengths = torch.multinomial(
            num_nodes_bincount.float(),
            batch_size,
            replacement=True,
        ).to(self.device)

        # Create dataset_idx tensor
        # NOTE 0 -> null class within EBT, while 0 -> MP20 elsewhere, so increment by 1 (for classifier-free guidance or CFG)
        use_cfg = self.ecoder.class_dropout_prob > 0
        dataset_idx = torch.full(
            (batch_size,), dataset_idx + int(use_cfg), dtype=torch.int64, device=self.device
        )

        # Create spacegroup tensor
        if not self.hparams.conditioning.spacegroup or spacegroups_bincount is None:
            # Null spacegroup
            spacegroup = torch.zeros(batch_size, dtype=torch.int64, device=self.device)
        else:
            # Sample random spacegroups from distribution: (B, 1)
            spacegroup = torch.multinomial(
                spacegroups_bincount.float(),
                batch_size,
                replacement=True,
            ).to(self.device)

        # Create token mask for visualization
        max_num_tokens = max(sample_lengths)
        token_mask = torch.zeros(
            batch_size,
            max_num_tokens,
            dtype=torch.bool,
            device=self.device,
        )
        for idx, length in enumerate(sample_lengths):
            token_mask[idx, :length] = True

        # Craft random samples using interpolant
        atom_types = torch.zeros(
            (batch_size, max_num_tokens), dtype=torch.long, device=self.device
        )
        pos = torch.zeros((batch_size, max_num_tokens, 3), dtype=torch.float32, device=self.device)
        frac_coords = torch.zeros(
            (batch_size, max_num_tokens, 3), dtype=torch.float32, device=self.device
        )
        lengths_scaled = torch.zeros((batch_size, 1, 3), dtype=torch.float32, device=self.device)
        angles_radians = torch.zeros((batch_size, 1, 3), dtype=torch.float32, device=self.device)
        token_long_mask = token_mask.long()
        token_full_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=self.device)

        self.interpolant.device = self.device
        t = self.interpolant._sample_t(batch_size)[:, None]

        noisy_dense_batch = Batch(
            atom_types=self.interpolant._corrupt_disc_x(
                atom_types, t, token_long_mask, token_long_mask
            ),
            pos=self.interpolant._corrupt_cont_x(pos, t, token_long_mask, token_long_mask),
            frac_coords=self.interpolant._corrupt_cont_x(
                frac_coords, t, token_long_mask, token_long_mask
            ),
            lengths_scaled=self.interpolant._corrupt_cont_x(
                lengths_scaled, t, token_full_mask, token_full_mask
            ),
            angles_radians=self.interpolant._corrupt_cont_x(
                angles_radians, t, token_full_mask, token_full_mask
            ),
        )

        # Use MCMC-based forward pass of EBM to denoise (generate) sample modalities
        denoised_modals_list, _ = self.ecoder.forward(
            noisy_dense_batch["atom_types"],
            noisy_dense_batch["pos"],
            noisy_dense_batch["frac_coords"],
            noisy_dense_batch["lengths_scaled"],
            noisy_dense_batch["angles_radians"],
            dataset_idx,
            spacegroup,
            token_mask,
            training=False,
            no_randomness=True,
            return_raw_discrete_logits=True,
        )

        # Sample atom types
        atom_types_logits = denoised_modals_list[-1]["atom_types"][token_mask.reshape(-1)]

        if self.hparams.sampling.atom_types_top_p is not None:
            atom_types_probs = torch.softmax(
                atom_types_logits / self.hparams.sampling.atom_types_temperature, dim=-1
            )
            atom_types = sample_top_p(
                atom_types_probs, self.hparams.sampling.atom_types_top_p
            ).squeeze(-1)
        else:
            atom_types = atom_types_logits.argmax(-1)

        # Collect final sample modalities and remove padding (to convert to PyG format)
        out = {
            "atom_types": atom_types,
            "pos": denoised_modals_list[-1]["pos"][token_mask],
            "frac_coords": denoised_modals_list[-1]["frac_coords"][token_mask],
            "lengths_scaled": denoised_modals_list[-1]["lengths_scaled"].squeeze(-2),
            "angles_radians": denoised_modals_list[-1]["angles_radians"].squeeze(-2),
        }

        batch = {
            "num_atoms": sample_lengths,
            # "batch": torch.repeat_interleave(
            #     torch.arange(len(sample_lengths), device=self.device), sample_lengths
            # ),
            # "token_idx": (torch.cumsum(token_mask, dim=-1, dtype=torch.int64) - 1)[token_mask],
        }
        return out, batch, denoised_modals_list[-1]

    #####################################################################################################

    def configure_model(self):
        """Configure the model to be used for training, validation, testing, or prediction."""
        if self.ecoder is not None:
            return

        if self.trainer is not None:
            sleep = self.trainer.global_rank
            log.info(f"Rank {self.trainer.global_rank}: Sleeping for {sleep}s to avoid CPU OOMs.")
            time.sleep(sleep)

        # In a memory-efficient way, exclude certain (unmanaged) modules from being managed by FSDP
        if isinstance(self.trainer.strategy, FSDPStrategy) and hasattr(self, "ignored_modules"):
            ignored_modules = []
            ecoder_list_grouped_modules = [
                lst
                for n in dir(self.ecoder)
                if isinstance(getattr(self.ecoder, n), list)
                and all(isinstance(x, torch.nn.Module) for x in getattr(self.ecoder, n))
                for lst in getattr(self.ecoder, n)
            ]
            ecoder_modules = list(self.ecoder.modules()) + ecoder_list_grouped_modules
            for module in ecoder_modules:
                if module.__class__ in self.ignored_modules:
                    ignored_modules.append(module)

            self.trainer.strategy.kwargs["ignored_modules"] = ignored_modules

        if self.hparams.compile:
            # Prefer `self.ecoder.compile` over `torch.compile(self.ecoder)` to avoid `_orig_` prefix checkpoint issues.
            # Reference: https://github.com/Lightning-AI/pytorch-lightning/issues/20233#issuecomment-2868169706.
            log.info(
                f"Rank {self.trainer.global_rank}: Compiling model with `torch.compile(fullgraph=True)`."
            )
            self.ecoder.compile(fullgraph=True)

        # Using WandB, log model topology once and gradients as well as parameter histogram every N steps
        if (
            isinstance(self.hparams.log_all, int)
            and self.hparams.log_all > 0
            and isinstance(self.logger, WandbLogger)
        ):
            log.info(
                f"Rank {self.trainer.global_rank}: Logging model topology once and gradients/parameters to WandB every {self.hparams.log_all} steps."
            )
            self.logger.watch(
                self.ecoder, log="all", log_freq=self.hparams.log_all, log_graph=True
            )

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        try:
            optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        except TypeError:
            # NOTE: strategies such as DeepSpeed require `params` to instead be specified as `model_params`
            optimizer = self.hparams.optimizer(model_params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_mp20/valid_rate",
                    "interval": "epoch",
                    "frequency": self.hparams.scheduler_frequency,
                },
            }
        return {"optimizer": optimizer}
