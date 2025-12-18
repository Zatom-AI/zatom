import copy
import os
import time
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import FSDPStrategy
from omegaconf import DictConfig
from tensordict import TensorDict
from torch.nn import ModuleDict
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
from torchmetrics import MeanMetric
from tqdm import tqdm

from zatom.data.components.matbench_dataset import (
    MATBENCH_QM9_TARGET_NAME_TO_LITERATURE_SCALE,
)
from zatom.data.components.preprocessing_utils import lattice_params_to_matrix_torch
from zatom.data.joint_datamodule import (
    EV_TO_MEV,
    QM9_TARGET_NAME_TO_IDX,
    QM9_TARGET_NAME_TO_LITERATURE_SCALE,
    QM9_TARGETS,
)
from zatom.eval.crystal_generation import CrystalGenerationEvaluator
from zatom.eval.mof_generation import MOFGenerationEvaluator
from zatom.eval.molecule_generation import MoleculeGenerationEvaluator
from zatom.utils import pylogger
from zatom.utils.training_utils import sample_uniform_rotation, scatter_mean_torch
from zatom.utils.typing_utils import typecheck

log = pylogger.RankedLogger(__name__)


TASK_NAMES = Literal["train_fm", "finetune_fm", "eval_fm", "overfit_fm", "debug_fm"]


class Zatom(LightningModule):
    """LightningModule for foundation modeling of 3D molecules and materials.

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

    @typecheck
    def __init__(
        self,
        architecture: torch.nn.Module,
        augmentations: DictConfig,
        sampling: DictConfig,
        conditioning: DictConfig,
        datasets: DictConfig,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scheduler_frequency: int,
        compile: bool,
        log_grads_every_n_steps: int | None,
        task_name: TASK_NAMES,
    ) -> None:
        super().__init__()

        # This line allows to access init params with 'self.hparams' attribute.
        # Also ensures init params will be stored in ckpt.
        self.save_hyperparameters(logger=False)

        # Model architecture
        self.model = architecture

        # Build dataset mappings dynamically from config
        self.index_to_dataset = {}
        self.dataset_to_index = {}
        self.dataset_to_idx = {}  # Name -> Type (0=Periodic, 1=Non-Periodic)
        self.index_to_idx = {}  # ID -> Type
        self.periodic_dataset_ids = []
        self.non_periodic_dataset_ids = []

        # We need to iterate over the datasets config to build these.
        # self.hparams.datasets is a DictConfig where keys are dataset names.
        for dataset, cfg in self.hparams.datasets.items():
            if "id" in cfg and "type_label" in cfg:
                ds_id = cfg.id
                ds_type = cfg.type_label
                self.index_to_dataset[ds_id] = dataset
                self.dataset_to_index[dataset] = ds_id
                self.dataset_to_idx[dataset] = ds_type
                self.index_to_idx[ds_id] = ds_type

                if ds_type == 0:  # Periodic
                    self.periodic_dataset_ids.append(ds_id)
                else:
                    self.non_periodic_dataset_ids.append(ds_id)
            else:
                # Fallback or skip if metadata missing (should ideally error or warn)
                if cfg.get("proportion", 0.0) > 0.0:
                    log.warning(
                        f"Dataset {dataset} is used but missing 'id' or 'type_label' in config."
                    )

        # Evaluator objects for computing metrics
        self.val_generation_evaluators = {}
        for dataset, cfg in self.hparams.datasets.items():
            if not (cfg.proportion > 0.0):
                continue

            # Keep hardcoded instantiation for now as it's complex logic per dataset type
            if dataset == "mp20":
                self.val_generation_evaluators[dataset] = CrystalGenerationEvaluator(
                    dataset_cif_list=pd.read_csv(
                        os.path.join(
                            self.hparams.sampling.data_dir,
                            "mp_20",
                            "raw",
                            "all.csv",
                        )
                    )["cif"].tolist()
                )
            elif dataset == "qm9":
                self.val_generation_evaluators[dataset] = MoleculeGenerationEvaluator(
                    dataset_smiles_list=torch.load(  # nosec
                        os.path.join(self.hparams.sampling.data_dir, "qm9", "smiles.pt"),
                    ),
                    removeHs=self.hparams.sampling.removeHs,
                )
            elif dataset == "qmof150":
                self.val_generation_evaluators[dataset] = MOFGenerationEvaluator()
            elif dataset == "omol25":
                self.val_generation_evaluators[dataset] = MoleculeGenerationEvaluator(
                    dataset_smiles_list=torch.load(  # nosec
                        os.path.join(self.hparams.sampling.data_dir, "omol25", "smiles.pt"),
                    ),
                    removeHs=self.hparams.sampling.removeHs,
                )
            elif dataset == "geom":
                self.val_generation_evaluators[dataset] = MoleculeGenerationEvaluator(
                    dataset_smiles_list=torch.load(  # nosec
                        os.path.join(self.hparams.sampling.data_dir, "geom", "smiles.pt"),
                    ),
                    removeHs=self.hparams.sampling.removeHs,
                )
            elif dataset in ("mptrj", "matbench"):
                self.val_generation_evaluators[dataset] = CrystalGenerationEvaluator(
                    dataset_cif_list=None
                )
            else:
                log.warning(
                    f"No specific evaluator configured for dataset {dataset}. Skipping evaluator."
                )

        self.test_generation_evaluators = copy.deepcopy(self.val_generation_evaluators)

        # Metric objects for calculating and averaging across batches
        self.train_metrics = ModuleDict(
            {
                "atom_types_loss": MeanMetric(),
                "pos_loss": MeanMetric(),
                "frac_coords_loss": MeanMetric(),
                "lengths_scaled_loss": MeanMetric(),
                "angles_radians_loss": MeanMetric(),
                "loss": MeanMetric(),
                "aux_global_property_loss": MeanMetric(),
                "aux_global_energy_loss": MeanMetric(),
                "aux_atomic_forces_loss": MeanMetric(),
                "dataset_idx": MeanMetric(),
            }
        )
        if (
            "qm9" in self.hparams.datasets
            and self.hparams.datasets["qm9"].global_property is not None
        ):
            for target in QM9_TARGETS:
                if self.hparams.datasets["qm9"].global_property in ("all", target):
                    self.train_metrics[f"aux_global_property_loss_{target}_scaled"] = MeanMetric()
        for dataset in ("omol25", "mptrj"):
            if (
                dataset in self.hparams.datasets
                and self.hparams.datasets[dataset].global_energy is not None
            ):
                self.train_metrics["aux_global_energy_loss_scaled"] = MeanMetric()
                self.train_metrics["aux_global_energy_per_atom_loss_scaled"] = MeanMetric()
                self.train_metrics["aux_atomic_forces_loss_scaled"] = MeanMetric()
                break

        val_metrics = {}
        for dataset, cfg in self.hparams.datasets.items():
            if not (cfg.proportion > 0.0):
                continue
            # General evaluation metrics
            val_metrics[dataset] = {
                "atom_types_loss": MeanMetric(),
                "pos_loss": MeanMetric(),
                "frac_coords_loss": MeanMetric(),
                "lengths_scaled_loss": MeanMetric(),
                "angles_radians_loss": MeanMetric(),
                "loss": MeanMetric(),
                "aux_global_property_loss": MeanMetric(),
                "aux_global_energy_loss": MeanMetric(),
                "aux_atomic_forces_loss": MeanMetric(),
                "valid_rate": MeanMetric(),
                "unique_rate": MeanMetric(),
                "sampling_time": MeanMetric(),
            }
            # Periodic sample evaluation metrics
            if cfg.id in self.periodic_dataset_ids:
                if dataset == "qmof150":
                    val_metrics[dataset].update(
                        {
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
                        }
                    )
                else:
                    val_metrics[dataset].update(
                        {
                            "novel_rate": MeanMetric(),
                            "struct_valid_rate": MeanMetric(),
                            "comp_valid_rate": MeanMetric(),
                        }
                    )
            # Non-periodic sample evaluation metrics
            elif cfg.id in self.non_periodic_dataset_ids:
                val_metrics[dataset].update(
                    {
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
                        "posebusters_rate": MeanMetric(),
                    }
                )
            val_metrics[dataset] = ModuleDict(val_metrics[dataset])

        if (
            "qm9" in self.hparams.datasets
            and self.hparams.datasets["qm9"].global_property is not None
        ):
            for target in QM9_TARGETS:
                if self.hparams.datasets["qm9"].global_property in ("all", target):
                    val_metrics["qm9"][f"aux_global_property_loss_{target}_scaled"] = MeanMetric()
                    if "matbench" in val_metrics:
                        val_metrics["matbench"][
                            f"aux_global_property_loss_{target}_scaled"
                        ] = MeanMetric()
        for dataset in ("omol25", "mptrj"):
            if (
                dataset in self.hparams.datasets
                and self.hparams.datasets[dataset].global_energy is not None
            ):
                val_metrics[dataset]["aux_global_energy_loss_scaled"] = MeanMetric()
                val_metrics[dataset]["aux_global_energy_per_atom_loss_scaled"] = MeanMetric()
                val_metrics[dataset]["aux_atomic_forces_loss_scaled"] = MeanMetric()

        self.val_metrics = ModuleDict(val_metrics)
        self.test_metrics = copy.deepcopy(self.val_metrics)

        # Load bincounts for sampling dynamically
        self.num_nodes_bincount = {}
        self.spacegroups_bincount = {}

        for dataset, cfg in self.hparams.datasets.items():
            if not (cfg.proportion > 0.0):
                continue
            # Map dataset name to directory name if needed.
            # Existing logic hardcoded paths:
            # mp20 -> mp_20, qmof150 -> qmof
            dir_name = dataset
            if dataset == "mp20":
                dir_name = "mp_20"
            elif dataset == "qmof150":
                dir_name = "qmof"

            # num_nodes_bincount
            nodes_path = os.path.join(
                self.hparams.sampling.data_dir, dir_name, "num_nodes_bincount.pt"
            )
            if os.path.exists(nodes_path):
                self.num_nodes_bincount[dataset] = torch.nn.Parameter(
                    torch.load(nodes_path, map_location="cpu"),  # nosec
                    requires_grad=False,
                )
            else:
                self.num_nodes_bincount[dataset] = None

            # spacegroups_bincount
            sg_path = os.path.join(
                self.hparams.sampling.data_dir, dir_name, "spacegroups_bincount.pt"
            )
            if os.path.exists(sg_path):
                self.spacegroups_bincount[dataset] = torch.nn.Parameter(
                    torch.load(sg_path, map_location="cpu"),  # nosec
                    requires_grad=False,
                )
            else:
                self.spacegroups_bincount[dataset] = None

        # Model configuration state
        self.model_configured = False

        # Constants
        self.register_buffer(
            "periodic_datasets",
            torch.tensor(self.periodic_dataset_ids, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "dataset_index_to_idx",
            torch.tensor(list(self.index_to_idx.values()), dtype=torch.long),
            persistent=False,
        )

    @typecheck
    def forward(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Perform a forward pass.

        Args:
            batch: A batch of data (a tuple) containing the input tensors and target labels.

        Returns:
            A dictionary of loss values.
        """
        # Subset distributions when overfitting (one-time only)
        overfitting = self.trainer.overfit_batches == 1
        if overfitting and self.trainer.global_step == 0:
            batch_num_nodes = torch.bincount(batch.batch)
            for dataset_index in batch.dataset_idx.unique_consecutive():
                num_nodes = batch_num_nodes[batch.dataset_idx == dataset_index]
                spacegroups = batch.spacegroup[batch.dataset_idx == dataset_index]
                dataset_name = self.index_to_dataset[dataset_index.item()]

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

        # Prepare batch metadata
        self.max_num_nodes = max(
            len(self.num_nodes_bincount[dataset]) - 1
            for dataset, cfg in self.hparams.datasets.items()
            if cfg.proportion > 0.0
        )
        if hasattr(self.model, "jvp_attn") and self.model.jvp_attn:
            # Find the smallest power of 2 >= max(max_num_nodes, 32)
            min_num_nodes = max(self.max_num_nodes, 32)
            closest_power_of_2 = 1 << (min_num_nodes - 1).bit_length()
            self.max_num_nodes = int(closest_power_of_2)
        if (
            hasattr(self.model, "context_length")
            and self.model.context_length < self.max_num_nodes
        ):
            raise ValueError(
                f"Model context length ({self.model.context_length}) is smaller than max_num_nodes ({self.max_num_nodes})."
            )

        # Densify batch
        token_is_periodic, _ = to_dense_batch(
            batch.node_is_periodic,
            batch.batch,
            max_num_nodes=self.max_num_nodes,
        )
        atom_types, mask = to_dense_batch(
            batch.atom_types, batch.batch, max_num_nodes=self.max_num_nodes
        )
        pos = (
            to_dense_batch(batch.pos, batch.batch, max_num_nodes=self.max_num_nodes)[0]
            * self.hparams.augmentations.scale
        )
        frac_coords, _ = to_dense_batch(
            batch.frac_coords, batch.batch, max_num_nodes=self.max_num_nodes
        )
        lengths_scaled = batch.lengths_scaled.unsqueeze(-2)  # Handle as global feature
        angles_radians = batch.angles_radians.unsqueeze(-2)  # Handle as global feature

        # Prepare conditioning inputs
        use_cfg = self.model.class_dropout_prob > 0
        dataset_idx = self.dataset_index_to_idx[batch.dataset_idx] + int(
            use_cfg
        )  # 0 -> null class (for classifier-free guidance or CFG)
        # if not self.hparams.conditioning.dataset_idx:
        #     dataset_idx = torch.zeros_like(dataset_idx)
        spacegroup = batch.spacegroup
        if not self.hparams.conditioning.spacegroup:
            spacegroup = torch.zeros_like(batch.spacegroup)

        # Build features for conditioning
        # NOTE: Atoms and tokens are currently treated synonymously (i.e. one atom per token)
        num_atoms = num_tokens = atom_types.shape[1]
        ref_pos = (
            torch.zeros_like(pos) * self.hparams.augmentations.ref_scale
        )  # Use scaled reference positions - (batch_size, num_atoms, 3)
        atom_to_token_idx = torch.arange(num_atoms, device=self.device).expand(
            batch.batch_size, -1
        )  # (batch_size, num_atoms)
        atom_to_token = F.one_hot(atom_to_token_idx, num_classes=num_tokens).type(
            torch.float32
        )  # (batch_size, num_atoms, num_tokens)

        # Prepare batch for forward pass
        max_num_tokens = mask.sum(dim=1)  # (batch_size,)
        dense_batch = TensorDict(
            {
                # modalities to predict
                "atom_types": F.one_hot(atom_types, num_classes=self.model.vocab_size),
                "pos": pos,
                "frac_coords": frac_coords,
                "lengths_scaled": lengths_scaled,
                "angles_radians": angles_radians,
                # # auxiliary prediction tasks
                # "global_property": torch.randn((batch.batch_size, 1), device=self.device),
                # "global_energy": torch.randn((batch.batch_size, 1), device=self.device),
                # "atomic_forces": torch.randn_like(pos),
                # features for conditioning
                "dataset_idx": dataset_idx,
                "spacegroup": spacegroup,
                "charge": batch.charge,
                "spin": batch.spin,
                "ref_pos": ref_pos,
                "ref_space_uid": atom_to_token_idx,
                "atom_to_token": atom_to_token,
                "atom_to_token_idx": atom_to_token_idx,
                "token_index": atom_to_token_idx,
                "max_num_tokens": max_num_tokens,
                # metadata
                "padding_mask": ~mask,
                "token_is_periodic": token_is_periodic,
            },
            batch_size=batch.batch_size,
            device=self.device,
        )

        # Add auxiliary targets to dense batch if applicable
        is_qm9_dataset = batch.dataset_idx == self.dataset_to_index.get("qm9", -1)
        is_matbench_dataset = batch.dataset_idx == self.dataset_to_index.get("matbench", -1)
        is_omol25_dataset = batch.dataset_idx == self.dataset_to_index.get("omol25", -1)
        is_mptrj_dataset = batch.dataset_idx == self.dataset_to_index.get("mptrj", -1)

        is_qm9_property_training = (
            is_qm9_dataset.any() and self.hparams.datasets["qm9"].global_property is not None
        )
        is_matbench_property_training = is_matbench_dataset.any()
        is_omol25_energy_training = (
            is_omol25_dataset.any() and self.hparams.datasets["omol25"].global_energy is not None
        )
        is_mptrj_energy_training = (
            is_mptrj_dataset.any() and self.hparams.datasets["mptrj"].global_energy is not None
        )

        if is_qm9_property_training or is_matbench_property_training:
            dense_batch["global_property"] = batch.y

        if is_omol25_energy_training or is_mptrj_energy_training:
            global_energy, _ = to_dense_batch(
                batch.y[:, 0:1],
                batch.batch,
                max_num_nodes=self.max_num_nodes,
            )
            atomic_forces, _ = to_dense_batch(
                batch.y[:, 1:4],
                batch.batch,
                max_num_nodes=self.max_num_nodes,
            )
            dense_batch["global_energy"] = global_energy[:, 0, :]
            dense_batch["atomic_forces"] = atomic_forces

        # Run forward pass
        loss_dict, _ = self.model.forward(dense_batch, compute_stats=False)

        # Sum losses over global properties
        aux_global_property_loss = loss_dict["aux_global_property_loss"]
        loss_dict["aux_global_property_loss"] = aux_global_property_loss.sum()

        # Recompute loss when finetuning
        if self.hparams.task_name == "finetune_fm":
            loss_dict["loss"] = sum(v for k, v in loss_dict.items() if k.startswith("aux_"))

        # Split global property metrics for logging
        pred_aux_global_property = loss_dict.pop("pred_aux_global_property")
        target_aux_global_property = loss_dict.pop("target_aux_global_property")
        mask_aux_global_property = loss_dict.pop("mask_aux_global_property")
        if (
            "qm9" in self.hparams.datasets
            and self.hparams.datasets["qm9"].global_property is not None
        ):
            for name, idx in QM9_TARGET_NAME_TO_IDX.items():
                if self.hparams.datasets["qm9"].global_property in ("all", name):
                    if is_qm9_dataset.any() or is_matbench_dataset.any():
                        aux_prop_scale = torch.ones_like(pred_aux_global_property[:, idx])
                        aux_prop_shift = torch.zeros_like(pred_aux_global_property[:, idx])
                        aux_to_lit_scale = torch.ones_like(pred_aux_global_property[:, idx])
                        aux_prop_scale[is_qm9_dataset] = (
                            self.trainer.datamodule.qm9_train_prop_std[0, idx]
                        )
                        aux_prop_shift[is_qm9_dataset] = (
                            self.trainer.datamodule.qm9_train_prop_mean[0, idx]
                        )
                        aux_prop_scale[is_matbench_dataset] = (
                            self.trainer.datamodule.matbench_train_dataset.scale[0, idx]
                        ).nan_to_num(1.0)
                        aux_prop_shift[is_matbench_dataset] = (
                            self.trainer.datamodule.matbench_train_dataset.shift[0, idx]
                        ).nan_to_num(0.0)
                        aux_to_lit_scale[is_qm9_dataset] = QM9_TARGET_NAME_TO_LITERATURE_SCALE[
                            name
                        ]
                        aux_to_lit_scale[is_matbench_dataset] = (
                            MATBENCH_QM9_TARGET_NAME_TO_LITERATURE_SCALE[name]
                        )
                        aux_prop_pred = (
                            pred_aux_global_property[:, idx] * aux_prop_scale + aux_prop_shift
                        ) * aux_to_lit_scale
                        aux_prop_target = (
                            target_aux_global_property[:, idx] * aux_prop_scale + aux_prop_shift
                        ) * aux_to_lit_scale
                        aux_prop_mask = mask_aux_global_property[:, idx]
                        aux_prop_err = (aux_prop_pred - aux_prop_target) * aux_prop_mask
                        aux_prop_loss_value = aux_prop_err.abs().sum() / (
                            aux_prop_mask.sum() + 1e-6
                        )
                    else:
                        aux_prop_loss_value = torch.tensor(0.0, device=self.device)
                    loss_dict[f"aux_global_property_loss_{name}_scaled"] = aux_prop_loss_value

        # Prepare global energy and atomic forces predictions/targets for logging
        pred_aux_global_energy = loss_dict.pop("pred_aux_global_energy")
        pred_aux_atomic_forces = loss_dict.pop("pred_aux_atomic_forces")
        target_aux_global_energy = loss_dict.pop("target_aux_global_energy")
        target_aux_atomic_forces = loss_dict.pop("target_aux_atomic_forces")
        mask_aux_global_energy = loss_dict.pop("mask_aux_global_energy")
        mask_aux_atomic_forces = loss_dict.pop("mask_aux_atomic_forces")
        if (
            "omol25" in self.hparams.datasets
            and self.hparams.datasets["omol25"].global_energy is not None
        ) or (
            "mptrj" in self.hparams.datasets
            and self.hparams.datasets["mptrj"].global_energy is not None
        ):
            if is_omol25_dataset.any() or is_mptrj_dataset.any():
                aux_energy_scale = torch.ones_like(pred_aux_global_energy)
                aux_energy_shift = torch.zeros_like(pred_aux_global_energy)
                aux_energy_scale[is_omol25_dataset] = (
                    self.trainer.datamodule.omol25_train_dataset.scale
                )
                aux_energy_shift[is_omol25_dataset] = (
                    self.trainer.datamodule.omol25_train_dataset.shift
                )
                aux_energy_scale[is_mptrj_dataset] = (
                    self.trainer.datamodule.mptrj_train_dataset.scale
                )
                aux_energy_shift[is_mptrj_dataset] = (
                    self.trainer.datamodule.mptrj_train_dataset.shift
                )
                # Global energy mean absolute error (in meV <- eV)
                aux_energy_pred = (
                    pred_aux_global_energy * aux_energy_scale + aux_energy_shift
                ) * EV_TO_MEV
                aux_energy_target = (
                    target_aux_global_energy * aux_energy_scale + aux_energy_shift
                ) * EV_TO_MEV
                aux_energy_err = (aux_energy_pred - aux_energy_target) * mask_aux_global_energy
                aux_energy_loss_value = aux_energy_err.abs().sum() / (
                    mask_aux_global_energy.sum() + 1e-6
                )
                # Global energy per atom mean absolute error (in meV <- eV)
                aux_energy_pred_per_atom = aux_energy_pred / max_num_tokens.unsqueeze(-1)
                aux_energy_target_per_atom = aux_energy_target / max_num_tokens.unsqueeze(-1)
                aux_energy_per_atom_err = (
                    aux_energy_pred_per_atom - aux_energy_target_per_atom
                ) * mask_aux_global_energy
                aux_energy_per_atom_loss_value = aux_energy_per_atom_err.abs().sum() / (
                    mask_aux_global_energy.sum() + 1e-6
                )
                # Atomic forces mean absolute error (in meV/Å <- eV/Å)
                aux_force_pred = (
                    pred_aux_atomic_forces * aux_energy_scale.unsqueeze(-1) * EV_TO_MEV
                )
                aux_force_target = (
                    target_aux_atomic_forces * aux_energy_scale.unsqueeze(-1) * EV_TO_MEV
                )
                aux_force_err = (aux_force_pred - aux_force_target) * mask_aux_atomic_forces
                aux_force_loss_value = aux_force_err.abs().sum() / (
                    mask_aux_atomic_forces.sum() + 1e-6
                )
            else:
                aux_energy_loss_value = torch.tensor(0.0, device=self.device)
                aux_energy_per_atom_loss_value = torch.tensor(0.0, device=self.device)
                aux_force_loss_value = torch.tensor(0.0, device=self.device)
            loss_dict["aux_global_energy_loss_scaled"] = aux_energy_loss_value
            loss_dict["aux_global_energy_per_atom_loss_scaled"] = aux_energy_per_atom_loss_value
            loss_dict["aux_atomic_forces_loss_scaled"] = aux_force_loss_value

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
            sample_is_periodic = torch.isin(batch.dataset_idx, self.periodic_datasets)
            batch.sample_is_periodic = sample_is_periodic
            batch.node_is_periodic = sample_is_periodic[batch.batch]

            # Center non-periodic molecules at origin before any augmentations
            pos_mean = scatter_mean_torch(src=batch.pos, index=batch.batch, dim=0)
            batch.pos[~batch.node_is_periodic] -= pos_mean[batch.batch][~batch.node_is_periodic]

            if self.hparams.augmentations.multiplicity > 1:
                # Augment batch (e.g., by random 3D rotations and translations) multiple times
                orig_batch_size = batch.num_graphs
                batch = Batch.from_data_list(
                    [copy.deepcopy(batch) for _ in range(self.hparams.augmentations.multiplicity)]
                )
                batch._num_graphs = self.hparams.augmentations.multiplicity * orig_batch_size
                # batch.id = [i for id_list in batch.id for i in id_list]
                # batch.ptr = torch.cat([
                #     torch.tensor([0], device=self.device, dtype=torch.long),
                #     torch.cumsum(torch.bincount(batch.batch), dim=0)
                # ])

            if self.hparams.augmentations.pos is True:
                rot_mat = sample_uniform_rotation(
                    shape=batch.cell.shape[:-2],
                    dtype=batch.pos.dtype,
                    device=self.device,
                )
                rot_for_nodes = rot_mat[batch.batch]
                pos_aug = torch.einsum("bi,bij->bj", batch.pos, rot_for_nodes.transpose(-2, -1))
                batch.pos = pos_aug
                cell_aug = torch.einsum("bij,bjk->bik", batch.cell, rot_mat.transpose(-2, -1))
                batch.cell = cell_aug
                # # NOTE: Fractional coordinates are rotation-invariant
                # cell_per_node_inv = torch.linalg.inv(
                #     batch.cell[batch.batch][batch.node_is_periodic]
                # )
                # assert torch.allclose(
                #     batch.frac_coords[batch.node_is_periodic],
                #     torch.einsum("bi,bij->bj", pos_aug[batch.node_is_periodic], cell_per_node_inv) % 1.0,
                #     rtol=1e-3,
                #     atol=1e-3,
                # )

                is_omol25_dataset = batch.dataset_idx == self.dataset_to_index.get("omol25", -1)
                is_mptrj_dataset = batch.dataset_idx == self.dataset_to_index.get("mptrj", -1)
                if is_omol25_dataset.any():
                    # Rotate non-periodic atomic forces accordingly
                    # TODO: Decide if non-periodic forces need to be zero-centered just like `batch.pos[~batch.node_is_periodic]`
                    is_omol25_node = is_omol25_dataset[batch.batch]
                    forces_aug = torch.einsum(
                        "bi,bij->bj", batch.y[:, 1:4], rot_for_nodes.transpose(-2, -1)
                    )
                    batch.y[is_omol25_node, 1:4] = forces_aug[is_omol25_node]
                # if is_mptrj_dataset.any():
                #     # NOTE: Force rotation augmentation is not implemented for MPtrj dataset samples
                #     pass

            if self.hparams.augmentations.frac_coords is True:
                if batch.sample_is_periodic.any():
                    # Sample random translation vector from periodic batch length distribution / 2
                    random_translation = (
                        torch.normal(
                            torch.abs(batch.lengths[batch.sample_is_periodic].mean(dim=0)),
                            torch.abs(
                                batch.lengths[batch.sample_is_periodic].std(dim=0).nan_to_num(1e-8)
                            ),
                        )
                        / 2
                    )
                    # Apply same random translation to all (periodic) Cartesian coordinates
                    pos_aug = batch.pos + random_translation
                    batch.pos[batch.node_is_periodic] = pos_aug[batch.node_is_periodic].type(
                        batch.pos.dtype
                    )
                    # Compute new fractional coordinates for periodic samples
                    cell_per_node_inv = torch.linalg.inv(
                        # NOTE: `torch.linalg.inv` does not support low precision dtypes
                        batch.cell[batch.batch][batch.node_is_periodic].float()
                    )
                    frac_coords_aug = torch.einsum(
                        "bi,bij->bj",
                        batch.pos[batch.node_is_periodic],
                        cell_per_node_inv,
                    )
                    frac_coords_aug = frac_coords_aug % 1.0
                    batch.frac_coords[batch.node_is_periodic] = frac_coords_aug.type(
                        batch.frac_coords.dtype
                    )
                    # # NOTE: Fractional coordinates are (still) rotation-invariant
                    # cell_per_node_inv = torch.linalg.inv(
                    #     batch.cell[batch.batch][batch.node_is_periodic]
                    # )
                    # assert torch.allclose(
                    #     batch.frac_coords[batch.node_is_periodic],
                    #     torch.einsum("bi,bij->bj", pos_aug[batch.node_is_periodic], cell_per_node_inv) % 1.0,
                    #     rtol=1e-3,
                    #     atol=1e-3,
                    # )

                    is_mptrj_dataset = batch.dataset_idx == self.dataset_to_index.get("mptrj", -1)
                    if is_mptrj_dataset.any():
                        raise NotImplementedError(
                            "Fractional coordinate augmentation is not implemented for MPtrj dataset samples."
                        )

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
            torch.tensor(self.global_step, device=self.device, dtype=torch.float32),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        # Log throughput metrics
        step_time = time.time() - t_start
        examples_per_second = torch.tensor(
            batch.batch_size / step_time, device=self.device, dtype=torch.float32
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

    @typecheck
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

    @typecheck
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

        metrics = getattr(self, f"{stage}_metrics")[self.index_to_dataset[dataloader_idx]]
        generation_evaluator = getattr(self, f"{stage}_generation_evaluators")[
            self.index_to_dataset[dataloader_idx]
        ]
        generation_evaluator.device = metrics["loss"].device

        # Save masks used to apply augmentations
        sample_is_periodic = torch.isin(batch.dataset_idx, self.periodic_datasets)
        batch.node_is_periodic = sample_is_periodic[batch.batch]

        # Forward pass with loss calculation
        loss_dict = self.forward(batch)

        # Update and log per-step eval metrics
        for k, v in loss_dict.items():
            metrics[k](v)
            self.log(
                f"{stage}_{self.index_to_dataset[dataloader_idx]}/{k}",
                metrics[k],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                add_dataloader_idx=False,
            )

    @typecheck
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
                range(
                    0,
                    self.hparams.sampling.num_samples,
                    self.hparams.sampling.batch_size,
                ),
                desc="    Sampling",
            ):
                # Perform sampling and decoding to crystal structures
                out, batch = self.sample_and_decode(
                    num_nodes_bincount=self.num_nodes_bincount[dataset],
                    spacegroups_bincount=self.spacegroups_bincount[dataset],
                    batch_size=self.hparams.sampling.batch_size,
                    cfg_scale=self.hparams.sampling.cfg_scale,
                    dataset_idx=self.dataset_to_index.get(dataset, -1),
                    steps=self.hparams.sampling.get("steps", 100),
                )
                # Save predictions for metrics and visualisation
                start_idx = 0
                for idx_in_batch, num_atom in enumerate(batch["num_atoms"].tolist()):
                    _atom_types = out["atom_types"].narrow(0, start_idx, num_atom)
                    _atom_types[_atom_types == 0] = 1  # Atom type 0 -> 1 (H) to prevent crash
                    _atom_types[_atom_types == self.hparams.sampling.mask_token_index] = (
                        1  # Mask atom type -> 1 (H) to prevent crash
                    )
                    _pos = out["pos"].narrow(0, start_idx, num_atom)
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
            save_dir = os.path.join(
                self.hparams.sampling.save_dir, f"{dataset}_{stage}_{self.global_rank}"
            )
            gen_metrics_dict = generation_evaluators[dataset].get_metrics(
                save=self.hparams.sampling.visualize,
                save_dir=save_dir,
                n_jobs=16,
            )
            gen_metrics_dict["sampling_time"] = t_end - t_start
            for k, v in gen_metrics_dict.items():
                metrics[dataset][k](v)
                self.log(
                    f"{stage}_{dataset}/{k}",
                    metrics[dataset][k],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True if k in ("valid_rate", "posebusters_rate") else False,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

            # For materials, save as a `.pt` file for easier in-depth evaluation later
            sample_is_periodic = torch.isin(
                self.dataset_to_index.get(dataset, -1), self.periodic_datasets
            )
            if sample_is_periodic.any():
                gen_save_dir = os.path.join(save_dir, f"generate_{self.global_rank:02d}")
                os.makedirs(gen_save_dir, exist_ok=True)

                predictions = [
                    {
                        "frac_coords": torch.tensor(x["frac_coords"]),
                        "atom_types": torch.tensor(x["atom_types"]),
                        "lattices": lattice_params_to_matrix_torch(
                            torch.tensor(x["lengths"])[None, :],
                            torch.tensor(x["angles"])[None, :],
                        ),
                        "lengths": torch.tensor([x["lengths"]]),
                        "angles": torch.tensor([x["angles"]]),
                        "num_atoms": torch.tensor([len(x["atom_types"])]),
                    }
                    for x in generation_evaluators[dataset].pred_arrays_list
                ]
                batch_indices = [[0] for _ in range(len(predictions))]
                torch.save(
                    [predictions],
                    os.path.join(gen_save_dir, f"predictions_{self.global_rank:02d}.pt"),
                )
                torch.save(
                    [batch_indices],
                    os.path.join(gen_save_dir, f"batch_indices_{self.global_rank:02d}.pt"),
                )

                # Record the number of sampling steps
                (Path(gen_save_dir) / "num_steps.txt").write_text(
                    str(self.hparams.sampling.get("steps", 100))
                )

            # Maybe log sample visualizations to WandB
            if self.hparams.sampling.visualize and isinstance(self.logger, WandbLogger):
                pred_table = generation_evaluators[dataset].get_wandb_table(
                    current_epoch=self.current_epoch,
                    save_dir=os.path.join(
                        self.hparams.sampling.save_dir,
                        f"{dataset}_{stage}_{self.global_rank}",
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
        cfg_scale: float = 0.0,
        dataset_idx: int = 0,
        steps: int = 100,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Sample and decode a batch of crystal structures.

        Args:
            num_nodes_bincount: A tensor containing the number of nodes for each crystal structure.
            spacegroups_bincount: A tensor containing the space group information for each crystal structure.
            batch_size: The number of crystal structures to sample.
            dataset_idx: The index of the dataset to sample from.
            steps: The number of ODE steps to use for sampling. Only applicable if using flow matching-based sampling.

        Returns:
            A tuple containing the sampled modalities and the original batch.
        """
        sample_is_periodic = torch.isin(dataset_idx, self.periodic_datasets)

        # Sample random lengths from distribution: (B, 1)
        sample_lengths = torch.multinomial(
            num_nodes_bincount.float(),
            batch_size,
            replacement=True,
        ).to(self.device)

        # Create dataset_idx tensor
        # NOTE 0 -> null class within model, while 0 -> MP20 elsewhere, so increment by 1 (for classifier-free guidance or CFG)
        use_cfg = self.model.class_dropout_prob > 0
        dataset_idx = torch.full(
            (batch_size,),
            dataset_idx + int(use_cfg),
            dtype=torch.int64,
            device=self.device,
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
        token_mask = torch.zeros(
            batch_size,
            self.max_num_nodes,
            dtype=torch.bool,
            device=self.device,
        )
        for idx, length in enumerate(sample_lengths):
            token_mask[idx, :length] = True

        # Craft initial modalities
        atom_types = torch.zeros(
            (batch_size, self.max_num_nodes), dtype=torch.long, device=self.device
        )
        pos = (
            torch.zeros(
                (batch_size, self.max_num_nodes, 3),
                dtype=torch.float32,
                device=self.device,
            )
            * self.hparams.augmentations.scale
        )
        frac_coords = torch.zeros(
            (batch_size, self.max_num_nodes, 3), dtype=torch.float32, device=self.device
        )
        lengths_scaled = torch.zeros((batch_size, 1, 3), dtype=torch.float32, device=self.device)
        angles_radians = torch.zeros((batch_size, 1, 3), dtype=torch.float32, device=self.device)

        # Build features for conditioning
        # NOTE: Atoms and tokens are currently treated synonymously (i.e. one atom per token)
        num_atoms = num_tokens = pos.shape[1]

        ref_pos = (
            torch.zeros_like(pos) * self.hparams.augmentations.ref_scale
        )  # Use scaled reference positions - (batch_size, num_atoms, 3)
        atom_to_token_idx = torch.arange(num_atoms, device=self.device).expand(
            batch_size, -1
        )  # (batch_size, num_atoms)
        atom_to_token = F.one_hot(atom_to_token_idx, num_classes=num_tokens).type(
            torch.float32
        )  # (batch_size, num_atoms, num_tokens)

        token_is_periodic = torch.zeros_like(token_mask)
        token_is_periodic[sample_is_periodic] = True

        charge = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        spin = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # Prepare batch for sampling
        max_num_tokens = token_mask.sum(dim=1)  # (batch_size,)
        dense_batch = TensorDict(
            {
                # modalities to predict
                "atom_types": F.one_hot(atom_types, num_classes=self.model.vocab_size),
                "pos": pos,
                "frac_coords": frac_coords,
                "lengths_scaled": lengths_scaled,
                "angles_radians": angles_radians,
                # features for conditioning
                "dataset_idx": dataset_idx,
                "spacegroup": spacegroup,
                "charge": charge,
                "spin": spin,
                "ref_pos": ref_pos,
                "ref_space_uid": atom_to_token_idx,
                "atom_to_token": atom_to_token,
                "atom_to_token_idx": atom_to_token_idx,
                "token_index": atom_to_token_idx,
                "max_num_tokens": max_num_tokens,
                # metadata
                "padding_mask": ~token_mask,
                "token_is_periodic": token_is_periodic * token_mask,
            },
            batch_size=batch_size,
            device=self.device,
        )

        # Sample modalities
        sampled_x_1, _ = self.model.sample(
            dense_batch,
            steps=steps,
            cfg_scale=cfg_scale,
            use_cfg=use_cfg and cfg_scale != 0.0,
        )

        # Collect final sample modalities and remove padding (to convert to PyG format)
        out = {
            "atom_types": sampled_x_1["atom_types"].argmax(-1)[token_mask],
            "pos": sampled_x_1["pos"][token_mask] / self.hparams.augmentations.scale,
            "frac_coords": sampled_x_1["frac_coords"][token_mask],
            "lengths_scaled": sampled_x_1["lengths_scaled"].squeeze(-2),
            "angles_radians": sampled_x_1["angles_radians"].squeeze(-2),
        }

        batch = {
            "num_atoms": sample_lengths,
            # "batch": torch.repeat_interleave(
            #     torch.arange(len(sample_lengths), device=self.device), sample_lengths
            # ),
            # "token_idx": (torch.cumsum(token_mask, dim=-1, dtype=torch.int64) - 1)[token_mask],
        }
        return out, batch

    #####################################################################################################

    def configure_model(self):
        """Configure the model to be used for training, validation, testing, or prediction."""
        if self.model_configured:
            return

        if self.trainer is not None:
            sleep = self.trainer.global_rank
            log.info(f"Rank {self.trainer.global_rank}: Sleeping for {sleep}s to avoid CPU OOMs.")
            time.sleep(sleep)

        # In a memory-efficient way, exclude certain (unmanaged) modules from being managed by FSDP
        if isinstance(self.trainer.strategy, FSDPStrategy) and hasattr(self, "ignored_modules"):
            ignored_modules = []
            model_list_grouped_modules = [
                lst
                for n in dir(self.model)
                if isinstance(getattr(self.model, n), list)
                and all(isinstance(x, torch.nn.Module) for x in getattr(self.model, n))
                for lst in getattr(self.model, n)
            ]
            model_modules = list(self.model.modules()) + model_list_grouped_modules
            for module in model_modules:
                if module.__class__ in self.ignored_modules:
                    ignored_modules.append(module)

            self.trainer.strategy.kwargs["ignored_modules"] = ignored_modules

        if self.hparams.compile:
            # Prefer `self.model.compile` over `torch.compile(self.model)` to avoid `_orig_` prefix checkpoint issues.
            # Reference: https://github.com/Lightning-AI/pytorch-lightning/issues/20233#issuecomment-2868169706.
            log.info(
                f"Rank {self.trainer.global_rank}: Compiling model with `torch.compile(fullgraph=True)`."
            )
            self.model.compile(fullgraph=True)

        # Using WandB, log model gradients every N steps
        if (
            isinstance(self.hparams.log_grads_every_n_steps, int)
            and self.hparams.log_grads_every_n_steps > 0
            and isinstance(self.logger, WandbLogger)
        ):
            log.info(
                f"Rank {self.trainer.global_rank}: Logging model gradients to WandB every {self.hparams.log_grads_every_n_steps} steps."
            )
            self.logger.watch(
                self.model,
                log="gradients",
                log_freq=self.hparams.log_grads_every_n_steps,
                log_graph=False,
            )

        # Finalize model configuration in case this hook is called multiple times
        self.model_configured = True

    @typecheck
    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        Applies scheduler only if provided and the current world size is greater than the base world size.
        Reference: https://arxiv.org/abs/1706.02677.

        Returns:
            A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        trainable_parameters = list(
            filter(lambda p: p.requires_grad, self.trainer.model.parameters())
        )
        try:
            optimizer = self.hparams.optimizer(params=trainable_parameters)
        except TypeError:
            # NOTE: Strategies such as DeepSpeed require `params` to instead be specified as `model_params`
            optimizer = self.hparams.optimizer(model_params=trainable_parameters)

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

    @typecheck
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        """Called when loading a checkpoint. Adds special handling for finetuning task.

        Args:
            checkpoint: The checkpoint dictionary to be loaded.
        """
        finetuning = self.hparams.task_name == "finetune_fm"
        init_run = (
            self.trainer.ckpt_path is not None
            and self.trainer.pretrained_ckpt_path is not None
            and Path(self.trainer.ckpt_path).samefile(Path(self.trainer.pretrained_ckpt_path))
        )

        if finetuning and init_run:
            # Reinitialize loop, optimizer, and relevant task
            # weights when beginning a new finetuning run
            del checkpoint["loops"]
            checkpoint["optimizer_states"] = []

            aux_tasks = set()
            if (
                "qm9" in self.hparams.datasets
                and self.hparams.datasets["qm9"].global_property is not None
            ):
                aux_tasks.add("global_property")
            if (
                "omol25" in self.hparams.datasets
                and self.hparams.datasets["omol25"].global_energy is not None
            ) or (
                "mptrj" in self.hparams.datasets
                and self.hparams.datasets["mptrj"].global_energy is not None
            ):
                aux_tasks.add("global_energy")
                aux_tasks.add("atomic_forces")

            heads_to_finetune = set()
            for aux_task in aux_tasks:
                for state in checkpoint["state_dict"]:
                    if f"{aux_task}_head" in state:
                        heads_to_finetune.add(state)

            for state in heads_to_finetune:
                del checkpoint["state_dict"][state]
