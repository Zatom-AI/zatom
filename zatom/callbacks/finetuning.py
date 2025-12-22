from lightning import LightningModule
from lightning.pytorch.callbacks.finetuning import BaseFinetuning
from torch.optim import Optimizer


class AuxiliaryTaskFinetuning(BaseFinetuning):
    """A finetuning callback that freezes the entire model initially and then unfreezes only the
    auxiliary task heads for finetuning."""

    def __init__(self):
        super().__init__()

    def freeze_before_training(self, pl_module: LightningModule):
        """Freeze the entire model initially, excluding the auxiliary task heads.

        Args:
            pl_module: The LightningModule containing the model to be frozen.

        Raises:
            AttributeError: If the provided LightningModule is missing required attributes.
        """
        if not hasattr(pl_module, "model"):
            raise AttributeError(
                "LightningModule must have a 'model' attribute for AuxiliaryTaskFinetuning."
            )
        if not hasattr(pl_module.model, "model"):
            raise AttributeError(
                "The model must have a 'model' attribute for AuxiliaryTaskFinetuning."
            )
        if not hasattr(pl_module.model, "auxiliary_tasks"):
            raise AttributeError(
                "The model must have an 'auxiliary_tasks' attribute for AuxiliaryTaskFinetuning."
            )
        if not isinstance(pl_module.model.auxiliary_tasks, list):
            raise TypeError(
                "The model's 'auxiliary_tasks' must be a list of auxiliary task names."
            )

        # Freeze all model parameters except auxiliary task heads
        named_modules = list(pl_module.model.model.named_modules())[
            1:
        ]  # Skip the top-level module
        for aux_task in pl_module.model.auxiliary_tasks:
            global_property_pretraining = (
                pl_module.hparams.datasets["qm9"].global_property is None
                and pl_module.hparams.datasets["matbench"].proportion == 0.0
            )
            if aux_task == "global_property" and global_property_pretraining:
                # NOTE: This does not allow one to finetune on Matbench
                # for any task other than property prediction.
                continue
            # Check all datasets for global_energy, not just omol25
            if aux_task in ("global_energy", "atomic_forces"):
                # Check if any dataset has global_energy enabled
                has_global_energy = any(
                    hasattr(pl_module.hparams.datasets[dataset], "global_energy")
                    and pl_module.hparams.datasets[dataset].global_energy is not None
                    for dataset in pl_module.hparams.datasets
                )
                if not has_global_energy:
                    continue
            aux_task_modules = list(
                filter(
                    lambda param: aux_task in param[0].lower(),
                    named_modules,
                )
            )
            assert (
                len(aux_task_modules) >= 1
            ), f"Expected at least one module for auxiliary task '{aux_task}'."
            for module in aux_task_modules:
                named_modules.remove(module)

        modules_to_freeze = list(named_module[1] for named_module in named_modules)
        self.freeze(modules_to_freeze)

    def finetune_function(self, pl_module: LightningModule, epoch: int, optimizer: Optimizer):
        """Finetuning function called at the beginning of each training epoch.

        Args:
            pl_module: The LightningModule being finetuned.
            epoch: The current epoch number.
            optimizer: The optimizer used for finetuning.
        """
        pass


class FlowMatchingAuxiliaryTaskFinetuning(AuxiliaryTaskFinetuning):
    """A finetuning callback that freezes the entire flow matching model initially, including each
    of its modalities' interpolants, and then unfreezes only the auxiliary task heads for
    finetuning.

    Args:
        t_min (float): The minimum time value for flow matching interpolants. Default is 1.0.
        t_max (float): The maximum time value for flow matching interpolants. Default is 1.0.
    """

    def __init__(self, t_min: float = 1.0, t_max: float = 1.0):
        super().__init__()
        self.t_min = t_min
        self.t_max = t_max

    def freeze_before_training(self, pl_module: LightningModule):
        """Freeze the entire flow matching model initially, including each of its modalities'
        interpolants while excluding the auxiliary task heads.

        Args:
            pl_module: The LightningModule containing the flow matching
                model to be frozen.

        Raises:
            AttributeError: If the provided LightningModule is missing required attributes.
        """
        if not hasattr(pl_module, "model"):
            raise AttributeError(
                "LightningModule must have a 'model' attribute for FlowMatchingAuxiliaryTaskFinetuning."
            )
        if not hasattr(pl_module.model, "model"):
            raise AttributeError(
                "The model must have a 'model' attribute for FlowMatchingAuxiliaryTaskFinetuning."
            )
        if not hasattr(pl_module.model, "modals"):
            raise AttributeError(
                "The model must have a 'modals' attribute for FlowMatchingAuxiliaryTaskFinetuning."
            )
        if not all(
            hasattr(pl_module.model, f"{modal}_interpolant") for modal in pl_module.model.modals
        ):
            raise AttributeError(
                "The model must have corresponding modality interpolants for FlowMatchingAuxiliaryTaskFinetuning."
            )
        if not hasattr(pl_module.model, "auxiliary_tasks"):
            raise AttributeError(
                "The model must have an 'auxiliary_tasks' attribute for FlowMatchingAuxiliaryTaskFinetuning."
            )
        if not isinstance(pl_module.model.auxiliary_tasks, list):
            raise TypeError(
                "The model's 'auxiliary_tasks' must be a list of auxiliary task names."
            )

        # Freeze all model parameters except auxiliary task heads
        named_modules = list(pl_module.model.model.named_modules())[
            1:
        ]  # Skip the top-level module
        for aux_task in pl_module.model.auxiliary_tasks:
            global_property_pretraining = (
                pl_module.hparams.datasets["qm9"].global_property is None
                and pl_module.hparams.datasets["matbench"].proportion == 0.0
            )
            if aux_task == "global_property" and global_property_pretraining:
                # NOTE: This does not allow one to finetune on Matbench
                # for any task other than property prediction.
                continue
            # Check all datasets for global_energy, not just omol25
            if aux_task in ("global_energy", "atomic_forces"):
                # Check if any dataset has global_energy enabled
                has_global_energy = any(
                    hasattr(pl_module.hparams.datasets[dataset], "global_energy")
                    and pl_module.hparams.datasets[dataset].global_energy is not None
                    for dataset in pl_module.hparams.datasets
                )
                if not has_global_energy:
                    continue
            aux_task_modules = list(
                filter(
                    lambda param: aux_task in param[0].lower(),
                    named_modules,
                )
            )
            assert (
                len(aux_task_modules) >= 1
            ), f"Expected at least one module for auxiliary task '{aux_task}'."
            for module in aux_task_modules:
                named_modules.remove(module)

        modules_to_freeze = list(named_module[1] for named_module in named_modules)
        self.freeze(modules_to_freeze)

        # Set the minimum and maximum path times for each modality's interpolant
        for modal in pl_module.model.modals:
            interpolant = getattr(pl_module.model, f"{modal}_interpolant")
            interpolant.path_t_min = self.t_min
            interpolant.path_t_max = self.t_max

    def finetune_function(self, pl_module: LightningModule, epoch: int, optimizer: Optimizer):
        """Finetuning function called at the beginning of each training epoch.

        Args:
            pl_module: The LightningModule being finetuned.
            epoch: The current epoch number.
            optimizer: The optimizer used for finetuning.
        """
        pass
