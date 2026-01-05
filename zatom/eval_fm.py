import os
import time
from typing import Any, Dict, List, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning.pytorch.loggers import Logger
from lightning.pytorch.strategies.strategy import Strategy
from omegaconf import DictConfig, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from zatom import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from zatom import (
    register_custom_omegaconf_resolvers,
    resolve_omegaconf_variable,
    set_omegaconf_flag_recursive,
)
from zatom.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg: DictConfig configuration composed by Hydra.

    Returns:
        A tuple containing two dictionaries - the first with evaluation metrics and the second with all instantiated objects.
    """
    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )

    # Avoid instantiating the `datasets` config as an object
    with open_dict(cfg):
        for dataset in cfg.model.datasets:
            cfg.model.datasets[dataset].pop("_target_")

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    plugins = None
    if "_target_" in cfg.environment:
        log.info(f"Instantiating environment <{cfg.environment._target_}>")
        plugins: ClusterEnvironment = hydra.utils.instantiate(cfg.environment)

    strategy = getattr(cfg.trainer, "strategy", None)
    if "_target_" in cfg.strategy:
        log.info(f"Instantiating strategy <{cfg.strategy._target_}>")
        strategy: Strategy = hydra.utils.instantiate(cfg.strategy)
        if (
            "mixed_precision" in strategy.__dict__
            and getattr(strategy, "mixed_precision", None) is not None
        ):
            strategy.mixed_precision.param_dtype = (
                resolve_omegaconf_variable(cfg.strategy.mixed_precision.param_dtype)
                if getattr(cfg.strategy.mixed_precision, "param_dtype", None) is not None
                else None
            )
            strategy.mixed_precision.reduce_dtype = (
                resolve_omegaconf_variable(cfg.strategy.mixed_precision.reduce_dtype)
                if getattr(cfg.strategy.mixed_precision, "reduce_dtype", None) is not None
                else None
            )
            strategy.mixed_precision.buffer_dtype = (
                resolve_omegaconf_variable(cfg.strategy.mixed_precision.buffer_dtype)
                if getattr(cfg.strategy.mixed_precision, "buffer_dtype", None) is not None
                else None
            )

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = (
        hydra.utils.instantiate(
            cfg.trainer,
            logger=logger,
            plugins=plugins,
            strategy=strategy,
        )
        if strategy is not None
        else hydra.utils.instantiate(
            cfg.trainer,
            logger=logger,
            plugins=plugins,
        )
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # Load checkpoint and update model state dict using only matching keys
    ckpt_path = cfg.ckpt_path
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        log.warning(
            "No valid checkpoint path provided. "
            "Will use untrained weights to perform model inference."
        )
    else:
        log.info(f"Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # nosec

        old_state_dict = checkpoint["state_dict"]
        new_state_dict = model.state_dict()

        updated_state_dict = {k: v for k, v in old_state_dict.items() if k in new_state_dict}
        model.load_state_dict(updated_state_dict, strict=False)

    if cfg.eval_split == "val":
        log.info("Starting validation!")
        trainer.validate(model=model, datamodule=datamodule)
    elif cfg.eval_split == "test":
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule)
    else:
        raise ValueError(
            f"Evaluation data split {cfg.eval_split} is not supported. Must be one of (`val`, `test`)."
        )

    # For predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_fm.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for foundation model evaluation.

    Args:
        cfg: DictConfig configuration composed by Hydra.
    """
    start_time = time.time()

    os.makedirs(cfg.paths.output_dir, exist_ok=True)

    # Apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # Set float32 matmul precision
    if cfg.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(cfg.float32_matmul_precision)
    if cfg.cuda_matmul_allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if cfg.cudnn_allow_tf32:
        torch.backends.cudnn.allow_tf32 = True

    # Run evaluation
    set_omegaconf_flag_recursive(
        cfg, "allow_objects", value=True
    )  # NOTE: Workaround for a Hydra issue: https://stackoverflow.com/q/69651138
    evaluate(cfg)

    # Report timing
    elapsed_time = time.time() - start_time
    log.info(f"Finished in {elapsed_time:.2f}s")


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
