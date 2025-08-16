import os
import time
from typing import Any, Dict, List, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning.pytorch.loggers import Logger
from lightning.pytorch.strategies.strategy import Strategy
from omegaconf import DictConfig

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

from zatom import register_custom_omegaconf_resolvers, resolve_omegaconf_variable
from zatom.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains an EBM model for generative modeling.

    Can additionally evaluate on a testset, using best weights obtained during training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg: A DictConfig configuration composed by Hydra.

    Returns:
        A tuple with metrics and dict with all instantiated objects.
    """
    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    # datamodule.setup()  # to save metadata the first time code is run

    log.info(f"Instantiating EBM model <{cfg.ebm_module._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.ebm_module)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(
        cfg.get("callbacks"), using_logger=(len(logger) > 0)
    )

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
            callbacks=callbacks,
            logger=logger,
            plugins=plugins,
            strategy=strategy,
        )
        if strategy is not None
        else hydra.utils.instantiate(
            cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            plugins=plugins,
        )
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # Merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_ebm.yaml")
def main(cfg: DictConfig) -> float | None:
    """Main entry point for energy-based model (EBM) training.

    Args:
        cfg: DictConfig configuration composed by Hydra.

    Returns:
        Optimized metric value if found, otherwise None.
    """
    start_time = time.time()

    os.makedirs(cfg.paths.output_dir, exist_ok=True)

    # Apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # Set float32 matmul precision
    if cfg.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(cfg.float32_matmul_precision)

    # Train the model
    metric_dict, _ = train(cfg)

    # Safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # Report timing
    elapsed_time = time.time() - start_time
    log.info(f"Finished in {elapsed_time:.2f}s")

    # Return optimized metric
    return metric_value


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
