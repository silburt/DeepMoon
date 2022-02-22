from typing import List
from omegaconf import DictConfig

from pytorch_lightning import (LightningModule, LightningDataModule, Trainer,
                               Callback)
from pytorch_lightning.loggers import (LightningLoggerBase, wandb)


def log_hyperparameters(
    config: DictConfig,
    model: LightningModule,
    datamodule: LightningDataModule,
    trainer: Trainer,
    callbacks: List[Callback],
    logger: List[LightningLoggerBase],
) -> None:
    """Controls which config parts are saved by Lightning loggers.
    Additionaly saves:
    - number of model parameters
    """

    hparams = {}

    # choose which parts of hydra config are saved by loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(p.numel()
                                            for p in model.parameters()
                                            if p.requires_grad)
    hparams["model/params/non_trainable"] = sum(p.numel()
                                                for p in model.parameters()
                                                if not p.requires_grad)

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def finish(
    config: DictConfig,
    model: LightningModule,
    datamodule: LightningDataModule,
    trainer: Trainer,
    callbacks: List[Callback],
    logger: List[LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, wandb.WandbLogger):
            import wandb

            wandb.finish()
