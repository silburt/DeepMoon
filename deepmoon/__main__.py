from os import environ
from sys import exit

import dotenv
import hydra
from omegaconf import DictConfig

dotenv.load_dotenv(override=True)

hydra_config_path = environ.get('DEEPMOONCONFIGPATH', "..")


@hydra.main(config_path=f"{hydra_config_path}/configs/",
            config_name="train.yml")
def train(config: DictConfig) -> None:

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from deepmoon.logger import extras

    extras(config)


exit(train())
