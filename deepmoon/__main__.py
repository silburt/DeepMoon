from sys import exit

import fire
import dotenv
import hydra
from omegaconf import DictConfig

dotenv.load_dotenv(override=True)

@hydra.main(config_path="configs/", config_name="train.yaml")
def train(config: DictConfig):
    pass

exit(fire.Fire(train))
