from deepmoon.model.cratervnet import Crater_VNet
from deepmoon.model.deepmoon import DeepMoon

def get_model(model_name):
    return {
            "deepmoon": DeepMoon,
            "cratervnet": Crater_VNet
        }.get(model_name, None)