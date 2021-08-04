from torch import (cat, add)

def merge(layers, cat_axis=None):
    return cat(layers, dim=cat_axis)

def passthrough(x, **kwargs):
    return x
