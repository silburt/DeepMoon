from torch.nn import (ReLU, PReLU, ELU)

def Activation(activation, nchan):
    if "elu" == activation:
        return ELU(inplace=True)
    elif "prelu" == activation:
        return PReLU(nchan)
    else:
        return ReLU(inplace=True)
