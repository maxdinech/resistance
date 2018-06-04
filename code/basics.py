"""
Basic PyTorch functions used in most of the other files.

"""


import os
import warnings

import torch

import architectures


def load_architecture(model_name):
    model = getattr(architectures, model_name)()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def load_model(dataset, model_name):
    try:
        path = os.path.join("..", "models", dataset, model_name + ".pt")
        with warnings.catch_warnings():
            # Ignores the compatibility warning between pytorch updates
            from torch.serialization import SourceChangeWarning
            warnings.simplefilter('ignore', SourceChangeWarning)
            model = torch.load(path, map_location=lambda storage, loc: storage)
        # if torch.cuda.is_available():
        #     model = model.cuda()
        return model
    except FileNotFoundError:
        raise ValueError('No trained model found.')
