from torch import nn
import torch
import numpy as np
import random

# fix random seeds
random.seed(2)
torch.manual_seed(2)
np.random.seed(2)


class mh_mse_loss(nn.Module):
    """
    This function computes MSE loss and should be used to train models
    with a multiple heads (mh)

    Warning! This implementation needs to be tested.
    """

    def __init__(self):
        super().__init__()

    def forward(self, predictions, data):
        squared_error = (predictions - data.y).abs().square()

        return ((squared_error * data.selector).sum() / data.selector.sum()).sqrt()


class sh_mse_loss(nn.Module):
    """
    This function computes MSE loss and should be used to train models
    with a single head (sh)
    """

    def __init__(self):
        super().__init__()
        self.mse_fn = nn.MSELoss()

    def forward(self, predictions, data):
        # enforce the right shapes
        predictions = predictions.view(
            data.num_graphs,
        )
        data.y = data.y.view(
            data.num_graphs,
        )
        # ########################
        mse = self.mse_fn(predictions, data.y)

        return mse
