from torch import nn, zeros, manual_seed, tensor
import warnings

from . import constants, infer, layers
from .utils import get_unit_sequence
from .std_module import StandardModule


class MlpOut(StandardModule):
    """
    This class represents an Mlp layer followed by a my_output layer.
    It extends the StandardModule class.
    """

    def __init__(self, input_dim: int, output_dim: int, hps, debug=False):
        """
        Initialize the MlpOut.

        Args:
            input_dim (int): Input dimension.
            output_dim (int): Output dimension.
            hps (HpConfig): Hyperparameters configuration.
            debug (bool, optional): Debug mode flag. Defaults to False.
        """
        super().__init__(hps)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hps = hps
        self.debug = debug

        self.unit_sequence = get_unit_sequence(
            self.input_dim, self.output_dim, self.hps.capacity.get_value()
        )

        self.mlp = layers.Mlp(
            None,
            None,
            self.hps,
            self.debug,
            self.unit_sequence[:-1],
        )

        self.outlayer = layers.my_output(self.unit_sequence[-2], self.unit_sequence[-1])

    def forward(self, data):
        """
        Perform forward pass through the model.

        Args:
            data: Input data.

        Returns:
            Tensor: Output tensor.
        """

        data.yhat = data.x
        data.yhat = self.assemble_data(data)
        data.yhat = self.mlp(data.yhat)

        return self.outlayer(data.yhat)


class LinearEnsemble(nn.Module):
    """
    An ensemble that takes a straight average of the predictions
    of its submodels
    """

    def __init__(self, submodel_dict, device, scalers):
        super(LinearEnsemble, self).__init__()
        self.submodel_dict = submodel_dict
        self.device = device
        self.scalers = scalers  # dictionary

    def forward(self, data, n_passes=None, monte_carlo=True):
        """
        Compute the forward pass of this model.

        Keyword arguments:
            n_passes: The number of forward passes to perform. This value
                is only meaningful when `monte_carlo` is set to `True`.
            monte_carlo: If True, Monte Carlo drop out is performed.
        """
        if monte_carlo:
            warn_msg = (
                "Monte Carlo (MC) drop out is turned on. As of version 0.2.0, "
                + "the current implementation of MC drop out is not recommended, "
                + "as it may lead to large errors in prediction. However, for "
                + "backwards compatibility, MC dropout is the default option. "
                + "You can turn it off by setting `monte_carlo` to `False` in "
                + "the forward method of this class."
            )
            warnings.warn(warn_msg)
            dropout_mode = "train"
        else:
            dropout_mode = "test"
        for _, model in self.submodel_dict.items():
            infer._model_eval_mode(model, dropout_mode=dropout_mode)
        if not monte_carlo and n_passes != None:
            warn_msg = (
                "The value you passed in for `n_passes` has been ignored "
                + f"since `monte_carlo` is set to `{monte_carlo}`."
            )
            warnings.warn(warn_msg)
        # The forward pass of ensembles should always return the prediction
        # *mean* and the prediction *standard deviation*
        manual_seed(constants.RANDOM_SEED)
        # We set the seed above so that all forward passes are reproducible
        batch_size = data.num_graphs
        n_submodels = len(self.submodel_dict)
        if monte_carlo:
            all_model_means = zeros((n_submodels, batch_size)).to(self.device)
            all_model_vars = zeros((n_submodels, batch_size)).to(self.device)
            # TODO: Parallelize?
            for i, model in self.submodel_dict.items():
                model_passes = zeros((n_passes, batch_size)).to(self.device)
                for j in range(n_passes):
                    output = model(data).view(
                        batch_size,
                    )
                    output = tensor(
                        [
                            self.scalers[str(ind_prop)].inverse_transform(val)
                            for (ind_prop, val) in zip(data.prop, output)
                        ]
                    ).view(
                        batch_size,
                    )
                    model_passes[j, :] = output

                all_model_means[i, :] = model_passes.mean(dim=0)
                all_model_vars[i, :] = model_passes.var(dim=0)

            # MC-ensemble uncertainty
            mean = all_model_means.mean(dim=0)
            var = (all_model_vars + all_model_means.square() - mean.square()).mean(
                dim=0
            )

            return mean.view(data.num_graphs,), var.sqrt().view(
                data.num_graphs,
            )
        else:
            all_model_preds = zeros((n_submodels, batch_size)).to(self.device)
            # TODO: Parallelize?
            for i, model in self.submodel_dict.items():
                output = model(data).view(
                    batch_size,
                )
                output = tensor(
                    [
                        self.scalers[str(ind_prop)].inverse_transform(val)
                        for (ind_prop, val) in zip(data.prop, output)
                    ]
                ).view(
                    batch_size,
                )
                all_model_preds[i] = output
            return (
                all_model_preds.mean(dim=0).squeeze(),
                all_model_preds.std(dim=0).squeeze(),
            )
