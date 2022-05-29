from torch import nn, zeros, manual_seed, tensor, cat

from . import constants
from polygnn_trainer import infer
from .utils import get_unit_sequence
from polygnn_trainer import layers
from .std_module import StandardModule


class MlpOut(StandardModule):
    """
    This is simply an Mlp layer followed by a my_output layer
    """

    def __init__(self, input_dim, output_dim, hps, debug=False):
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
        x = self.assemble_data(data)
        x = self.mlp(x)

        return self.outlayer(x)


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

    def forward(self, data, n_passes):
        """
        Compute the forward pass of this model
        """
        # The forward pass of ensembles should always return the prediction
        # *mean* and the prediction *standard deviation*
        manual_seed(constants.RANDOM_SEED)
        # We set the seed above so that all forward passes are reproducible
        batch_size = data.num_graphs
        n_submodels = len(self.submodel_dict)
        all_model_means = zeros((n_submodels, batch_size)).to(self.device)
        all_model_vars = zeros((n_submodels, batch_size)).to(self.device)
        # TODO: Parallelize?
        prop = data.prop
        for i, model in self.submodel_dict.items():
            model_passes = zeros((n_passes, batch_size)).to(self.device)
            infer._model_eval_mode(model, dropout_mode="train")
            for j in range(n_passes):
                output = model(data).view(
                    batch_size,
                )
                scaled_output = tensor(
                    [
                        self.scalers[str(ind_prop)].inverse_transform(val)
                        for (ind_prop, val) in zip(prop, output)
                    ]
                ).view(
                    batch_size,
                )
                model_passes[j, :] = scaled_output

            all_model_means[i, :] = model_passes.mean(dim=0)
            all_model_vars[i, :] = model_passes.var(dim=0)

        # MC-ensemble uncertainty
        mean = all_model_means.mean(dim=0)
        var = (all_model_vars + all_model_means.square() - mean.square()).mean(dim=0)

        return mean.view(data.num_graphs,), var.sqrt().view(
            data.num_graphs,
        )
