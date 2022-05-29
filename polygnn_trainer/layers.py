from torch import nn
from .std_module import StandardModule
from .utils import get_unit_sequence


class my_hidden(StandardModule):
    """
    Hidden layer with xavier initialization and batch norm
    """

    def __init__(self, size_in, size_out, hps):
        super().__init__(hps)
        self.size_in, self.size_out = size_in, size_out
        self.linear = nn.Linear(self.size_in, self.size_out)
        nn.init.xavier_uniform_(self.linear.weight)
        self.activation = hps.activation.get_value()
        self.bn = nn.BatchNorm1d(self.size_out)

    def forward(self, x):
        if self.activation is not None:
            return self.bn(self.activation(self.linear(x)))
        else:
            return self.bn(self.linear(x))


class my_hidden2(StandardModule):
    """
    Hidden layer with xavier initialization and dropout
    """

    def __init__(self, size_in, size_out, hps):
        super().__init__(hps)
        self.size_in, self.size_out = size_in, size_out
        self.linear = nn.Linear(self.size_in, self.size_out)
        nn.init.xavier_uniform_(self.linear.weight)
        self.activation = hps.activation.get_value()
        self.dropout = nn.Dropout(hps.dropout_pct.get_value())

    def forward(self, x):
        if self.activation is not None:
            return self.dropout(self.activation(self.linear(x)))
        else:
            return self.dropout(self.linear(x))


class my_output(StandardModule):
    """
    Output layer with xavier initialization on weights
    Output layer with target mean (plus noise) on bias. Suggestion from: http://karpathy.github.io/2019/04/25/recipe/
    """

    def __init__(self, size_in, size_out, target_mean=None):
        super().__init__(None)
        self.size_in, self.size_out = size_in, size_out
        self.target_mean = target_mean

        self.linear = nn.Linear(self.size_in, self.size_out)
        nn.init.xavier_uniform_(self.linear.weight)
        if self.target_mean != None:
            self.linear.bias.data.uniform_(0.99 * target_mean, 1.01 * target_mean)

    def forward(self, x):
        return self.linear(x)


class Mlp(StandardModule):
    """
    A Feed-Forward neural Network that uses DenseHidden layers
    """

    def __init__(self, input_dim, output_dim, hps, debug, unit_sequence=None):
        super().__init__(hps)
        self.debug = debug
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layers = nn.ModuleList()
        if unit_sequence:
            self.unit_sequence = unit_sequence
            # if the unit_sequence is passed in then several other
            # attributes should be reset to be compatible with the
            # data in unit_sequence
            self.input_dim = self.unit_sequence[0]
            self.output_dim = self.unit_sequence[-1]
            self.hps.capacity.set_value(len(self.unit_sequence) - 2)
        else:
            self.unit_sequence = get_unit_sequence(
                input_dim, output_dim, self.hps.capacity.get_value()
            )
        # set up hidden layers
        for ind, n_units in enumerate(self.unit_sequence[:-1]):
            size_out_ = self.unit_sequence[ind + 1]
            self.layers.append(
                my_hidden2(
                    size_in=n_units,
                    size_out=size_out_,
                    hps=self.hps,
                )
            )

    def forward(self, x):
        """
        Compute the forward pass of this model
        """
        for layer in self.layers:
            x = layer(x)

        return x
